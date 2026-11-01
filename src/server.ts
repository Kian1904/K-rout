import fastifyStatic from '@fastify/static';
import Fastify, { type FastifyInstance } from 'fastify';
import { createHash, randomUUID, timingSafeEqual } from 'node:crypto';
import { resolve } from 'node:path';
import { ClientKeyStore } from './client-key-store.js';
import { ProviderConfigStore, type StoredProvider } from './provider-config-store.js';
import {
  type GeminiResponse,
  GeminiCompatibilityError,
  geminiEndpoint,
  readGeminiEventStream,
  toGeminiRequest,
  toOpenAiChunk,
  toOpenAiResponse,
  toOpenAiUsageChunk,
} from './gemini-adapter.js';

export type ProviderType = 'openai' | 'gemini';

export interface ProviderConfig {
  name: string;
  type?: ProviderType;
  baseUrl: string;
  apiKey: string;
  models: string[];
}

export interface RequestLog {
  id: string;
  timestamp: string;
  status: 'success' | 'error';
  provider: string;
  model: string;
  latencyMs: number;
  httpStatus: number;
}

export interface RouterConfig {
  providers: ProviderConfig[];
  adminApiKey: string;
  host?: string;
  port?: number;
  maxRequestLogs?: number;
  clientKeyStorePath?: string;
  providerConfigPath?: string;
  fetchImpl?: typeof fetch;
  serveDashboard?: boolean;
}

interface ChatCompletionBody {
  model?: unknown;
  messages?: unknown;
  stream?: unknown;
  [key: string]: unknown;
}

interface ProviderUpdateBody {
  type?: unknown;
  baseUrl?: unknown;
  apiKey?: unknown;
  models?: unknown;
}

interface ClientKeyCreateBody {
  name?: unknown;
}

const DEFAULT_LOG_LIMIT = 100;

function errorResponse(message: string, type = 'invalid_request_error') {
  return { error: { message, type } };
}

function safeProvider(provider: ProviderConfig) {
  return {
    name: provider.name,
    type: provider.type ?? 'openai',
    baseUrl: provider.baseUrl,
    models: [...provider.models],
    hasApiKey: Boolean(provider.apiKey),
  };
}

function readProviderUpdate(body: ProviderUpdateBody, current?: ProviderConfig): ProviderConfig | null {
  if (typeof body.baseUrl !== 'string' || !Array.isArray(body.models)) return null;
  if (body.apiKey !== undefined && typeof body.apiKey !== 'string') return null;
  let url: URL;
  try {
    url = new URL(body.baseUrl);
  } catch {
    return null;
  }
  if (url.username || url.password || url.search || url.hash) return null;
  const loopback = url.hostname === '127.0.0.1' || url.hostname === 'localhost' || url.hostname === '[::1]';
  if (url.protocol !== 'https:' && !(url.protocol === 'http:' && loopback)) return null;

  if (body.models.some((model) => typeof model !== 'string')) return null;
  const models = body.models.map((model) => (model as string).trim());
  if (models.some((model) => !model || model.length > 200)) return null;
  const apiKey = typeof body.apiKey === 'string' && body.apiKey.trim()
    ? body.apiKey.trim()
    : current?.apiKey;
  if (models.length === 0 || !apiKey) return null;
  if (new Set(models).size !== models.length) return null;

  const type = body.type ?? current?.type ?? 'openai';
  if (type !== 'openai' && type !== 'gemini') return null;
  return { name: current?.name ?? '', type, baseUrl: body.baseUrl.trim().replace(/\/+$/, ''), apiKey, models };
}

function validateProviderRegistry(input: ProviderConfig[]): ProviderConfig[] {
  const providers = input.map((provider) => ({
    ...provider,
    type: provider.type ?? 'openai',
    baseUrl: provider.baseUrl?.trim().replace(/\/+$/, ''),
    apiKey: provider.apiKey?.trim(),
    models: Array.isArray(provider.models) ? [...provider.models] : [],
  }));
  const ids = new Set<string>();
  const models = new Set<string>();
  for (const provider of providers) {
    if (!/^[A-Za-z0-9_-]{1,80}$/.test(provider.name)
      || (provider.type !== 'openai' && provider.type !== 'gemini')
      || !provider.apiKey
      || provider.models.length === 0
      || provider.models.some((model) => typeof model !== 'string' || !model || model !== model.trim() || model.length > 200)) {
      throw new Error('Provider configuration is invalid.');
    }
    let url: URL;
    try {
      url = new URL(provider.baseUrl);
    } catch {
      throw new Error('Provider configuration is invalid.');
    }
    if (!['https:', 'http:'].includes(url.protocol) || url.username || url.password || url.search || url.hash || ids.has(provider.name)) {
      throw new Error('Provider configuration is invalid.');
    }
    ids.add(provider.name);
    for (const model of provider.models) {
      if (models.has(model)) throw new Error('Provider model routes conflict.');
      models.add(model);
    }
    if (new Set(provider.models).size !== provider.models.length) throw new Error('Provider model routes conflict.');
  }
  return providers;
}

function toStoredProvider(provider: ProviderConfig): StoredProvider {
  return {
    name: provider.name,
    type: provider.type ?? 'openai',
    baseUrl: provider.baseUrl,
    apiKey: provider.apiKey,
    models: [...provider.models],
  };
}

async function withProviderMutation<T>(operation: () => Promise<T>): Promise<T> {
  let release!: () => void;
  const previous = providerMutationQueue;
  providerMutationQueue = new Promise<void>((resolveQueue) => { release = resolveQueue; });
  await previous;
  try {
    return await operation();
  } finally {
    release();
  }
}

let providerMutationQueue = Promise.resolve();

export function createApp(config: RouterConfig): FastifyInstance {
  if (!config.adminApiKey) throw new Error('ADMIN_API_KEY is required.');

  const providers = new Map<string, ProviderConfig>();
  const requestLogs: RequestLog[] = [];
  const logLimit = Math.max(1, config.maxRequestLogs ?? DEFAULT_LOG_LIMIT);
  const startedAt = Date.now();
  const fetchImpl = config.fetchImpl ?? fetch;
  const adminKeyHash = createHash('sha256').update(config.adminApiKey).digest();
  const clientKeyStore = new ClientKeyStore(config.clientKeyStorePath ?? resolve(process.cwd(), '.data/client-keys.json'));
  const providerConfigPath = config.providerConfigPath ?? process.env.PROVIDER_CONFIG_PATH ?? '.data/providers.json';
  const providerConfigStore = new ProviderConfigStore(resolve(process.cwd(), providerConfigPath));
  const app = Fastify({ logger: false });
  app.addHook('onReady', async () => {
    const savedProviders = await providerConfigStore.load();
    const initialProviders = validateProviderRegistry(savedProviders ?? config.providers);
    providers.clear();
    for (const provider of initialProviders) providers.set(provider.name, provider);
    await clientKeyStore.initialize();
  });

  const models = () => [...providers.values()].flatMap((provider) =>
    provider.models.map((model) => ({ model, provider: provider.name })),
  );
  const recordRequest = (entry: Omit<RequestLog, 'id' | 'timestamp'>) => {
    requestLogs.push({
      id: randomUUID(),
      timestamp: new Date().toISOString(),
      ...entry,
    });
    if (requestLogs.length > logLimit) requestLogs.splice(0, requestLogs.length - logLimit);
  };
  const findProviderForModel = (model: string) => {
    for (const provider of providers.values()) {
      if (provider.models.includes(model)) return provider;
    }
    return undefined;
  };

  app.get('/health', async () => ({ status: 'ok' }));

  app.post<{ Body: ChatCompletionBody }>('/v1/chat/completions', async (request, reply) => {
    const authorization = request.headers.authorization ?? '';
    const clientKey = authorization.startsWith('Bearer ') ? authorization.slice(7) : '';
    if (!await clientKeyStore.authenticate(clientKey)) {
      return reply.code(401).send(errorResponse('A valid client API key is required.', 'authentication_error'));
    }
    const started = Date.now();
    const body = request.body;
    if (!body || typeof body !== 'object' || typeof body.model !== 'string' || !Array.isArray(body.messages)) {
      return reply.code(400).send(errorResponse('Expected a model string and messages array.'));
    }
    if (body.stream !== undefined && typeof body.stream !== 'boolean') {
      return reply.code(400).send(errorResponse('The stream field must be a boolean.'));
    }
    const provider = findProviderForModel(body.model);
    if (!provider) {
      recordRequest({ status: 'error', provider: '—', model: body.model, latencyMs: Date.now() - started, httpStatus: 404 });
      return reply.code(404).send(errorResponse(`No provider is configured for model "${body.model}".`, 'model_not_found'));
    }

    const providerType = provider.type ?? 'openai';
    let upstreamBody: string;
    let upstreamUrl: string;
    try {
      if (providerType === 'gemini') {
        upstreamBody = JSON.stringify(toGeminiRequest(body));
        upstreamUrl = geminiEndpoint(provider.baseUrl, body.model, body.stream === true);
      } else {
        upstreamBody = JSON.stringify(body);
        upstreamUrl = provider.baseUrl.replace(/\/+$/, '') + '/chat/completions';
      }
    } catch (error) {
      const message = error instanceof GeminiCompatibilityError
        ? error.message
        : 'The request could not be translated for this provider.';
      recordRequest({ status: 'error', provider: provider.name, model: body.model, latencyMs: Date.now() - started, httpStatus: 400 });
      return reply.code(400).send(errorResponse(message));
    }

    const abortController = new AbortController();
    const abortUpstream = () => {
      if (!reply.raw.writableEnded) abortController.abort();
    };
    request.raw.once('aborted', abortUpstream);
    reply.raw.once('close', abortUpstream);

    let upstream: Response;
    try {
      upstream = await fetchImpl(upstreamUrl, {
        method: 'POST',
        headers: {
          ...(providerType === 'gemini'
            ? { 'x-goog-api-key': provider.apiKey }
            : { authorization: `Bearer ${provider.apiKey}` }),
          'content-type': 'application/json',
          accept: body.stream === true ? 'text/event-stream' : 'application/json',
        },
        body: upstreamBody,
        signal: abortController.signal,
      });
    } catch {
      if (abortController.signal.aborted) return reply;
      const latencyMs = Date.now() - started;
      recordRequest({ status: 'error', provider: provider.name, model: body.model, latencyMs, httpStatus: 502 });
      return reply.code(502).send(errorResponse('The upstream provider could not be reached.', 'upstream_error'));
    }

    if (!upstream.ok) {
      const latencyMs = Date.now() - started;
      recordRequest({ status: 'error', provider: provider.name, model: body.model, latencyMs, httpStatus: 502 });
      return reply.code(502).send(errorResponse(`The upstream provider returned HTTP ${upstream.status}.`, 'upstream_error'));
    }

    if (body.stream !== true) {
      const contentType = upstream.headers.get('content-type');
      let responseBody: string | object;
      if (providerType === 'gemini') {
        try {
          responseBody = toOpenAiResponse(await upstream.json(), body.model);
        } catch {
          recordRequest({ status: 'error', provider: provider.name, model: body.model, latencyMs: Date.now() - started, httpStatus: 502 });
          return reply.code(502).send(errorResponse('The upstream provider returned an invalid response.', 'upstream_error'));
        }
      } else {
        if (contentType) reply.header('content-type', contentType);
        responseBody = await upstream.text();
      }
      recordRequest({ status: 'success', provider: provider.name, model: body.model, latencyMs: Date.now() - started, httpStatus: upstream.status });
      return reply.code(upstream.status).send(responseBody);
    }

    if (!upstream.body) {
      recordRequest({ status: 'error', provider: provider.name, model: body.model, latencyMs: Date.now() - started, httpStatus: 502 });
      return reply.code(502).send(errorResponse('The upstream provider returned an empty stream.', 'upstream_error'));
    }

    if (providerType === 'gemini') {
      reply.hijack();
      reply.raw.writeHead(upstream.status, {
        'content-type': 'text/event-stream; charset=utf-8',
        'cache-control': 'no-cache, no-transform',
        connection: 'keep-alive',
        'x-accel-buffering': 'no',
      });
      const id = 'chatcmpl-' + randomUUID();
      const created = Math.floor(Date.now() / 1000);
      const includeUsage = Boolean((body.stream_options as { include_usage?: unknown } | undefined)?.include_usage);
      let includeRole = true;
      let lastGeminiResponse: GeminiResponse | undefined;
      let streamFailed = false;
      const writeEvent = async (event: unknown) => {
        if (reply.raw.destroyed) return false;
        if (reply.raw.write('data: ' + JSON.stringify(event) + '\n\n')) return true;
        await new Promise<void>((resolveDrain) => {
          const finish = () => {
            reply.raw.off('drain', finish);
            reply.raw.off('close', finish);
            reply.raw.off('error', finish);
            resolveDrain();
          };
          reply.raw.once('drain', finish);
          reply.raw.once('close', finish);
          reply.raw.once('error', finish);
        });
        return !reply.raw.destroyed;
      };
      try {
        for await (const chunk of readGeminiEventStream(upstream.body)) {
          lastGeminiResponse = chunk;
          if (!await writeEvent(toOpenAiChunk(chunk, body.model, id, created, includeRole))) break;
          includeRole = false;
        }
        if (includeRole) {
          await writeEvent(toOpenAiChunk({}, body.model, id, created, true));
          await writeEvent(toOpenAiChunk({ candidates: [{ finishReason: 'STOP' }] }, body.model, id, created, false));
        }
        if (includeUsage && lastGeminiResponse) {
          const usageChunk = toOpenAiUsageChunk(lastGeminiResponse, body.model, id, created);
          if (usageChunk) await writeEvent(usageChunk);
        }
        if (!reply.raw.destroyed) reply.raw.write('data: [DONE]\n\n');
      } catch {
        streamFailed = true;
      } finally {
        if (!reply.raw.destroyed) reply.raw.end();
        recordRequest({
          status: streamFailed ? 'error' : 'success',
          provider: provider.name,
          model: body.model,
          latencyMs: Date.now() - started,
          httpStatus: streamFailed ? 502 : upstream.status,
        });
      }
      return reply;
    }

    reply.hijack();
    reply.raw.writeHead(upstream.status, {
      'content-type': upstream.headers.get('content-type') ?? 'text/event-stream; charset=utf-8',
      'cache-control': 'no-cache, no-transform',
      connection: 'keep-alive',
      'x-accel-buffering': 'no',
    });
    const reader = upstream.body.getReader();
    let streamFailed = false;
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (!reply.raw.write(value)) await new Promise<void>((resolveDrain) => reply.raw.once('drain', resolveDrain));
      }
    } catch {
      streamFailed = true;
    } finally {
      reader.releaseLock();
      reply.raw.end();
      recordRequest({
        status: streamFailed ? 'error' : 'success',
        provider: provider.name,
        model: body.model,
        latencyMs: Date.now() - started,
        httpStatus: streamFailed ? 502 : upstream.status,
      });
    }
    return reply;
  });

  app.register(async (admin) => {
    admin.addHook('preHandler', async (request, reply) => {
      reply.header('cache-control', 'no-store');
      reply.header('pragma', 'no-cache');
      const authorization = request.headers.authorization ?? '';
      const token = authorization.startsWith('Bearer ') ? authorization.slice(7) : '';
      const suppliedHash = createHash('sha256').update(token).digest();
      if (!timingSafeEqual(suppliedHash, adminKeyHash)) {
        return reply.code(401).send(errorResponse('Admin authentication required.', 'authentication_error'));
      }
    });

    admin.get('/overview', async () => ({
      status: 'ok',
      uptimeSeconds: Math.floor((Date.now() - startedAt) / 1000),
      providerCount: providers.size,
      modelCount: models().length,
      clientKeyCount: await clientKeyStore.countActive(),
      requestCount: requestLogs.length,
      requestLogLimit: logLimit,
    }));

    admin.get('/providers', async () => ({ providers: [...providers.values()].map(safeProvider) }));

    admin.get('/client-keys', async () => ({ clientKeys: await clientKeyStore.list() }));

    admin.post<{ Body: ClientKeyCreateBody }>('/client-keys', async (request, reply) => {
      if (typeof request.body?.name !== 'string') {
        return reply.code(400).send(errorResponse('A client key name is required.'));
      }
      try {
        return reply.code(201).send(await clientKeyStore.create(request.body.name));
      } catch {
        return reply.code(400).send(errorResponse('Client key name must be between 1 and 80 characters.'));
      }
    });

    admin.delete<{ Params: { id: string } }>('/client-keys/:id', async (request, reply) => {
      if (!await clientKeyStore.revoke(request.params.id)) {
        return reply.code(404).send(errorResponse('Active client key not found.', 'not_found'));
      }
      return reply.code(204).send();
    });

    admin.put<{ Params: { id: string }; Body: ProviderUpdateBody }>('/providers/:id', async (request, reply) => {
      const id = request.params.id;
      if (!/^[a-zA-Z0-9_-]+$/.test(id)) return reply.code(400).send(errorResponse('Invalid provider id.'));
      return withProviderMutation(async () => {
        const current = providers.get(id);
        const updated = readProviderUpdate(request.body ?? {}, current);
        if (!updated) return reply.code(400).send(errorResponse('Provide a valid HTTPS base URL, provider type, API key, and model list.'));
        updated.name = id;
        const next = [...providers.values()].filter((provider) => provider.name !== id).concat(updated);
        try {
          const validated = validateProviderRegistry(next);
          await providerConfigStore.save(validated.map(toStoredProvider));
          providers.clear();
          for (const provider of validated) providers.set(provider.name, provider);
        } catch (error) {
          if (error instanceof Error && error.message === 'Provider model routes conflict.') {
            return reply.code(409).send(errorResponse('Each model can be assigned to only one provider.'));
          }
          return reply.code(500).send(errorResponse('Provider configuration could not be saved.', 'storage_error'));
        }
        return { provider: safeProvider(updated) };
      });
    });

    admin.delete<{ Params: { id: string } }>('/providers/:id', async (request, reply) => {
      return withProviderMutation(async () => {
        const id = request.params.id;
        if (!providers.has(id)) return reply.code(404).send(errorResponse('Provider not found.', 'not_found'));
        const next = [...providers.values()].filter((provider) => provider.name !== id);
        try {
          await providerConfigStore.save(next.map(toStoredProvider));
        } catch {
          return reply.code(500).send(errorResponse('Provider configuration could not be saved.', 'storage_error'));
        }
        providers.delete(id);
        return reply.code(204).send();
      });
    });

    admin.get('/models', async () => ({ models: models() }));
    admin.get('/requests', async () => ({ requests: [...requestLogs].reverse() }));
    admin.get('/settings', async () => ({
      host: config.host ?? '127.0.0.1',
      port: config.port ?? 3000,
      adminAuthentication: 'enabled',
      providerSecrets: 'server-only',
      clientKeyStorage: 'local file with restricted permissions',
      requestLogLimit: logLimit,
      uptimeSeconds: Math.floor((Date.now() - startedAt) / 1000),
    }));
  }, { prefix: '/admin' });

  if (config.serveDashboard !== false) {
    app.get('/dashboard', async (_request, reply) => reply.redirect('/dashboard/'));
    app.register(fastifyStatic, {
      root: resolve(process.cwd(), 'dist/dashboard'),
      prefix: '/dashboard/',
      decorateReply: false,
    });
  }

  return app;
}

function requiredEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`Missing required environment variable: ${name}`);
  return value;
}

function providersFromEnv(): ProviderConfig[] {
  const ids = requiredEnv('PROVIDER_IDS').split(',').map((id) => id.trim()).filter(Boolean);
  if (ids.length === 0) throw new Error('PROVIDER_IDS must contain at least one provider id.');
  return ids.map((id) => {
    if (!/^[a-zA-Z0-9_-]+$/.test(id)) throw new Error(`Invalid provider id: ${id}`);
    const prefix = `PROVIDER_${id.toUpperCase().replace(/[^A-Z0-9]/g, '_')}`;
    const configuredType = process.env[`${prefix}_TYPE`]?.trim().toLowerCase() || 'openai';
    if (configuredType !== 'openai' && configuredType !== 'gemini') throw new Error(`${prefix}_TYPE must be "openai" or "gemini".`);
    const modelList = requiredEnv(`${prefix}_MODELS`).split(',').map((model) => model.trim()).filter(Boolean);
    if (modelList.length === 0) throw new Error(`${prefix}_MODELS must contain at least one model.`);
    const baseUrl = requiredEnv(`${prefix}_BASE_URL`);
    let parsedBaseUrl: URL;
    try {
      parsedBaseUrl = new URL(baseUrl);
    } catch {
      throw new Error(`${prefix}_BASE_URL must be a valid URL.`);
    }
    if (parsedBaseUrl.username || parsedBaseUrl.password || parsedBaseUrl.search || parsedBaseUrl.hash) {
      throw new Error(`${prefix}_BASE_URL must not contain credentials, query parameters, or fragments.`);
    }
    return {
      name: id,
      type: configuredType,
      baseUrl,
      apiKey: requiredEnv(`${prefix}_API_KEY`),
      models: modelList,
    };
  });
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const port = Number(process.env.PORT ?? 3000);
  const host = process.env.HOST ?? '127.0.0.1';
  const app = createApp({
    providers: providersFromEnv(),
    adminApiKey: requiredEnv('ADMIN_API_KEY'),
    host,
    port,
  });
  await app.listen({ port, host });
}
