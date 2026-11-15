import {
  AnthropicCompatibilityError,
  anthropicEndpoint,
  anthropicStreamChunks,
  toAnthropicRequest,
  toOpenAiResponse as anthropicToOpenAiResponse,
} from './anthropic-adapter.js';
import {
  GeminiCompatibilityError,
  geminiEndpoint,
  geminiStreamChunks,
  toGeminiRequest,
  toOpenAiResponse as geminiToOpenAiResponse,
} from './gemini-adapter.js';

export const ADAPTER_TYPES = ['openai-compatible', 'gemini', 'anthropic'] as const;
export type AdapterType = typeof ADAPTER_TYPES[number];

export type AdapterHeaders = Record<string, string>;

export type BuildRequest = {
  url: string;
  headers: AdapterHeaders;
  body: Record<string, unknown>;
};

export type AdapterDefinition = {
  type: AdapterType;
  label: string;
  defaultBaseUrl?: string;
  buildRequest: (baseUrl: string, request: Record<string, unknown>, apiKey: string) => BuildRequest;
  normalizeResponse?: (response: unknown, requestedModel: string) => unknown;
  streamChunks?: (
    body: ReadableStream<Uint8Array>,
    requestedModel: string,
    id: string,
    created: number,
    includeUsage: boolean,
  ) => AsyncGenerator<unknown>;
};

const openAiAdapter: AdapterDefinition = {
  type: 'openai-compatible',
  label: '[OI]-compatible',
  buildRequest(baseUrl, request, apiKey) {
    return {
      url: `${baseUrl.replace(/\/+$/, '')}/chat/completions`,
      headers: { authorization: `Bearer ${apiKey}`, 'content-type': 'application/json' },
      body: request,
    };
  },
};

const geminiAdapter: AdapterDefinition = {
  type: 'gemini',
  label: 'Google Gemini API',
  defaultBaseUrl: 'https://generativelanguage.googleapis.com/v1beta',
  buildRequest(baseUrl, request, apiKey) {
    const model = String(request.model ?? '');
    const streaming = request.stream === true;
    return {
      url: geminiEndpoint(baseUrl, model, streaming),
      headers: { 'x-goog-api-key': apiKey, 'content-type': 'application/json' },
      body: toGeminiRequest(request),
    };
  },
  normalizeResponse(response, requestedModel) {
    return geminiToOpenAiResponse(response as never, requestedModel);
  },
  streamChunks: geminiStreamChunks,
};

const anthropicAdapter: AdapterDefinition = {
  type: 'anthropic',
  label: 'Anthropic Messages',
  defaultBaseUrl: 'https://api.anthropic.com/v1',
  buildRequest(baseUrl, request, apiKey) {
    const body = toAnthropicRequest(request);
    if (request.stream === true) body.stream = true;
    return {
      url: anthropicEndpoint(baseUrl),
      headers: {
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
      },
      body,
    };
  },
  normalizeResponse(response, requestedModel) {
    return anthropicToOpenAiResponse(response as never, requestedModel);
  },
  streamChunks: anthropicStreamChunks,
};

const registry = new Map<AdapterType, AdapterDefinition>([
  ['openai-compatible', openAiAdapter],
  ['gemini', geminiAdapter],
  ['anthropic', anthropicAdapter],
]);

export function getAdapter(type: string): AdapterDefinition | undefined {
  return registry.get(type as AdapterType);
}

export function requireAdapter(type: string): AdapterDefinition {
  const adapter = getAdapter(type);
  if (!adapter) throw new Error(`Unsupported provider type: ${type}`);
  return adapter;
}

export function isAdapterType(type: string): type is AdapterType {
  return registry.has(type as AdapterType);
}

export function listAdapters(): Array<{ type: AdapterType; label: string; defaultBaseUrl?: string }> {
  return [...registry.values()].map((adapter) => ({
    type: adapter.type,
    label: adapter.label,
    ...(adapter.defaultBaseUrl ? { defaultBaseUrl: adapter.defaultBaseUrl } : {}),
  }));
}

/**
 * Returns a client-facing compatibility error for adapter translation failures.
 */
export function toCompatibilityError(error: unknown): Error | undefined {
  if (error instanceof GeminiCompatibilityError || error instanceof AnthropicCompatibilityError) return error;
  return undefined;
}

export { GeminiCompatibilityError, AnthropicCompatibilityError };
