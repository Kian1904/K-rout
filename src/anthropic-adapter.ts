import { randomUUID } from 'node:crypto';

export class AnthropicCompatibilityError extends Error {}

type ChatMessage = { role?: unknown; content?: unknown; tool_calls?: unknown; function_call?: unknown };
type ChatRequest = {
  model?: unknown;
  messages?: unknown;
  system?: unknown;
  temperature?: unknown;
  top_p?: unknown;
  top_k?: unknown;
  max_tokens?: unknown;
  max_completion_tokens?: unknown;
  stop?: unknown;
  stream?: unknown;
  tools?: unknown;
  tool_choice?: unknown;
  functions?: unknown;
  function_call?: unknown;
  response_format?: unknown;
  logprobs?: unknown;
  logit_bias?: unknown;
  n?: unknown;
  [key: string]: unknown;
};

export type AnthropicResponse = {
  id?: string;
  type?: string;
  role?: string;
  model?: string;
  content?: Array<{ type?: string; text?: string }>;
  stop_reason?: string | null;
  stop_sequence?: string | null;
  usage?: { input_tokens?: number; output_tokens?: number };
  error?: unknown;
};

const DEFAULT_MAX_TOKENS = 4096;

function textFromContent(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content.map((part) => {
      if (!part || typeof part !== 'object' || (part as { type?: unknown }).type !== 'text' || typeof (part as { text?: unknown }).text !== 'string') {
        throw new AnthropicCompatibilityError('Anthropic adapter supports text-only message content.');
      }
      return (part as { text: string }).text;
    }).join('');
  }
  throw new AnthropicCompatibilityError('Anthropic adapter requires message content to be text.');
}

function unsupported(request: ChatRequest) {
  if (request.tools !== undefined || request.tool_choice !== undefined) throw new AnthropicCompatibilityError('Anthropic adapter does not support [OI] tools or tool_choice yet.');
  if (request.functions !== undefined || request.function_call !== undefined) throw new AnthropicCompatibilityError('Anthropic adapter does not support [OI] function calling yet.');
  if (request.response_format !== undefined) throw new AnthropicCompatibilityError('Anthropic adapter does not support response_format yet.');
  if (request.logprobs !== undefined) throw new AnthropicCompatibilityError('Anthropic adapter does not support logprobs.');
  if (request.logit_bias !== undefined) throw new AnthropicCompatibilityError('Anthropic adapter does not support logit_bias.');
  if (request.n !== undefined && request.n !== 1) throw new AnthropicCompatibilityError('Anthropic adapter supports one completion choice per request.');
}

export function anthropicEndpoint(baseUrl: string): string {
  const root = baseUrl.replace(/\/+$/, '');
  return root.endsWith('/v1') ? `${root}/messages` : `${root}/v1/messages`;
}

export function toAnthropicRequest(input: Record<string, unknown>): Record<string, unknown> {
  const request = input as ChatRequest;
  unsupported(request);
  if (!Array.isArray(request.messages)) throw new AnthropicCompatibilityError('Anthropic adapter requires a messages array.');

  const systemParts: string[] = [];
  const contents: Array<{ role: 'user' | 'assistant'; content: string }> = [];
  for (const rawMessage of request.messages) {
    if (!rawMessage || typeof rawMessage !== 'object') throw new AnthropicCompatibilityError('Each message must be an object.');
    const message = rawMessage as ChatMessage;
    if (message.tool_calls !== undefined || message.function_call !== undefined) {
      throw new AnthropicCompatibilityError('Anthropic adapter does not support tool-call messages yet.');
    }
    const text = textFromContent(message.content);
    if (message.role === 'system' || message.role === 'developer') {
      systemParts.push(text);
      continue;
    }
    if (message.role !== 'user' && message.role !== 'assistant') {
      throw new AnthropicCompatibilityError('Anthropic adapter does not support this message role.');
    }
    const role = message.role === 'assistant' ? 'assistant' : 'user';
    const previous = contents[contents.length - 1];
    if (previous && previous.role === role) {
      previous.content += `\n\n${text}`;
      continue;
    }
    contents.push({ role, content: text });
  }
  if (!contents.length) throw new AnthropicCompatibilityError('At least one user or assistant message is required.');
  if (!contents[0].role || contents[0].role !== 'user') throw new AnthropicCompatibilityError('Anthropic requires the conversation to start with a user message.');

  const body: Record<string, unknown> = {
    model: request.model,
    messages: contents,
    max_tokens: request.max_tokens ?? request.max_completion_tokens ?? DEFAULT_MAX_TOKENS,
  };
  if (systemParts.length) body.system = systemParts.join('\n\n');
  if (request.temperature !== undefined) body.temperature = request.temperature;
  if (request.top_p !== undefined) body.top_p = request.top_p;
  if (request.top_k !== undefined) body.top_k = request.top_k;
  if (request.stop !== undefined) body.stop_sequences = Array.isArray(request.stop) ? request.stop : [request.stop];
  return body;
}

function finishReason(reason?: string | null): 'stop' | 'length' {
  if (reason === 'max_tokens') return 'length';
  return 'stop';
}

export function toOpenAiResponse(response: AnthropicResponse, requestedModel: string) {
  if (response.error) throw new AnthropicCompatibilityError('Anthropic returned an error payload.');
  const text = (response.content ?? [])
    .filter((part) => part?.type === 'text' && typeof part.text === 'string')
    .map((part) => part.text)
    .join('');
  const usage = response.usage
    ? {
        prompt_tokens: response.usage.input_tokens ?? 0,
        completion_tokens: response.usage.output_tokens ?? 0,
        total_tokens: (response.usage.input_tokens ?? 0) + (response.usage.output_tokens ?? 0),
      }
    : undefined;
  return {
    id: `chatcmpl-${response.id ?? randomUUID()}`,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model: response.model ?? requestedModel,
    choices: [{
      index: 0,
      message: { role: 'assistant', content: text },
      finish_reason: finishReason(response.stop_reason),
    }],
    ...(usage ? { usage } : {}),
  };
}

type AnthropicStreamEvent = {
  type?: string;
  message?: AnthropicResponse;
  delta?: { type?: string; text?: string; stop_reason?: string | null };
  usage?: { input_tokens?: number; output_tokens?: number };
};

export type OpenAiStreamChunk = {
  id: string;
  object: 'chat.completion.chunk';
  created: number;
  model: string;
  choices: Array<{ index: number; delta: Record<string, unknown>; finish_reason: string | null }>;
  usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
};

export function openAiChunk(
  model: string,
  id: string,
  created: number,
  delta: Record<string, unknown>,
  finish: string | null = null,
): OpenAiStreamChunk {
  return {
    id,
    object: 'chat.completion.chunk',
    created,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  };
}

export async function* readAnthropicEventStream(body: ReadableStream<Uint8Array>): AsyncGenerator<AnthropicStreamEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const parseEvent = (event: string): AnthropicStreamEvent | undefined => {
    const data = event.split(/\r?\n/).filter((line) => line.startsWith('data:')).map((line) => line.slice(5).trimStart()).join('\n');
    if (!data || data === '[DONE]') return undefined;
    return JSON.parse(data) as AnthropicStreamEvent;
  };
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      let separatorIndex: number;
      while ((separatorIndex = buffer.search(/\r?\n\r?\n/)) >= 0) {
        const separator = buffer.slice(separatorIndex).match(/^\r?\n\r?\n/)?.[0] ?? '\n\n';
        const event = buffer.slice(0, separatorIndex);
        buffer = buffer.slice(separatorIndex + separator.length);
        const parsed = parseEvent(event);
        if (parsed) yield parsed;
      }
      if (done) break;
    }
    if (buffer.trim()) {
      const parsed = parseEvent(buffer);
      if (parsed) yield parsed;
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Translates the Anthropic SSE event stream into [OI] chat completion chunks.
 * Yields chunks only; usage is reported through the returned usage accumulator.
 */
export async function* anthropicStreamChunks(
  body: ReadableStream<Uint8Array>,
  requestedModel: string,
  id: string,
  created: number,
  includeUsage: boolean,
): AsyncGenerator<OpenAiStreamChunk> {
  let model = requestedModel;
  let roleSent = false;
  let usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number } | undefined;
  let finish: string | null = null;

  for await (const event of readAnthropicEventStream(body)) {
    if (event.type === 'error') throw new AnthropicCompatibilityError('Anthropic returned an error payload.');
    if (event.type === 'message_start') {
      const message = event.message;
      if (message?.model) model = message.model;
      if (message?.usage) {
        usage = {
          prompt_tokens: message.usage.input_tokens ?? 0,
          completion_tokens: message.usage.output_tokens ?? 0,
          total_tokens: (message.usage.input_tokens ?? 0) + (message.usage.output_tokens ?? 0),
        };
      }
      if (!roleSent) {
        roleSent = true;
        yield openAiChunk(model, id, created, { role: 'assistant' });
      }
      continue;
    }
    if (event.type === 'content_block_delta') {
      const text = event.delta?.text;
      if (typeof text === 'string' && text) {
        if (!roleSent) {
          roleSent = true;
          yield openAiChunk(model, id, created, { role: 'assistant' });
        }
        yield openAiChunk(model, id, created, { content: text });
      }
      continue;
    }
    if (event.type === 'message_delta') {
      if (event.delta?.stop_reason) finish = finishReason(event.delta.stop_reason);
      if (event.usage) {
        const prompt = usage?.prompt_tokens ?? event.usage.input_tokens ?? 0;
        const completion = event.usage.output_tokens ?? usage?.completion_tokens ?? 0;
        usage = { prompt_tokens: prompt, completion_tokens: completion, total_tokens: prompt + completion };
      }
      continue;
    }
  }

  if (!roleSent) yield openAiChunk(model, id, created, { role: 'assistant' });
  yield openAiChunk(model, id, created, {}, finish ?? 'stop');
  if (includeUsage && usage) {
    yield {
      id,
      object: 'chat.completion.chunk',
      created,
      model,
      choices: [],
      usage,
    };
  }
}
