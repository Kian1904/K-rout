import { randomUUID } from 'node:crypto';

export class GeminiCompatibilityError extends Error {}

type ChatMessage = { role?: unknown; content?: unknown; tool_calls?: unknown; function_call?: unknown };
type ChatRequest = {
  model?: unknown;
  messages?: unknown;
  temperature?: unknown;
  top_p?: unknown;
  top_k?: unknown;
  max_tokens?: unknown;
  max_completion_tokens?: unknown;
  stop?: unknown;
  presence_penalty?: unknown;
  frequency_penalty?: unknown;
  seed?: unknown;
  n?: unknown;
  tools?: unknown;
  tool_choice?: unknown;
  functions?: unknown;
  function_call?: unknown;
  response_format?: unknown;
  logprobs?: unknown;
  logit_bias?: unknown;
  [key: string]: unknown;
};

type GeminiCandidate = {
  index?: number;
  content?: { parts?: Array<{ text?: string; thought?: boolean }> };
  finishReason?: string;
};

export type GeminiResponse = {
  error?: unknown;
  responseId?: string;
  modelVersion?: string;
  candidates?: GeminiCandidate[];
  usageMetadata?: {
    promptTokenCount?: number;
    candidatesTokenCount?: number;
    totalTokenCount?: number;
  };
  promptFeedback?: { blockReason?: string };
};

function textFromContent(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content.map((part) => {
      if (!part || typeof part !== 'object' || (part as { type?: unknown }).type !== 'text' || typeof (part as { text?: unknown }).text !== 'string') {
        throw new GeminiCompatibilityError('Gemini adapter supports text-only message content.');
      }
      return (part as { text: string }).text;
    }).join('');
  }
  throw new GeminiCompatibilityError('Gemini adapter requires message content to be text.');
}

function unsupported(request: ChatRequest) {
  if (request.tools !== undefined || request.tool_choice !== undefined) throw new GeminiCompatibilityError('Gemini adapter does not support OpenAI tools or tool_choice yet.');
  if (request.functions !== undefined || request.function_call !== undefined) throw new GeminiCompatibilityError('Gemini adapter does not support OpenAI function calling yet.');
  if (request.response_format !== undefined) throw new GeminiCompatibilityError('Gemini adapter does not support response_format yet.');
  if (request.logprobs !== undefined) throw new GeminiCompatibilityError('Gemini adapter does not support logprobs.');
  if (request.logit_bias !== undefined) throw new GeminiCompatibilityError('Gemini adapter does not support logit_bias.');
  if (request.n !== undefined && request.n !== 1) throw new GeminiCompatibilityError('Gemini adapter supports one completion choice per request.');
}

export function toGeminiRequest(input: Record<string, unknown>): Record<string, unknown> {
  const request = input as ChatRequest;
  unsupported(request);
  if (!Array.isArray(request.messages)) throw new GeminiCompatibilityError('Gemini adapter requires a messages array.');

  const systemParts: Array<{ text: string }> = [];
  const contents: Array<{ role: 'user' | 'model'; parts: Array<{ text: string }> }> = [];
  for (const rawMessage of request.messages) {
    if (!rawMessage || typeof rawMessage !== 'object') throw new GeminiCompatibilityError('Each message must be an object.');
    const message = rawMessage as ChatMessage;
    if (message.tool_calls !== undefined || message.function_call !== undefined) {
      throw new GeminiCompatibilityError('Gemini adapter does not support tool-call messages yet.');
    }
    const text = textFromContent(message.content);
    if (message.role === 'system' || message.role === 'developer') {
      systemParts.push({ text });
      continue;
    }
    if (message.role !== 'user' && message.role !== 'assistant') {
      throw new GeminiCompatibilityError('Gemini adapter does not support this message role.');
    }
    contents.push({ role: message.role === 'assistant' ? 'model' : 'user', parts: [{ text }] });
  }
  if (!contents.length) throw new GeminiCompatibilityError('At least one user or assistant message is required.');

  const generationConfig: Record<string, unknown> = {};
  const mappings: Array<[keyof ChatRequest, string]> = [
    ['temperature', 'temperature'], ['top_p', 'topP'], ['top_k', 'topK'],
    ['max_tokens', 'maxOutputTokens'], ['max_completion_tokens', 'maxOutputTokens'],
    ['presence_penalty', 'presencePenalty'], ['frequency_penalty', 'frequencyPenalty'], ['seed', 'seed'],
  ];
  for (const [source, target] of mappings) {
    if (request[source] !== undefined) generationConfig[target] = request[source];
  }
  if (request.stop !== undefined) generationConfig.stopSequences = Array.isArray(request.stop) ? request.stop : [request.stop];

  const body: Record<string, unknown> = { contents };
  if (systemParts.length) body.systemInstruction = { parts: systemParts };
  if (Object.keys(generationConfig).length) body.generationConfig = generationConfig;
  return body;
}

export function geminiEndpoint(baseUrl: string, model: string, streaming: boolean): string {
  const method = streaming ? 'streamGenerateContent' : 'generateContent';
  const endpoint = `${baseUrl.replace(/\/+$/, '')}/models/${encodeURIComponent(model)}:${method}`;
  return streaming ? `${endpoint}?alt=sse` : endpoint;
}

function openAiUsage(usage?: GeminiResponse['usageMetadata']) {
  if (!usage) return undefined;
  return {
    prompt_tokens: usage.promptTokenCount ?? 0,
    completion_tokens: usage.candidatesTokenCount ?? 0,
    total_tokens: usage.totalTokenCount ?? 0,
  };
}

function finishReason(reason?: string, blocked = false): 'stop' | 'length' | 'content_filter' {
  if (blocked || ['SAFETY', 'RECITATION', 'BLOCKLIST', 'PROHIBITED_CONTENT', 'SPII', 'IMAGE_SAFETY', 'JAILBREAK', 'LANGUAGE', 'MALFORMED_FUNCTION_CALL'].includes(reason ?? '')) return 'content_filter';
  if (reason === 'MAX_TOKENS') return 'length';
  return 'stop';
}

function candidateText(candidate?: GeminiCandidate): string {
  return (candidate?.content?.parts ?? []).filter((part) => !part.thought && typeof part.text === 'string').map((part) => part.text).join('');
}

export function toOpenAiResponse(response: GeminiResponse, requestedModel: string) {
  if (response.error) throw new GeminiCompatibilityError('Gemini returned an error payload.');
  if (response.error) throw new GeminiCompatibilityError('Gemini returned an error payload.');
  const candidate = response.candidates?.[0];
  const blocked = Boolean(response.promptFeedback?.blockReason);
  const usage = openAiUsage(response.usageMetadata);
  return {
    id: `chatcmpl-${response.responseId ?? randomUUID()}`,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model: response.modelVersion ?? requestedModel,
    choices: [{
      index: candidate?.index ?? 0,
      message: { role: 'assistant', content: candidateText(candidate) },
      finish_reason: finishReason(candidate?.finishReason, blocked),
    }],
    ...(usage ? { usage } : {}),
  };
}

export function toOpenAiChunk(response: GeminiResponse, requestedModel: string, id: string, created: number, includeRole: boolean) {
  const candidate = response.candidates?.[0];
  const delta: Record<string, string> = {};
  if (includeRole) delta.role = 'assistant';
  const text = candidateText(candidate);
  if (text) delta.content = text;
  const blocked = Boolean(response.promptFeedback?.blockReason);
  return {
    id, object: 'chat.completion.chunk', created,
    model: response.modelVersion ?? requestedModel,
    choices: [{ index: candidate?.index ?? 0, delta, finish_reason: candidate?.finishReason || blocked ? finishReason(candidate?.finishReason, blocked) : null }],
  };
}

export function toOpenAiUsageChunk(response: GeminiResponse, requestedModel: string, id: string, created: number) {
  const usage = openAiUsage(response.usageMetadata);
  return usage ? { id, object: 'chat.completion.chunk', created, model: response.modelVersion ?? requestedModel, choices: [], usage } : undefined;
}

export async function* readGeminiEventStream(body: ReadableStream<Uint8Array>): AsyncGenerator<GeminiResponse> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const parseEvent = (event: string): GeminiResponse | undefined => {
    const data = event.split(/\r?\n/).filter((line) => line.startsWith('data:')).map((line) => line.slice(5).trimStart()).join('\n');
    if (!data || data === '[DONE]') return undefined;
    const response = JSON.parse(data) as GeminiResponse;
    if (response.error) throw new GeminiCompatibilityError('Gemini returned an error payload.');
    return response;
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
