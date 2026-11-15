import { randomUUID } from 'node:crypto';

export class AnthropicCompatibilityError extends Error {}

type ChatMessage = {
  role?: unknown;
  content?: unknown;
  tool_calls?: unknown;
  tool_call_id?: unknown;
  function_call?: unknown;
  name?: unknown;
};
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
  content?: AnthropicContentBlock[];
  stop_reason?: string | null;
  stop_sequence?: string | null;
  usage?: { input_tokens?: number; output_tokens?: number };
  error?: unknown;
};

export type AnthropicContentBlock =
  | { type: 'text'; text: string }
  | { type: 'tool_use'; id: string; name: string; input: unknown }
  | { type: 'tool_result'; tool_use_id: string; content: string; is_error?: boolean }
  | { type: 'image'; source: { type: string; media_type?: string; data?: string; url?: string } };

const DEFAULT_MAX_TOKENS = 4096;

function unsupported(request: ChatRequest) {
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

function parseDataUrl(url: string): { mediaType: string; data: string } | undefined {
  const match = /^data:([^;,]+);base64,(.*)$/.exec(url);
  if (!match) return undefined;
  return { mediaType: match[1], data: match[2] };
}

function toContentBlocks(content: unknown): AnthropicContentBlock[] {
  if (typeof content === 'string') {
    return content ? [{ type: 'text', text: content }] : [];
  }
  if (!Array.isArray(content)) {
    throw new AnthropicCompatibilityError('Anthropic adapter requires message content to be text or an array of content parts.');
  }
  const blocks: AnthropicContentBlock[] = [];
  for (const rawPart of content) {
    if (!rawPart || typeof rawPart !== 'object') {
      throw new AnthropicCompatibilityError('Each content part must be an object.');
    }
    const part = rawPart as Record<string, unknown>;
    if (part.type === 'text') {
      if (typeof part.text !== 'string') throw new AnthropicCompatibilityError('Text content parts need a text field.');
      blocks.push({ type: 'text', text: part.text });
      continue;
    }
    if (part.type === 'image_url') {
      const imageUrl = part.image_url as { url?: unknown } | undefined;
      const url = typeof imageUrl?.url === 'string' ? imageUrl.url : '';
      if (!url) throw new AnthropicCompatibilityError('Image content parts need an image_url.url field.');
      const parsed = parseDataUrl(url);
      if (parsed) {
        blocks.push({ type: 'image', source: { type: 'base64', media_type: parsed.mediaType, data: parsed.data } });
        continue;
      }
      if (/^https?:\/\//i.test(url)) {
        blocks.push({ type: 'image', source: { type: 'url', url } });
        continue;
      }
      throw new AnthropicCompatibilityError('Image URLs must be http(s) URLs or base64 data URLs.');
    }
    if (part.type === 'input_audio') {
      throw new AnthropicCompatibilityError('Anthropic adapter does not support audio content parts.');
    }
    throw new AnthropicCompatibilityError(`Anthropic adapter does not support content part type "${String(part.type)}".`);
  }
  return blocks;
}

function textOnly(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) {
    throw new AnthropicCompatibilityError('Anthropic adapter requires message content to be text or an array of content parts.');
  }
  return content.map((part) => {
    if (!part || typeof part !== 'object' || (part as { type?: unknown }).type !== 'text' || typeof (part as { text?: unknown }).text !== 'string') {
      throw new AnthropicCompatibilityError('Anthropic adapter supports text-only message content.');
    }
    return (part as { text: string }).text;
  }).join('');
}

type ToolDefinition = {
  name: string;
  description?: string;
  input_schema: Record<string, unknown>;
};

function toAnthropicTools(tools: unknown): ToolDefinition[] {
  if (!Array.isArray(tools) || tools.length === 0) {
    throw new AnthropicCompatibilityError('Anthropic adapter requires a non-empty tools array.');
  }
  return tools.map((rawTool) => {
    if (!rawTool || typeof rawTool !== 'object') {
      throw new AnthropicCompatibilityError('Each tool must be an object.');
    }
    const tool = rawTool as { type?: unknown; function?: Record<string, unknown> };
    if (tool.type !== 'function' || !tool.function || typeof tool.function !== 'object') {
      throw new AnthropicCompatibilityError('Anthropic adapter supports only function-type tools.');
    }
    const fn = tool.function;
    if (typeof fn.name !== 'string' || !fn.name) {
      throw new AnthropicCompatibilityError('Each tool needs a function name.');
    }
    const schema = fn.parameters;
    return {
      name: fn.name,
      ...(typeof fn.description === 'string' ? { description: fn.description } : {}),
      input_schema: schema && typeof schema === 'object' ? (schema as Record<string, unknown>) : { type: 'object', properties: {} },
    };
  });
}

function toAnthropicToolChoice(toolChoice: unknown): Record<string, unknown> | undefined {
  if (toolChoice === undefined || toolChoice === null) return undefined;
  if (typeof toolChoice === 'string') {
    if (toolChoice === 'auto') return { type: 'auto' };
    if (toolChoice === 'none') return { type: 'none' };
    if (toolChoice === 'required' || toolChoice === 'any') return { type: 'any' };
    throw new AnthropicCompatibilityError('Anthropic adapter supports tool_choice values auto, none, required, or a named function.');
  }
  if (typeof toolChoice === 'object') {
    const choice = toolChoice as { type?: unknown; function?: { name?: unknown } };
    if (choice.type === 'function' && typeof choice.function?.name === 'string') {
      return { type: 'tool', name: choice.function.name };
    }
  }
  throw new AnthropicCompatibilityError('Anthropic adapter supports tool_choice values auto, none, required, or a named function.');
}

function toAnthropicMessages(messages: ChatMessage[]): Array<{ role: 'user' | 'assistant'; content: AnthropicContentBlock[] }> {
  const contents: Array<{ role: 'user' | 'assistant'; content: AnthropicContentBlock[] }> = [];

  const push = (role: 'user' | 'assistant', blocks: AnthropicContentBlock[]) => {
    if (!blocks.length) return;
    const previous = contents[contents.length - 1];
    if (previous && previous.role === role) {
      previous.content.push(...blocks);
      return;
    }
    contents.push({ role, content: blocks });
  };

  for (const message of messages) {
    const role = message.role;
    if (role === 'system' || role === 'developer') continue;

    if (role === 'tool') {
      if (typeof message.tool_call_id !== 'string' || !message.tool_call_id) {
        throw new AnthropicCompatibilityError('Tool result messages need a tool_call_id.');
      }
      push('user', [{
        type: 'tool_result',
        tool_use_id: message.tool_call_id,
        content: textOnly(message.content),
      }]);
      continue;
    }

    if (role !== 'user' && role !== 'assistant') {
      throw new AnthropicCompatibilityError('Anthropic adapter does not support this message role.');
    }

    const blocks: AnthropicContentBlock[] = [];
    const toolCalls = message.tool_calls;
    if (toolCalls !== undefined) {
      if (!Array.isArray(toolCalls)) throw new AnthropicCompatibilityError('Tool calls must be an array.');
      for (const rawCall of toolCalls) {
        if (!rawCall || typeof rawCall !== 'object') throw new AnthropicCompatibilityError('Each tool call must be an object.');
        const call = rawCall as { id?: unknown; type?: unknown; function?: { name?: unknown; arguments?: unknown } };
        if (call.type !== undefined && call.type !== 'function') {
          throw new AnthropicCompatibilityError('Anthropic adapter supports only function-type tool calls.');
        }
        const name = call.function?.name;
        if (typeof name !== 'string' || !name) throw new AnthropicCompatibilityError('Each tool call needs a function name.');
        const rawArguments = call.function?.arguments;
        let input: unknown = {};
        if (typeof rawArguments === 'string' && rawArguments.trim()) {
          try {
            input = JSON.parse(rawArguments);
          } catch {
            throw new AnthropicCompatibilityError('Tool call arguments must be valid JSON.');
          }
        } else if (rawArguments && typeof rawArguments === 'object') {
          input = rawArguments;
        }
        blocks.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : randomUUID(), name, input });
      }
    }

    const content = message.content;
    if (content !== undefined && content !== null) {
      if (typeof content === 'string') {
        if (content) blocks.unshift({ type: 'text', text: content });
      } else {
        blocks.unshift(...toContentBlocks(content));
      }
    }

    push(role === 'assistant' ? 'assistant' : 'user', blocks);
  }

  return contents;
}

export function toAnthropicRequest(input: Record<string, unknown>): Record<string, unknown> {
  const request = input as ChatRequest;
  unsupported(request);
  if (!Array.isArray(request.messages)) throw new AnthropicCompatibilityError('Anthropic adapter requires a messages array.');

  const systemParts: string[] = [];
  const conversation: ChatMessage[] = [];
  for (const rawMessage of request.messages) {
    if (!rawMessage || typeof rawMessage !== 'object') throw new AnthropicCompatibilityError('Each message must be an object.');
    const message = rawMessage as ChatMessage;
    if (message.function_call !== undefined) {
      throw new AnthropicCompatibilityError('Anthropic adapter does not support legacy function_call messages.');
    }
    if (message.role === 'system' || message.role === 'developer') {
      const systemText = typeof message.content === 'string'
        ? message.content
        : toContentBlocks(message.content).map((block) => (block.type === 'text' ? block.text : '')).join('');
      if (systemText) systemParts.push(systemText);
      continue;
    }
    conversation.push(message);
  }

  const contents = toAnthropicMessages(conversation);
  if (!contents.length) throw new AnthropicCompatibilityError('At least one user or assistant message is required.');
  if (contents[0].role !== 'user') {
    throw new AnthropicCompatibilityError('Anthropic requires the conversation to start with a user message.');
  }

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
  const toolsProvided = Array.isArray(request.tools) && request.tools.length > 0;
  if (toolsProvided) body.tools = toAnthropicTools(request.tools);
  const toolChoice = toAnthropicToolChoice(request.tool_choice);
  if (toolChoice && toolsProvided) body.tool_choice = toolChoice;
  return body;
}

function finishReason(reason?: string | null): 'stop' | 'length' | 'tool_calls' {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  return 'stop';
}

function toolCallFromBlock(block: Extract<AnthropicContentBlock, { type: 'tool_use' }>) {
  let args = '{}';
  if (block.input !== undefined) {
    args = typeof block.input === 'string' ? block.input : JSON.stringify(block.input);
  }
  return {
    id: block.id,
    type: 'function' as const,
    function: { name: block.name, arguments: args },
  };
}

export function toOpenAiResponse(response: AnthropicResponse, requestedModel: string) {
  if (response.error) throw new AnthropicCompatibilityError('Anthropic returned an error payload.');
  const blocks = response.content ?? [];
  const text = blocks
    .filter((block): block is Extract<AnthropicContentBlock, { type: 'text' }> => block?.type === 'text')
    .map((block) => block.text)
    .join('');
  const toolCalls = blocks
    .filter((block): block is Extract<AnthropicContentBlock, { type: 'tool_use' }> => block?.type === 'tool_use')
    .map(toolCallFromBlock);
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
      message: {
        role: 'assistant',
        content: text || (toolCalls.length ? null : ''),
        ...(toolCalls.length ? { tool_calls: toolCalls } : {}),
      },
      finish_reason: toolCalls.length ? 'tool_calls' : finishReason(response.stop_reason),
    }],
    ...(usage ? { usage } : {}),
  };
}

type AnthropicStreamEvent = {
  type?: string;
  message?: AnthropicResponse;
  content_block?: AnthropicContentBlock;
  index?: number;
  delta?: {
    type?: string;
    text?: string;
    partial_json?: string;
    stop_reason?: string | null;
    thinking?: string;
  };
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

type ToolCallAccumulator = {
  id: string;
  name: string;
  args: string;
  sent: boolean;
};

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
  let toolCallIndex = 0;
  const toolCalls: ToolCallAccumulator[] = [];

  const ensureRole = (): OpenAiStreamChunk | undefined => {
    if (roleSent) return undefined;
    roleSent = true;
    return openAiChunk(model, id, created, { role: 'assistant' });
  };

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
      const roleChunk = ensureRole();
      if (roleChunk) yield roleChunk;
      continue;
    }

    if (event.type === 'content_block_start') {
      const block = event.content_block;
      if (block?.type === 'tool_use') {
        const roleChunk = ensureRole();
        if (roleChunk) yield roleChunk;
        const accumulator: ToolCallAccumulator = {
          id: block.id,
          name: block.name,
          args: '',
          sent: false,
        };
        toolCalls.push(accumulator);
        yield openAiChunk(model, id, created, {
          tool_calls: [{
            index: toolCallIndex++,
            id: block.id,
            type: 'function',
            function: { name: block.name, arguments: '' },
          }],
        });
        accumulator.sent = true;
      }
      continue;
    }

    if (event.type === 'content_block_delta') {
      const text = event.delta?.text;
      if (typeof text === 'string' && text) {
        const roleChunk = ensureRole();
        if (roleChunk) yield roleChunk;
        yield openAiChunk(model, id, created, { content: text });
        continue;
      }
      const partial = event.delta?.partial_json;
      if (typeof partial === 'string' && partial) {
        const accumulator = toolCalls[toolCalls.length - 1];
        if (accumulator) {
          accumulator.args += partial;
          yield openAiChunk(model, id, created, {
            tool_calls: [{
              index: toolCallIndex - 1,
              function: { arguments: partial },
            }],
          });
        }
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

  const roleChunk = ensureRole();
  if (roleChunk) yield roleChunk;
  const finalFinish = toolCalls.length ? 'tool_calls' : finish ?? 'stop';
  yield openAiChunk(model, id, created, {}, finalFinish);
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
