# Changelog

Notable project changes are recorded here. The notes below describe features documented in the repository; publication and release testing are tracked separately.

## [Unreleased]

### Added
- Provider adapter registry: provider types are data-driven (`openai-compatible`, `gemini`, `anthropic`), so new upstreams plug in without touching routing code.
- Native Anthropic Messages adapter with full parity: tool calling (definitions, `tool_choice`, assistant `tool_calls`, `tool` results), image content parts (base64 and http(s) URLs), streaming text and incremental tool-call deltas, and token usage mapping.
- `GET /v1/models` for OpenAI-style model discovery.
- `POST /admin/providers` to create providers, with duplicate-id conflict detection.
- `POST /admin/providers/:id/test` to probe a provider connection and discover its model list from the dashboard.
- Dashboard provider form with adapter-specific hints, create/edit flows, and a test-connection button that can auto-fill models.

### Compatibility
- Legacy `openai` provider type is accepted as `openai-compatible` across environment configuration, dashboard edits, and the stored provider registry.
- Anthropic streaming emits OpenAI-style `data: [DONE]` termination; tool calls surface as incremental `tool_calls` deltas compatible with OpenAI clients.

### Limitations
- Gemini remains text-only; tool calls and multimodal parts are still rejected.
- Anthropic legacy `functions`/`function_call`, `response_format`, `logprobs`, `logit_bias`, audio parts, and `n > 1` are rejected with a clear error.

## [Unreleased] — preparing v0.2.0

### Added
- Dashboard-managed provider configuration with local JSON persistence, atomic writes, and documented configuration precedence.
- Native Google Gemini text-chat adapter with non-streaming and SSE translation to OpenAI-compatible responses.
- Persistent client API keys with separate admin and client authentication.
- Contributor guidelines, issue and pull request templates, a security reporting policy, and CI for Node.js 22 and 24.

### Security and reliability
- Administrative provider GET responses do not return provider API credentials.
- Invalid or unreadable stored provider configuration fails startup without overwriting the file.
- Provider API credentials stored in the JSON registry remain plaintext on disk and must be protected by filesystem permissions and backup controls.

### Migration and limitations
- Existing `PROVIDER_*` environment configuration works until a saved provider registry exists.
- Once created, the saved provider registry takes precedence over environment provider entries. Read the README before upgrading.
- Native Gemini integration currently supports text-only conversations; tool calls, multimodal requests, and multiple candidates are not supported.

### Release status
This entry does **not** mean `v0.2.0` has been tagged or published. Complete CI, security regression, integration, and restart-persistence checks before creating the GitHub Release.
