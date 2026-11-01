# Changelog

Notable project changes are recorded here. The notes below describe features documented in the repository; publication and release testing are tracked separately.

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
