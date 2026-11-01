# Contributing to K'sRouter

Thanks for considering a contribution. K'sRouter is a self-hosted AI gateway. Keep changes small, testable, and compatible with the existing OpenAI-compatible client endpoint.

## Before you start

- For bugs or feature proposals, check [open issues](https://github.com/Kian1904/K-rout/issues) first.
- For security vulnerabilities, **do not post credentials, tokens, or exploitation details in public issues**. Contact the maintainer privately using a GitHub-supported private channel if available.
- For larger changes, open an issue describing the problem, expected behavior, and intended scope before implementing.

## Local development

Requirements: Node.js 22 or 24 recommended (CI-tested), and npm. Other versions are outside the current CI matrix.

```sh
npm ci
cp .env.example .env
npm run build
npm test
```

Configure your own provider credentials in `.env` for live testing; **never commit real credentials**. For local development:

```sh
npm run dev
```

The dashboard is served at `http://127.0.0.1:5173/dashboard/`. The API defaults to `127.0.0.1:3000`.

## Testing

- Run `npm test` and `npm run build` before opening a pull request.
- Prefer mock upstreams and regression tests; do not consume someone else's paid API credits.
- For changes to streaming, include cancellation and SSE framing test cases.
- For provider changes, test model mapping, malformed payloads, credential redaction, and provider errors.
- Benchmarks are local baselines, not production throughput guarantees.

## Pull request checklist

- [ ] My change solves a specific issue or documented bug.
- [ ] Tests cover relevant behavior and edge cases.
- [ ] `npm test` and `npm run build` pass.
- [ ] README or relevant documentation is updated.
- [ ] No credentials, private URLs, request bodies containing sensitive data, or secrets were added.
- [ ] Backward compatibility and unsupported features are documented.

## Scope and design

Prefer small changes over large rewrites. Avoid introducing databases, complex routing layers, or dependencies without a concrete use case. The project currently supports OpenAI-compatible upstreams and a text-only native Gemini adapter; do not claim broader feature support without tests.
