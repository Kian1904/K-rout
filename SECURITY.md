Security Policy

Thank you for helping keep K-rout (K'sRouter) and its users secure.

K-rout is an open-source, self-hosted AI gateway that handles provider API credentials, client authentication, and outbound requests to AI services.

Security reports are taken seriously, especially vulnerabilities involving credential exposure, unauthorized access, configuration handling, and request routing.

Supported Versions

K-rout is currently under active development.

Version| Security Support
Latest "main" branch| Supported
Older commits and development builds| Not guaranteed
Forks and modified distributions| Not officially supported

Security fixes are currently prioritized for the latest code on the "main" branch.

This policy may change as stable releases become available.

Reporting a Vulnerability

Please do not report security vulnerabilities through public GitHub Issues, Pull Requests, or Discussions.

Use GitHub's Private Vulnerability Reporting feature instead:

"Report a Vulnerability Privately" (https://github.com/Kian1904/K-sRouter-CLI/security/advisories/new)

When submitting a report, please include:

- A clear description of the vulnerability.
- The affected version or commit SHA.
- The potential security impact.
- Steps to reproduce using dummy credentials and a controlled environment.
- Relevant sanitized logs or error messages.
- Suggested mitigations, if available.

Do not submit real API keys, OAuth tokens, private user information, or production credentials.

Please avoid testing against infrastructure or accounts you do not own or have permission to assess.

Response and Disclosure Policy

K-rout is currently maintained by an independent developer.

Security reports will be reviewed as maintainer availability allows.

The project does not currently guarantee a fixed response or remediation timeline.

When a vulnerability is confirmed, the maintainer will aim to:

1. Assess the security impact and affected components.
2. Develop and test an appropriate fix.
3. Prepare a patch or security advisory when necessary.
4. Coordinate public disclosure with the reporter.

Responsible disclosure is encouraged to give affected users time to apply available fixes.

Security Architecture

Credential Management

K-rout separates administrator authentication from client API authentication.

- Administrator credentials are required for management endpoints.
- Client API keys are used to authenticate gateway requests.
- Client keys are displayed once when generated and stored as hashes.
- Provider API credentials are not returned in administrative GET responses.
- Logs should not contain API keys or sensitive request content.

Dashboard-managed provider credentials may be stored in plaintext within a local JSON configuration file.

Filesystem permissions provide a level of protection but do not constitute encryption at rest.

Configuration Storage

Default local storage paths include:

- ".data/providers.json" — provider configuration and credentials.
- ".data/client-keys.json" — client authentication records.

Custom paths can be configured using "PROVIDER_CONFIG_PATH" and "CLIENT_KEY_STORE_PATH".

These files must be protected from unauthorized access. Backups containing provider credentials should be handled as sensitive data.

Network Security

K-rout makes outbound requests to configured AI providers.

Administrators should:

- Use trusted upstream API endpoints.
- Avoid exposing management interfaces directly to the public internet.
- Use HTTPS and appropriate network access controls for remote deployments.
- Keep Node.js and dependencies updated.
- Protect environment files and stored credentials.

Security Limitations

K-rout is under active development and has not been independently certified or formally security-audited.

Current security limitations include:

- Provider credentials are not encrypted at rest.
- Filesystem protections depend on the hosting environment.
- Compromised administrator access may expose or modify sensitive provider configurations.
- Self-hosted deployments require administrators to secure their own infrastructure.

The existence of this security policy does not guarantee that the software is free from vulnerabilities.

Non-Security Issues

For regular bugs, feature requests, and documentation improvements, please use:

"GitHub Issues" (https://github.com/Kian1904/K-sRouter-CLI/issues)

For contribution instructions, see:

"CONTRIBUTING.md" (https://github.com/Kian1904/K-sRouter-CLI/blob/main/CONTRIBUTING.md)

---

Security is an ongoing engineering responsibility, not a one-time feature.

Thank you for helping make K-rout safer for everyone.
