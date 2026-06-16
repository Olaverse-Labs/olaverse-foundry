# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x (latest) | Yes |

## Reporting a vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Email **hello@olaverse.co.uk** with:

- A description of the vulnerability
- Steps to reproduce
- Potential impact

We will acknowledge receipt within 48 hours and aim to release a fix within 14 days for confirmed issues. We will credit you in the changelog unless you prefer to remain anonymous.

## Scope

`olaverse-foundry` is a local training library. It does not run a server, handle user authentication, or process network requests directly. The main security considerations are:

- **Pickle / checkpoint loading** — `resume_from_checkpoint()` uses `torch.load(weights_only=False)`. Only load checkpoints from sources you trust.
- **HuggingFace model loading** — `load_model()` passes `trust_remote_code` from your `ModelRef`. Do not set `trust_remote_code=True` for untrusted repositories.
- **YAML recipe files** — recipes are parsed with PyYAML and validated with Pydantic. Do not run recipe files from untrusted sources.
