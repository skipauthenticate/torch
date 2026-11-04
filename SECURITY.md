# Security

The engine loads model files and serves an HTTP API, so two kinds of issues matter most:

- **Malformed model files** that crash the loader or make it read or write out of bounds.
- **The HTTP server**, which listens on `127.0.0.1` by default. Exposing it on a network without an API key and a
  firewall is not supported.

Please report a vulnerability privately through
[GitHub security advisories](https://github.com/skipauthenticate/torch/security/advisories/new), not in a
public issue. We aim to reply within a week.
