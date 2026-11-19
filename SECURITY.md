# Security

The engine loads model files and serves an HTTP API, so two kinds of issues matter most:

- **Malformed model files** that crash the loader or make it read or write out of bounds.
- **The HTTP server**, which listens on `127.0.0.1` by default. Exposing it on a network without an API key and a
  firewall is not supported. It refuses another `--host` without `--api-key`, has no TLS, requires
  `Content-Type: application/json`, sends no CORS headers and checks the `Host` header, so a web page cannot post to
  it; message text is not isolated from the chat structure, except tool results, which cannot carry control tokens
  ([docs/server.md](docs/server.md#security)). The server never runs a tool: tool calls go back to the client.

The server holds prompt checkpoints in device memory. Disk caching requires an explicit `--prefix-disk DIR`.
When enabled, unnamed files contain model state derived from conversation text. The files use mode 0600 in a
directory owned by the server account with mode 0700. The kernel releases the files when the process exits,
including after a crash. File removal does not securely erase the storage device. A running engine shares its
prefix cache across requests. Use a separate process for each group of clients that needs private cache activity.

Please report a vulnerability privately through
[GitHub security advisories](https://github.com/skipauthenticate/torch/security/advisories/new), not in a
public issue. We aim to reply within a week.
