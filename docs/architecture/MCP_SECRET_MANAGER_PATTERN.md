# MCP Secret Manager Pattern

This document describes the production-safe pattern for Robinhood MCP auth material in QuantFlow.

## Goal

Avoid passing long-lived MCP credentials via shell env for every process start. Instead:

1. Keep secrets in a managed secret service (Vault/AWS/GCP/Azure).
2. Send only secret references to QuantFlow runtime config endpoint.
3. Let backend resolve references at runtime and inject headers/tokens internally.
4. Never return raw secret values from API responses.

## Runtime API

Endpoints:

- `GET /broker/mcp/config`
- `PUT /broker/mcp/config`
- `DELETE /broker/mcp/config`

Example request to `PUT /broker/mcp/config`:

```json
{
  "auth_header": "Authorization",
  "api_key_header": "X-API-Key",
  "custom_headers": {},
  "secret_provider": "vault",
  "secret_refs": {
    "bearer_token": "secret/data/quantflow/mcp#token",
    "custom_headers": "secret/data/quantflow/mcp#headers_json"
  },
  "provider_options": {}
}
```

Fields resolved from secret refs:

- `bearer_token`
- `api_key`
- `cookie`
- `custom_headers` (expects JSON object string)

## Provider Formats

### Vault (`vault`)

- `path#field`
- `mount:path#field`

Examples:

- `secret/data/quantflow/mcp#token`
- `kv:trading/robinhood#mcp_api_key`

Required env for provider client auth:

- `QF_VAULT_ADDR`
- `QF_VAULT_TOKEN`
- Optional: `QF_VAULT_NAMESPACE`, `QF_VAULT_MOUNT_POINT`

### AWS Secrets Manager (`aws`)

- `secret_id`
- `secret_id#json_key`

Examples:

- `prod/quantflow/mcp`
- `prod/quantflow/mcp#bearer_token`

Client auth handled via normal AWS credential chain.

### Google Secret Manager (`gcp`)

- `projects/<project>/secrets/<name>/versions/<version>`
- Optional `#json_key`

Examples:

- `projects/my-project/secrets/quantflow-mcp/versions/latest`
- `projects/my-project/secrets/quantflow-mcp/versions/latest#token`

### Azure Key Vault (`azure`)

- `https://<vault>.vault.azure.net/secrets/<name>`
- `https://<vault>.vault.azure.net/secrets/<name>/<version>`
- Optional `#json_key`

Examples:

- `https://myvault.vault.azure.net/secrets/quantflow-mcp`
- `https://myvault.vault.azure.net/secrets/quantflow-mcp#bearer_token`

## Security Guidance

- Enable API auth (`QF_API_AUTH_ENABLED=true`) before exposing runtime config endpoint.
- Restrict caller identity/role to backend operators only.
- Log provider/ref keys, not secret values.
- Use short TTL secrets and rotate in secret manager; do not bake into container images.
- Prefer workload identity / instance profile auth for cloud SDKs over static credentials.

## Fallback Behavior

Runtime override is preferred. If unset, env-based MCP config still works for local development.
