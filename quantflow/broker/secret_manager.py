from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import json
import os


class SecretResolutionError(RuntimeError):
    pass


@dataclass
class SecretProvider:
    name: str

    def resolve(self, ref: str) -> str:
        raise NotImplementedError


class EnvSecretProvider(SecretProvider):
    def __init__(self):
        super().__init__(name="env")

    def resolve(self, ref: str) -> str:
        key = str(ref or "").strip()
        if key.lower().startswith("env:"):
            key = key[4:].strip()
        if not key:
            raise SecretResolutionError("empty env secret reference")
        val = os.getenv(key, "")
        if not val:
            raise SecretResolutionError(f"env secret reference not found: {key}")
        return val


class VaultSecretProvider(SecretProvider):
    """HashiCorp Vault KV v2 provider.

    Reference format:
    - path#field (uses QF_VAULT_MOUNT_POINT, default: secret)
    - mount:path#field
    """

    def __init__(self):
        super().__init__(name="vault")

    def resolve(self, ref: str) -> str:
        try:
            import hvac  # type: ignore
        except Exception as e:
            raise SecretResolutionError("vault provider requires hvac package") from e

        url = os.getenv("QF_VAULT_ADDR", "").strip()
        token = os.getenv("QF_VAULT_TOKEN", "").strip()
        if not url or not token:
            raise SecretResolutionError("vault provider requires QF_VAULT_ADDR and QF_VAULT_TOKEN")

        mount_point = os.getenv("QF_VAULT_MOUNT_POINT", "secret").strip() or "secret"
        raw = str(ref or "").strip()
        if not raw or "#" not in raw:
            raise SecretResolutionError("vault secret reference must be path#field or mount:path#field")

        path_part, field = raw.split("#", 1)
        field = field.strip()
        if not field:
            raise SecretResolutionError("vault field component is required")

        if ":" in path_part:
            mount_point, path = path_part.split(":", 1)
            mount_point = mount_point.strip() or mount_point
            path = path.strip()
        else:
            path = path_part.strip()

        if not path:
            raise SecretResolutionError("vault path component is required")

        try:
            client = hvac.Client(url=url, token=token)
            secret = client.secrets.kv.v2.read_secret_version(path=path, mount_point=mount_point)
            data = (secret or {}).get("data", {}).get("data", {})
            val = data.get(field)
            if val is None:
                raise SecretResolutionError(f"vault field not found: {path}#{field}")
            return str(val)
        except SecretResolutionError:
            raise
        except Exception as e:
            raise SecretResolutionError(f"vault read failed: {e}") from e


class AWSSecretsManagerProvider(SecretProvider):
    """AWS Secrets Manager provider.

    Reference format:
    - secret_id
    - secret_id#json_key
    """

    def __init__(self):
        super().__init__(name="aws")

    def resolve(self, ref: str) -> str:
        try:
            import boto3  # type: ignore
        except Exception as e:
            raise SecretResolutionError("aws provider requires boto3 package") from e

        raw = str(ref or "").strip()
        if not raw:
            raise SecretResolutionError("empty aws secret reference")

        secret_id, json_key = _split_ref_key(raw)
        region = os.getenv("AWS_REGION", "").strip() or os.getenv("AWS_DEFAULT_REGION", "").strip() or None

        try:
            client = boto3.client("secretsmanager", region_name=region)
            resp = client.get_secret_value(SecretId=secret_id)
            value = str(resp.get("SecretString", "") or "")
            if not value:
                raise SecretResolutionError(f"aws secret is empty: {secret_id}")
            if not json_key:
                return value
            parsed = json.loads(value)
            if not isinstance(parsed, dict) or json_key not in parsed:
                raise SecretResolutionError(f"aws json key not found: {secret_id}#{json_key}")
            return str(parsed[json_key])
        except SecretResolutionError:
            raise
        except Exception as e:
            raise SecretResolutionError(f"aws secret read failed: {e}") from e


class GCPSecretManagerProvider(SecretProvider):
    """Google Secret Manager provider.

    Reference format:
    - projects/<project>/secrets/<name>/versions/<version>
    - projects/<project>/secrets/<name>/versions/<version>#json_key
    """

    def __init__(self):
        super().__init__(name="gcp")

    def resolve(self, ref: str) -> str:
        try:
            from google.cloud import secretmanager  # type: ignore
        except Exception as e:
            raise SecretResolutionError("gcp provider requires google-cloud-secret-manager package") from e

        raw = str(ref or "").strip()
        if not raw:
            raise SecretResolutionError("empty gcp secret reference")

        resource_name, json_key = _split_ref_key(raw)
        try:
            client = secretmanager.SecretManagerServiceClient()
            response = client.access_secret_version(request={"name": resource_name})
            value = response.payload.data.decode("utf-8")
            if not value:
                raise SecretResolutionError(f"gcp secret is empty: {resource_name}")
            if not json_key:
                return value
            parsed = json.loads(value)
            if not isinstance(parsed, dict) or json_key not in parsed:
                raise SecretResolutionError(f"gcp json key not found: {resource_name}#{json_key}")
            return str(parsed[json_key])
        except SecretResolutionError:
            raise
        except Exception as e:
            raise SecretResolutionError(f"gcp secret read failed: {e}") from e


class AzureKeyVaultProvider(SecretProvider):
    """Azure Key Vault provider.

    Reference format:
    - https://<vault-name>.vault.azure.net/secrets/<secret-name>
    - https://<vault-name>.vault.azure.net/secrets/<secret-name>/<version>
    - either form with optional #json_key suffix
    """

    def __init__(self):
        super().__init__(name="azure")

    def resolve(self, ref: str) -> str:
        try:
            from azure.identity import DefaultAzureCredential  # type: ignore
            from azure.keyvault.secrets import SecretClient  # type: ignore
        except Exception as e:
            raise SecretResolutionError("azure provider requires azure-identity and azure-keyvault-secrets") from e

        raw = str(ref or "").strip()
        if not raw:
            raise SecretResolutionError("empty azure secret reference")

        resource, json_key = _split_ref_key(raw)
        parts = resource.split("/")
        if len(parts) < 5 or "/secrets/" not in resource:
            raise SecretResolutionError("azure ref must be key vault secret URL")

        vault_url = "/".join(parts[:3])
        if not vault_url.startswith("https://"):
            raise SecretResolutionError("azure vault url must start with https://")

        # URL shape: https://vault.vault.azure.net/secrets/<name>[/<version>]
        try:
            secret_idx = parts.index("secrets")
            secret_name = parts[secret_idx + 1]
            secret_version: Optional[str] = parts[secret_idx + 2] if len(parts) > secret_idx + 2 else None
        except Exception as e:
            raise SecretResolutionError("invalid azure secret URL format") from e

        if not secret_name:
            raise SecretResolutionError("azure secret name is required")

        try:
            client = SecretClient(vault_url=vault_url, credential=DefaultAzureCredential())
            secret = client.get_secret(name=secret_name, version=secret_version)
            value = str(secret.value or "")
            if not value:
                raise SecretResolutionError(f"azure secret is empty: {secret_name}")
            if not json_key:
                return value
            parsed = json.loads(value)
            if not isinstance(parsed, dict) or json_key not in parsed:
                raise SecretResolutionError(f"azure json key not found: {secret_name}#{json_key}")
            return str(parsed[json_key])
        except SecretResolutionError:
            raise
        except Exception as e:
            raise SecretResolutionError(f"azure secret read failed: {e}") from e


def _split_ref_key(ref: str) -> tuple[str, Optional[str]]:
    if "#" not in ref:
        return ref, None
    base, key = ref.split("#", 1)
    base = base.strip()
    key = key.strip() or None
    return base, key


def get_secret_provider(name: str) -> SecretProvider:
    key = str(name or "env").strip().lower() or "env"
    if key == "env":
        return EnvSecretProvider()
    if key == "vault":
        return VaultSecretProvider()
    if key in {"aws", "aws_secrets_manager", "secretsmanager"}:
        return AWSSecretsManagerProvider()
    if key in {"gcp", "gsm", "google"}:
        return GCPSecretManagerProvider()
    if key in {"azure", "akv", "keyvault"}:
        return AzureKeyVaultProvider()
    raise SecretResolutionError(f"unknown secret provider: {name}")


def resolve_secret(provider_name: str, ref: str) -> str:
    provider = get_secret_provider(provider_name)
    return provider.resolve(ref)
