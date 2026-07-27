const API_BASE =
  (typeof process !== "undefined" && process.env?.EXPO_PUBLIC_API_BASE_URL) ||
  "http://127.0.0.1:3000";

const TIMEOUT_MS = 10000;
const MAX_RETRIES = 3;
const RETRY_BASE_MS = 500;

async function request(method, path, body, authToken) {
  const url = path.startsWith("http") ? path : `${API_BASE}${path}`;

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

    try {
      const headers = { "Content-Type": "application/json" };
      if (authToken) headers["Authorization"] = `Bearer ${authToken}`;

      const opts = { method, headers, signal: controller.signal };
      if (body && method !== "GET") opts.body = JSON.stringify(body);

      const res = await fetch(url, opts);
      clearTimeout(timer);

      if (res.status === 401) {
        const data = await res.json().catch(() => null);
        const error = new Error(data?.detail || "Unauthorized");
        error.status = 401;
        error.code = data?.code || "auth_required";
        throw error;
      }

      if (!res.ok) {
        const data = await res.json().catch(() => null);
        const detail = typeof data?.detail === "string" ? data.detail : (data?.detail ? JSON.stringify(data.detail) : null);
        const error = new Error(detail || `HTTP ${res.status}`);
        error.status = res.status;
        error.code = data?.code || "server_error";
        error.detail = data?.detail || null;
        throw error;
      }

      const data = await res.json();
      return data;
    } catch (err) {
      clearTimeout(timer);
      if (err.name === "AbortError") {
        err = new Error("Request timed out");
        err.code = "timeout";
      }
      if (attempt < MAX_RETRIES && err.code !== "auth_required") {
        const delay = RETRY_BASE_MS * Math.pow(2, attempt);
        await new Promise((r) => setTimeout(r, delay));
        continue;
      }
      const normalized = new Error(err.message);
      normalized.status = err.status;
      normalized.code = err.code || "network_error";
      normalized.detail = err.detail;
      throw normalized;
    }
  }
}

export function apiGet(path, authToken) {
  return request("GET", path, null, authToken);
}

export function apiPost(path, body, authToken) {
  return request("POST", path, body, authToken);
}

export function apiPut(path, body, authToken) {
  return request("PUT", path, body, authToken);
}

export function apiDelete(path, authToken) {
  return request("DELETE", path, null, authToken);
}

export function getApiBase() {
  return API_BASE;
}

export async function apiPostPath(pathWithQuery, authToken) {
  const url = pathWithQuery.startsWith("http") ? pathWithQuery : `${API_BASE}${pathWithQuery}`;
  const headers = { "Content-Type": "application/json" };
  if (authToken) headers["Authorization"] = `Bearer ${authToken}`;
  const res = await fetch(url, { method: "POST", headers });
  if (!res.ok) {
    const data = await res.json().catch(() => null);
    const detail = typeof data?.detail === "string" ? data.detail : (data?.detail ? JSON.stringify(data.detail) : null);
    const error = new Error(detail || `HTTP ${res.status}`);
    error.status = res.status;
    error.code = data?.code || "server_error";
    error.detail = data?.detail || null;
    throw error;
  }
  return res.json();
}

export function cacheMarketSeries(ticker, interval, period, authToken) {
  const params = new URLSearchParams({ ticker, interval });
  if (period) params.set("period", period);
  return apiPostPath(`/market/series/cache?${params.toString()}`, authToken);
}
