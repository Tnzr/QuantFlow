const API_BASE =
  (typeof process !== "undefined" && process.env?.EXPO_PUBLIC_API_BASE_URL) ||
  "http://127.0.0.1:3000";

const ML_ENGINE_URL =
  (typeof process !== "undefined" && process.env?.EXPO_PUBLIC_ML_ENGINE_URL) ||
  "http://127.0.0.1:8000";

export const APP_CONFIG = {
  apiBase: API_BASE,
  mlEngineUrl: ML_ENGINE_URL,
};

export function getApiBase() {
  return API_BASE;
}

export function getMlEngineUrl() {
  return ML_ENGINE_URL;
}
