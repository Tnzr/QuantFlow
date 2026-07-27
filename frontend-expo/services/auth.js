import AsyncStorage from "@react-native-async-storage/async-storage";

const AUTH_TOKEN_KEY = "quantflow_auth_token";
const ALPACA_KEYS_KEY = "quantflow_alpaca_keys";
const LEGACY_COMBINED_KEY = "quantflow_credentials";

let migrationPromise = null;

async function migrateLegacyCredentials() {
  try {
    const raw = await AsyncStorage.getItem(LEGACY_COMBINED_KEY);
    if (!raw) return;
    const data = JSON.parse(raw);
    if (data && typeof data === "object") {
      if (data.token && !(await AsyncStorage.getItem(AUTH_TOKEN_KEY))) {
        await AsyncStorage.setItem(AUTH_TOKEN_KEY, data.token);
      }
      if ((data.apiKey || data.secret) && !(await AsyncStorage.getItem(ALPACA_KEYS_KEY))) {
        await AsyncStorage.setItem(
          ALPACA_KEYS_KEY,
          JSON.stringify({ apiKey: data.apiKey, secret: data.secret })
        );
      }
    }
    await AsyncStorage.removeItem(LEGACY_COMBINED_KEY);
  } catch {
    // ignore migration errors
  }
}

function ensureMigrated() {
  if (!migrationPromise) migrationPromise = migrateLegacyCredentials();
  return migrationPromise;
}

export async function loadAuthToken() {
  await ensureMigrated();
  try {
    const token = await AsyncStorage.getItem(AUTH_TOKEN_KEY);
    if (token) return token;
    return null;
  } catch {
    return null;
  }
}

export async function saveAuthToken(token) {
  await ensureMigrated();
  await AsyncStorage.setItem(AUTH_TOKEN_KEY, String(token || ""));
}

export async function clearAuthToken() {
  await ensureMigrated();
  await AsyncStorage.removeItem(AUTH_TOKEN_KEY);
}

export async function getAlpacaKeys() {
  await ensureMigrated();
  try {
    const raw = await AsyncStorage.getItem(ALPACA_KEYS_KEY);
    if (!raw) return null;
    const data = JSON.parse(raw);
    return { apiKey: data.apiKey || "", secret: data.secret || "" };
  } catch {
    return null;
  }
}

export async function saveAlpacaKeys(apiKey, secret) {
  await ensureMigrated();
  await AsyncStorage.setItem(
    ALPACA_KEYS_KEY,
    JSON.stringify({ apiKey: String(apiKey || ""), secret: String(secret || "") })
  );
}

export async function clearAlpacaKeys() {
  await ensureMigrated();
  await AsyncStorage.removeItem(ALPACA_KEYS_KEY);
}

export async function hasCredentials() {
  await ensureMigrated();
  const token = await AsyncStorage.getItem(AUTH_TOKEN_KEY);
  return token != null;
}
