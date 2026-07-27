import AsyncStorage from "@react-native-async-storage/async-storage";

function unwrap(entry) {
  if (entry == null) return null;
  if (typeof entry === "string") {
    try { return unwrap(JSON.parse(entry)); } catch { return entry; }
  }
  if (typeof entry !== "object") return entry;
  if (Array.isArray(entry)) return entry;
  if ("_value" in entry) {
    if (entry._expiresAt && Date.now() > entry._expiresAt) return null;
    return entry._value;
  }
  return entry;
}

export async function storageGet(key) {
  try {
    const raw = await AsyncStorage.getItem(key);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return unwrap(parsed);
  } catch {
    return null;
  }
}

export async function storageSet(key, value, ttlSeconds) {
  if (ttlSeconds) {
    const entry = { _value: value, _expiresAt: Date.now() + ttlSeconds * 1000 };
    await AsyncStorage.setItem(key, JSON.stringify(entry));
    return;
  }
  await AsyncStorage.setItem(key, JSON.stringify(value));
}

export async function storageRemove(key) {
  await AsyncStorage.removeItem(key);
}

export async function storageGetRaw(key) {
  return AsyncStorage.getItem(key);
}

export async function storageSetRaw(key, value) {
  return AsyncStorage.setItem(key, value);
}
