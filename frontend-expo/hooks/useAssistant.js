import { useState, useCallback, useRef } from "react";
import { storageGet, storageSet } from "../services/storage";
import { apiPost } from "../services/api";

const CHAT_STORAGE_KEY = "quantflow_chat";
const MAX_TURNS = 20;

export default function useAssistant(token) {
  const [messages, setMessages] = useState([]);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState(null);
  const loadedRef = useRef(false);

  const loadHistory = useCallback(async () => {
    if (loadedRef.current) return;
    loadedRef.current = true;
    try {
      const stored = await storageGet(CHAT_STORAGE_KEY);
      if (stored && Array.isArray(stored)) {
        setMessages(stored.slice(-MAX_TURNS * 2));
      }
    } catch {}
  }, []);

  const persistMessages = useCallback(async (msgs) => {
    try {
      await storageSet(CHAT_STORAGE_KEY, msgs.slice(-MAX_TURNS * 2));
    } catch {}
  }, []);

  const send = useCallback(
    async (text, context) => {
      if (!text.trim()) return;
      setSending(true);
      setError(null);

      const userMsg = { role: "user", text: text.trim(), ts: Date.now() };
      const updated = [...messages, userMsg];
      setMessages(updated);
      persistMessages(updated);

      try {
        const body = {
          message: text.trim(),
          context: context || {},
        };

        const llmConfig = await storageGet("llm_config");
        let reply;

        if (llmConfig?.endpoint && llmConfig?.key) {
          try {
            const res = await fetch(`${llmConfig.endpoint}/v1/chat/completions`, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${llmConfig.key}`,
              },
              body: JSON.stringify({
                model: "deepseek-chat",
                messages: [
                  { role: "system", content: "You are a quantitative finance assistant for the QuantFlow platform. Answer concisely." },
                  ...messages.map((m) => ({ role: m.role, content: m.text })),
                  { role: "user", content: text.trim() },
                ],
              }),
            });
            const json = await res.json();
            reply =
              json.choices?.[0]?.message?.content ||
              json.answer ||
              "No response from LLM.";
          } catch {
            // Fall through to local assistant
          }
        }

        if (!reply) {
          const data = await apiPost("/assistant/query", body, token);
          reply = data.answer || data.response || data.reply || "No response from assistant.";
        }

        const assistantMsg = {
          role: "assistant",
          text: reply,
          ts: Date.now(),
          actions: parseActions(reply),
        };

        const final = [...updated, assistantMsg];
        setMessages(final);
        persistMessages(final);
      } catch (err) {
        const errText = err?.message || (err && typeof err === "object" ? JSON.stringify(err) : String(err || "Unknown error"));
        setError(errText);
        const errMsg = {
          role: "assistant",
          text: `Error: ${errText}`,
          ts: Date.now(),
          error: true,
        };
        const final = [...updated, errMsg];
        setMessages(final);
        persistMessages(final);
      } finally {
        setSending(false);
      }
    },
    [messages, token, persistMessages]
  );

  const clear = useCallback(async () => {
    setMessages([]);
    await storageSet(CHAT_STORAGE_KEY, []);
  }, []);

  return { messages, sending, error, send, clear, loadHistory };
}

function parseActions(text) {
  if (!text) return [];
  const actions = [];
  try {
    const jsonMatch = text.match(/```json\s*([\s\S]*?)```/);
    if (jsonMatch) {
      const parsed = JSON.parse(jsonMatch[1]);
      if (parsed.suggested_tool_calls || parsed.actions) {
        return parsed.suggested_tool_calls || parsed.actions || [];
      }
    }
  } catch {}
  return actions;
}
