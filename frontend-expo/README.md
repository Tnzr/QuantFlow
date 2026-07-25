# QuantFlow Expo Frontend (Demo Studio)

This is a React Native + Expo demo frontend for QuantFlow with a landing-style narrative and operational screens wired to the FastAPI backend.

## Prerequisites
- Node.js 18+
- npm
- Expo Go app on phone (optional for device testing)

## Run
1. Start backend (port 8100 recommended):
   - `cd /mnt/d/Dev/QuantFlow`
   - `make api API_PORT=8100`
2. Start Expo app:
   - `cd /mnt/d/Dev/QuantFlow/frontend-expo`
   - `npm install`
   - `EXPO_PUBLIC_API_BASE_URL=http://<YOUR_LAN_IP>:8100 npm start`

Optional auth header (token mode backend):
- `EXPO_PUBLIC_API_KEY=<your-token> EXPO_PUBLIC_API_BASE_URL=http://<YOUR_LAN_IP>:8100 npm start`

For static export (landing/demo hosting):
- `cd /mnt/d/Dev/QuantFlow/frontend-expo`
- `npx expo export --platform web --output-dir dist`

For Android emulator/web, `localhost` can work.
For phone testing on same Wi-Fi, use your machine LAN IP.

## Experience Split
- Public surface: landing-safe narrative with live system metrics for demos and external-facing walkthroughs.
- Workspace surface: gated operator tabs unlocked after auth check.

### Web Direct Links
- Public: `/?view=public`
- Workspace gate: `/?view=workspace`
- Workspace tab deep link (after unlock): `/?view=workspace&tab=Overview`

## Workspace Tabs
- Overview: health, environment validation, ops report, MCP status/readiness, presets, intents.
- Scanner: run scan pipeline and inspect latest universe rows.
- Recommend: run recommendation pipeline and review latest recs.
- Options: fetch affordable options ideas.
- Backtest: run short-term backtest summary.
- Execution: create policy-gated intents.
- Assistant: query offline assistant and execute suggested API tool calls (with POST dry-run guard).
- Auth: manage API key/Bearer token and read/write Firebase-backed user settings.

## Auth Notes
- Token mode can use `x-api-key` (from `EXPO_PUBLIC_API_KEY` or entered in Auth tab).
- Firebase mode uses Bearer token entered in the Auth tab.
- User settings (`/user/settings`) require Firebase auth enabled in backend and a valid user token.
- Workspace unlock is a client-side UX gate for demo safety; backend endpoint authorization remains the source of truth.

### Google Login (Web)
Set these before running web if you want popup Google auth in the Auth tab:
- `EXPO_PUBLIC_FIREBASE_API_KEY`
- `EXPO_PUBLIC_FIREBASE_AUTH_DOMAIN`
- `EXPO_PUBLIC_FIREBASE_PROJECT_ID`
- `EXPO_PUBLIC_FIREBASE_APP_ID`

Example:
- `EXPO_PUBLIC_API_BASE_URL=http://127.0.0.1:8100 EXPO_PUBLIC_FIREBASE_API_KEY=... EXPO_PUBLIC_FIREBASE_AUTH_DOMAIN=... EXPO_PUBLIC_FIREBASE_PROJECT_ID=... EXPO_PUBLIC_FIREBASE_APP_ID=... npm run web`

## Scanner Controls
- You can run selected preset profiles only (e.g. `weekly_momo,weekly_bear`) instead of scanning everything.
- Category quick-select buttons are available for weekly/monthly/all.
- Latest universe can be filtered by preset and row limit.
- Assistant now handles weekly scan intent prompts and returns runnable actions.

## Next Suggested Steps
1. Add native Google sign-in in Expo and auto-inject Firebase ID token (remove manual token paste).
2. Add charts screen using `/charts/indicators` and `/charts/seasonality` for visual storytelling.
3. Add persisted profile presets (dev/stage/prod API endpoints + auth profiles).
4. Add shareable public landing route and a gated operator workspace route.
