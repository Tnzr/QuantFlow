# QuantFlow API + Expo MVP Guide

## Goal
Run QuantFlow backend in WSL and consume it from React/Expo on web + phone for shareable MVP testing.

## 1) Backend Startup

Default ports:
- API: 8000
- Streamlit (legacy): 8501

Commands:
- `make api`
- `make app`

If a port is already occupied:
- `make api-stop`
- `make app-stop`

Custom ports:
- `make api API_PORT=8100`
- `make app APP_PORT=8601`

## 2) API Endpoints (Current)
Read endpoints:
- `GET /health`
- `GET /scanner/presets`
- `GET /scanner/latest-universe`
- `GET /recommend/latest`
- `GET /options/ideas`
- `GET /charts/indicators`
- `GET /charts/seasonality`
- `GET /backtest/short-term`
- `GET /execution/intents`

Write endpoints (policy/audit tracked):
- `POST /pipeline/scan`
- `POST /pipeline/recommend`
- `POST /execution/propose`

## 3) Auth Modes for Shared MVP
By default auth is disabled for local development.

Enable write protection:
- `QF_API_AUTH_ENABLED=true`

Token mode:
- `QF_API_TOKEN=<your-secret-token>`
- Pass either:
  - Header `x-api-key: <token>`
  - Header `Authorization: Bearer <token>`

Optional Firebase mode (scaffold):
- `QF_API_AUTH_ENABLED=true`
- `QF_FIREBASE_AUTH_ENABLED=true`
- Install and configure `firebase-admin` in backend environment
- Set `GOOGLE_APPLICATION_CREDENTIALS` for service account path

## 4) Expo Frontend
Starter app path:
- `frontend-expo`

Run:
1. `make api`
2. `cd frontend-expo && npm install`
3. `EXPO_PUBLIC_API_BASE_URL=http://<LAN_IP>:8000 npm start`

Phone testing notes:
- Phone and dev machine must share Wi-Fi network
- Use machine LAN IP, not localhost
- Keep backend CORS enabled (currently open in dev)

## 5) Container + Firebase Next
Recommended next for Play Store trajectory:
1. Containerize FastAPI with env-based config and health checks
2. Add Firebase Auth in Expo app and send ID tokens
3. Enforce token/Firebase write auth in production
4. Add rate limiting and request audit export
5. Add remote config for API base URL by environment (dev/stage/prod)
