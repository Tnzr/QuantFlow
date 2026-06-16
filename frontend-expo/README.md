# QuantFlow Expo Frontend (Starter)

This is a minimal React Native + Expo starter that talks to the QuantFlow FastAPI backend.

## Prerequisites
- Node.js 18+
- npm
- Expo Go app on phone (optional for device testing)

## Run
1. Start backend:
   - `make api`
2. Start Expo app:
   - `cd frontend-expo`
   - `npm install`
   - `EXPO_PUBLIC_API_BASE_URL=http://<YOUR_LAN_IP>:8000 npm start`

Optional auth header (token mode backend):
- `EXPO_PUBLIC_API_KEY=<your-token> EXPO_PUBLIC_API_BASE_URL=http://<YOUR_LAN_IP>:8000 npm start`

For Android emulator/web, `localhost` can work.
For phone testing on same Wi-Fi, use your machine LAN IP.

## Current screens/features
- Overview (health, presets, recent intents)
- Scanner (run scan pipeline + latest universe)
- Recommend (run recommendation pipeline + latest recs)
- Options (fetch option ideas)
- Backtest (run short-term backtest summary)
- Execution (create policy-gated intents)

## Next suggested steps
1. Add Firebase ID token flow and `Authorization: Bearer <token>` support in the app.
2. Add charts screen using `/charts/indicators` and `/charts/seasonality`.
3. Add richer validation and error messaging for input fields.
4. Add persistent app state and refresh-on-focus behavior.
5. Add environment profile support (dev/stage/prod API URLs).
