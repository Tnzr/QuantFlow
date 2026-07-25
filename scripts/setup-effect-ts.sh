#!/bin/bash
# Scaffold Effect-TS backend for QuantFlow
# Usage: bash scripts/setup-effect-ts.sh
# Creates: backend-effect/ with Bun + Effect-TS + Drizzle ORM

set -e

BACKEND_DIR="backend-effect"

if [ -d "$BACKEND_DIR" ]; then
    echo "Backend directory already exists: $BACKEND_DIR"
    echo "Remove it first or run: cd $BACKEND_DIR && bun install"
    exit 1
fi

echo "=== QuantFlow Effect-TS Backend Scaffold ==="
echo ""

# Check Bun
if ! command -v bun &> /dev/null; then
    echo "❌ Bun is not installed."
    echo "   Install: curl -fsSL https://bun.sh/install | bash"
    exit 1
fi
echo "✅ Bun $(bun --version)"

# Create directory structure
mkdir -p $BACKEND_DIR/src/{layers/{auth,api,portfolio,orders,risk,ml,market-data},db/migrations,__tests__}
mkdir -p $BACKEND_DIR/docker

# package.json
cat > $BACKEND_DIR/package.json << 'JSONEOF'
{
  "name": "quantflow-backend",
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "bun run --watch src/main.ts",
    "build": "bun build src/main.ts --outdir dist",
    "start": "bun run dist/main.js",
    "test": "bun test",
    "db:generate": "drizzle-kit generate",
    "db:push": "drizzle-kit push",
    "db:migrate": "bun run src/db/migrate.ts"
  },
  "dependencies": {
    "effect": "^3.13",
    "@effect/platform": "^0.68",
    "@effect/platform-node": "^0.66",
    "drizzle-orm": "^0.38",
    "drizzle-kit": "^0.30",
    "postgres": "^3.4",
    "elysia": "^1.2"
  },
  "devDependencies": {
    "bun-types": "latest",
    "@types/node": "^22"
  }
}
JSONEOF

# tsconfig.json
cat > $BACKEND_DIR/tsconfig.json << 'TSCEOF'
{
  "compilerOptions": {
    "target": "ESNext",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "outDir": "./dist",
    "rootDir": "./src",
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["src/**/*.ts"],
  "exclude": ["node_modules", "dist"]
}
TSCEOF

# Main entry point
cat > $BACKEND_DIR/src/main.ts << 'MAINEOF'
import { Effect, Console, Schedule } from "effect"
import { Elysia } from "elysia"

// API server
const app = new Elysia()
  .get("/health", () => ({ status: "ok", timestamp: new Date().toISOString() }))
  .get("/api/portfolio", () => ({
    equity: 100000,
    cash: 25000,
    positions: [],
    signals: [],
  }))
  .listen(3000)

console.log(`🦊 QuantFlow Effect-TS backend running at http://localhost:${app.server?.port}`)

// Export for testing
export { app }
MAINEOF

# Auth layer
cat > $BACKEND_DIR/src/layers/auth/index.ts << 'AUTHEOF'
import { Effect, Layer, Redacted } from "effect"

export interface AuthConfig {
  readonly jwtSecret: Redacted.Redacted<string>
  readonly tokenExpirySeconds: number
}

export const AuthConfig = Layer.succeed(
  {} as AuthConfig,
  {
    jwtSecret: Redacted.make(process.env.JWT_SECRET || "dev-secret-change-me"),
    tokenExpirySeconds: 3600,
  }
)

export const verifyToken = (token: string): Effect.Effect<{ userId: string }, Error> =>
  Effect.gen(function* (_) {
    if (!token || token.length < 10) {
      return yield* _(Effect.fail(new Error("Invalid token")))
    }
    // Placeholder: real JWT verification here
    return { userId: "demo-user" }
  })
AUTHEOF

# Portfolio layer
cat > $BACKEND_DIR/src/layers/portfolio/index.ts << 'PORTEOF'
import { Effect } from "effect"

export interface Position {
  ticker: string
  shares: number
  avgPrice: number
  unrealizedPnl: number
  unrealizedPnlPct: number
}

export interface PortfolioState {
  equity: number
  cash: number
  positions: Position[]
}

export const getPortfolio = (): Effect.Effect<PortfolioState, Error> =>
  Effect.succeed({
    equity: 100000,
    cash: 25000,
    positions: [
      { ticker: "AAPL", shares: 100, avgPrice: 195.40, unrealizedPnl: 460, unrealizedPnlPct: 2.35 },
    ],
  })
PORTEOF

# ML Gateway layer
cat > $BACKEND_DIR/src/layers/ml/index.ts << 'MLEOF'
import { Effect } from "effect"

export interface MLPrediction {
  ticker: string
  predictedReturn: number
  sigma: number
  direction: "BUY" | "SELL" | "HOLD"
  confidence: number
}

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://localhost:8000"

export const getPrediction = (ticker: string, bars: number[][]): Effect.Effect<MLPrediction, Error> =>
  Effect.tryPromise({
    try: async () => {
      const resp = await fetch(`${ML_SERVICE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker, bars }),
      })
      if (!resp.ok) throw new Error(`ML service returned ${resp.status}`)
      return resp.json()
    },
    catch: (err) => new Error(`ML prediction failed: ${err}`),
  })
MLEOF

# DB schema
cat > $BACKEND_DIR/src/db/schema.ts << 'DBEOF'
import { pgTable, serial, text, integer, timestamp, real } from "drizzle-orm/pg-core"

export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  email: text("email").notNull().unique(),
  createdAt: timestamp("created_at").defaultNow(),
})

export const trades = pgTable("trades", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").references(() => users.id),
  ticker: text("ticker").notNull(),
  side: text("side").notNull(), // "buy" | "sell"
  quantity: integer("quantity").notNull(),
  price: real("price").notNull(),
  orderType: text("order_type").notNull(), // "market" | "limit"
  status: text("status").notNull(), // "filled" | "canceled" | "pending"
  filledAt: timestamp("filled_at").defaultNow(),
})

export const signals = pgTable("signals", {
  id: serial("id").primaryKey(),
  ticker: text("ticker").notNull(),
  predictedReturn: real("predicted_return").notNull(),
  sigma: real("sigma").notNull(),
  direction: text("direction").notNull(),
  confidence: real("confidence").notNull(),
  generatedAt: timestamp("generated_at").defaultNow(),
})
DBEOF

# Docker
cat > $BACKEND_DIR/docker-compose.yml << 'DCEOF'
services:
  backend:
    image: oven/bun:latest
    working_dir: /app
    volumes:
      - .:/app
    ports:
      - "3000:3000"
    environment:
      - JWT_SECRET=${JWT_SECRET:-dev-secret}
      - ML_SERVICE_URL=http://ml-service:8000
      - DATABASE_URL=postgres://user:pass@db:5432/quantflow
    command: bun run --watch src/main.ts
    depends_on:
      - db

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: quantflow
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  ml-service:
    image: python:3.11-slim
    working_dir: /app
    volumes:
      - ../../checkpoints:/app/checkpoints:ro
      - ../../data:/app/data:ro
    ports:
      - "8000:8000"
    command: python -m uvicorn api:app --host 0.0.0.0 --port 8000

volumes:
  pgdata:
DCEOF

# .env
cat > $BACKEND_DIR/.env << 'ENVEOF'
JWT_SECRET=dev-secret-change-me-in-production
ML_SERVICE_URL=http://localhost:8000
DATABASE_URL=postgres://user:pass@localhost:5432/quantflow
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
ENVEOF

# .gitignore
cat > $BACKEND_DIR/.gitignore << 'GIEOF'
node_modules/
dist/
.env
*.log
GIEOF

echo ""
echo "=== Scaffold Complete ==="
echo ""
echo "Next steps:"
echo "  cd $BACKEND_DIR && bun install"
echo "  bun run dev"
echo ""
echo "API endpoints:"
echo "  GET  http://localhost:3000/health"
echo "  GET  http://localhost:3000/api/portfolio"
echo ""
echo "ML Gateway (calls Python):"
echo "  POST http://localhost:3000/ml/predict"
