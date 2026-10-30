-- QuantFlow election sentiment, point-in-time agent state and signed proposals.
-- Writes use the service-role key from the QuantFlow backend; RLS blocks anon/auth roles.

create table if not exists public.election_events (
  id bigint generated always as identity primary key,
  created_at timestamptz not null default now(),
  source text not null,
  ticker_or_topic text not null,
  polarity_score double precision not null,
  rationale jsonb,
  title text,
  url text,
  dedupe_key text unique
);
create index if not exists election_events_topic_time on public.election_events (ticker_or_topic, created_at);

create table if not exists public.sentiment_snapshots (
  id bigint generated always as identity primary key,
  created_at timestamptz not null default now(),
  source text not null,
  ticker_or_topic text not null,
  polarity_score double precision not null,
  probability double precision,
  n_items integer not null default 0,
  rationale jsonb
);
create index if not exists sentiment_snapshots_topic_time on public.sentiment_snapshots (ticker_or_topic, created_at);

create table if not exists public.agent_state_snapshots (
  id bigint generated always as identity primary key,
  created_at timestamptz not null default now(),
  as_of timestamptz not null,
  run_id text not null,
  ticker text not null,
  horizon text not null,
  base_confidence double precision not null,
  agent_confidence double precision not null,
  final_confidence double precision not null,
  override_applied boolean not null default false,
  volatility_triggered boolean not null default false,
  sentiment_multiplier double precision not null default 1,
  state jsonb
);
create index if not exists agent_state_ticker_asof on public.agent_state_snapshots (ticker, as_of);

create table if not exists public.allocation_proposals (
  id bigint generated always as identity primary key,
  proposal_id uuid not null unique,
  created_at timestamptz not null default now(),
  ticker text not null,
  structure text not null,
  payload jsonb not null,
  signature text not null,
  public_key text not null,
  status text not null default 'pending'
);
create index if not exists allocation_proposals_ticker_time on public.allocation_proposals (ticker, created_at);

alter table public.election_events enable row level security;
alter table public.sentiment_snapshots enable row level security;
alter table public.agent_state_snapshots enable row level security;
alter table public.allocation_proposals enable row level security;
