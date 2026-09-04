-- FraudLens durable review state.
-- The application uses the Supabase secret key only from its server processes.
-- Keep these tables unavailable to browser/public roles; the server is the API boundary.

create table if not exists public.processed_webhook_events (
    event_id text primary key,
    event_type text not null,
    payment_id text,
    processed_at timestamptz not null default now()
);

create table if not exists public.authorization_revocations (
    account_id text primary key,
    event_id text not null unique,
    revoked_at timestamptz not null default now()
);

create table if not exists public.payment_reviews (
    payment_id text primary key,
    order_id text,
    amount bigint not null check (amount >= 0),
    currency text not null,
    payment_status text not null,
    review_status text not null,
    fulfillment_status text not null default 'on_hold',
    risk_score double precision,
    evidence_json jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    decided_at timestamptz,
    decided_by text,
    decision text
);

create table if not exists public.enforcement_audit_log (
    id bigint generated always as identity primary key,
    timestamp timestamptz not null default now(),
    payment_id text not null references public.payment_reviews(payment_id),
    action text not null,
    actor text not null,
    outcome text not null,
    risk_score double precision,
    evidence_json jsonb,
    detail text
);

create index if not exists payment_reviews_review_status_idx
    on public.payment_reviews (review_status);
create index if not exists payment_reviews_created_at_idx
    on public.payment_reviews (created_at desc);
create index if not exists enforcement_audit_log_payment_id_idx
    on public.enforcement_audit_log (payment_id, id);

-- All four tables are in the exposed public schema, so enforce RLS and remove
-- Data API access for browser roles. The server-side secret key maps to the
-- service_role database role and receives only the operations this store needs.
alter table public.processed_webhook_events enable row level security;
alter table public.authorization_revocations enable row level security;
alter table public.payment_reviews enable row level security;
alter table public.enforcement_audit_log enable row level security;

revoke all on public.processed_webhook_events from anon, authenticated;
revoke all on public.authorization_revocations from anon, authenticated;
revoke all on public.payment_reviews from anon, authenticated;
revoke all on public.enforcement_audit_log from anon, authenticated;

grant select, insert, delete on public.processed_webhook_events to service_role;
grant select, insert, update on public.authorization_revocations to service_role;
grant select, insert, update on public.payment_reviews to service_role;
grant select, insert on public.enforcement_audit_log to service_role;
grant usage, select on sequence public.enforcement_audit_log_id_seq to service_role;
