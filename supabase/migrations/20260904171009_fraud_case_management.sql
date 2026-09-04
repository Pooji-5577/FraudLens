-- FraudLens case-management storage for the synthetic/demo analyst workflow.
-- Deliberately separate from payment_reviews/enforcement_audit_log: those track
-- real, money-moving Test Mode enforcement, while these two tables track an
-- analyst's investigation status and notes on synthetic demo transactions.
-- Keep both tables unavailable to browser/public roles; the server is the API boundary.

create table if not exists public.fraud_cases (
    transaction_id text primary key,
    status text not null default 'open'
        check (status in ('open', 'under_investigation', 'confirmed_fraud', 'false_positive')),
    risk_score double precision,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    updated_by text
);

create table if not exists public.fraud_case_notes (
    id bigint generated always as identity primary key,
    transaction_id text not null references public.fraud_cases(transaction_id) on delete cascade,
    note text not null,
    author text,
    created_at timestamptz not null default now()
);

create index if not exists fraud_cases_status_idx
    on public.fraud_cases (status);
create index if not exists fraud_case_notes_transaction_id_idx
    on public.fraud_case_notes (transaction_id, id);

alter table public.fraud_cases enable row level security;
alter table public.fraud_case_notes enable row level security;

revoke all on public.fraud_cases from anon, authenticated;
revoke all on public.fraud_case_notes from anon, authenticated;

grant select, insert, update on public.fraud_cases to service_role;
grant select, insert on public.fraud_case_notes to service_role;
grant usage, select on sequence public.fraud_case_notes_id_seq to service_role;
