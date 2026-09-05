-- Durable storage for analyst-uploaded transaction spreadsheets and their
-- model outputs. Files are parsed by the server; queryable rows and scores are
-- stored here so no Supabase secret is ever exposed to the Streamlit client.

create table if not exists public.fraud_datasets (
    id uuid primary key,
    filename text not null,
    row_count integer not null check (row_count >= 0),
    status text not null default 'processing'
        check (status in ('processing', 'completed', 'failed')),
    error_message text,
    created_at timestamptz not null default now(),
    completed_at timestamptz
);

create table if not exists public.fraud_dataset_rows (
    dataset_id uuid not null references public.fraud_datasets(id) on delete cascade,
    row_number integer not null check (row_number >= 1),
    transaction_id text not null,
    transaction_timestamp timestamptz not null,
    user_id text not null,
    device_id text not null,
    card_id text not null,
    amount double precision not null check (amount >= 0),
    billing_country text not null,
    ip_country text not null,
    ip_address text,
    merchant_category text not null,
    uploaded_velocity_per_hour double precision,
    score double precision not null check (score >= 0 and score <= 1),
    flagged boolean not null,
    blocked boolean not null,
    reasons jsonb not null default '[]'::jsonb,
    created_at timestamptz not null default now(),
    primary key (dataset_id, row_number)
);

create index if not exists fraud_datasets_created_at_idx
    on public.fraud_datasets (created_at desc);
create index if not exists fraud_dataset_rows_transaction_id_idx
    on public.fraud_dataset_rows (transaction_id);
create index if not exists fraud_dataset_rows_flagged_idx
    on public.fraud_dataset_rows (dataset_id, flagged, score desc);

alter table public.fraud_datasets enable row level security;
alter table public.fraud_dataset_rows enable row level security;

revoke all on public.fraud_datasets from anon, authenticated;
revoke all on public.fraud_dataset_rows from anon, authenticated;

grant select, insert, update on public.fraud_datasets to service_role;
grant select, insert on public.fraud_dataset_rows to service_role;
