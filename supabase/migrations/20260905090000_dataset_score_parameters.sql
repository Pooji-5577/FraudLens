-- Preserve the reviewer-facing model parameters returned with each scored
-- transaction. Existing datasets remain readable; newly scored rows populate
-- these columns from the server-side feature pipeline.

alter table public.fraud_dataset_rows
    add column if not exists velocity double precision,
    add column if not exists ip_billing text,
    add column if not exists device text,
    add column if not exists amount_deviation double precision,
    add column if not exists hour smallint,
    add column if not exists status text,
    add column if not exists actual text;

alter table public.fraud_dataset_rows
    drop constraint if exists fraud_dataset_rows_hour_check;

alter table public.fraud_dataset_rows
    add constraint fraud_dataset_rows_hour_check
    check (hour is null or hour between 0 and 23);

create index if not exists fraud_dataset_rows_status_idx
    on public.fraud_dataset_rows (dataset_id, status);
