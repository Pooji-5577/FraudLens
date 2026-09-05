# FraudLens

[![tests](https://github.com/Pooji-5577/FraudLens/actions/workflows/tests.yml/badge.svg)](https://github.com/Pooji-5577/FraudLens/actions/workflows/tests.yml)

FraudLens is a payment-risk review workspace built for the Razorpay Buildathon AI Risk Manager track. It helps an analyst:

- Find suspicious transactions.
- See the signals behind a fraud score.
- Generate a plain-language evidence report.
- Record an investigation status and notes.
- Demonstrate human-approved capture and refund decisions in Razorpay Test Mode.

The model produces the score and evidence. AI can explain that evidence, but it cannot change the score or take a payment action.

> [!IMPORTANT]
> FraudLens is a hackathon prototype, not a production fraud-prevention system. The model is trained and evaluated on generated data. FraudLens never automatically captures, blocks, or refunds a real payment.

## Quick start

### Requirements

- Python 3.11
- macOS, Linux, or Windows
- OpenMP for XGBoost on macOS: `brew install libomp`

### 1. Install the project

Run these commands from the directory that contains this repository:

```bash
cd fraud-spike-detector
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Start the API

```bash
python -m uvicorn backend.api.main:app --reload --port 8000
```

The API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

### 3. Start the dashboard

Open a second terminal in the repository, activate the environment, and run:

```bash
SCORING_API_URL=http://localhost:8000 \
streamlit run frontend/app.py --server.port 8501
```

Open [http://localhost:8501](http://localhost:8501).

The app starts in safe demo mode. You do not need Razorpay, Supabase, or Azure OpenAI credentials to browse the demo transactions and model results. Saving case statuses and notes requires the Supabase setup described below.

## What to try first

The sidebar has four pages:

1. **Overview** — See transaction trends, high-risk alerts, model metrics, and the simulated payment-decision workflow.
2. **Transaction explorer** — Search, filter, inspect, and export transactions. You can also upload a CSV and run the trained model.
3. **Transaction investigation** — Review one transaction's score, evidence, case status, notes, report, and grounded chat.
4. **Case management** — View and filter transactions that analysts marked for investigation, confirmed fraud, or false positive.

For a short demo:

1. Open **Overview** and inspect the high-risk transactions.
2. Select **Investigate** on an alert.
3. Generate the full evidence report.
4. After configuring Supabase, change the case status and add a note.
5. Open **Case management** to see the saved case.
6. Return to **Overview** to try the simulated capture, hold, and refund actions.

## What is real and what is simulated?

FraudLens keeps its demo, model, and Razorpay flows separate.

| Area | Data source | Behavior |
|---|---|---|
| Dashboard demo | 80 generated UI fixtures | Uses deterministic rule-based display scores. Actions update only the current Streamlit session. |
| Fraud model | 50,000 reproducible synthetic transactions | Trains and evaluates XGBoost using chronological features and synthetic fraud labels. |
| CSV scoring | Analyst-uploaded model-ready data | Sends the batch to FastAPI, calculates XGBoost scores and SHAP evidence, and optionally stores results in Supabase. |
| Razorpay integration | Razorpay Test Mode payments | Loads real Test Mode payment records and supports explicit human-approved capture or refund actions. |
| AI explanation | Existing transaction fields and deterministic evidence | Summarizes supplied facts. It cannot invent missing evidence or initiate a payment action. |

Razorpay payment records are not automatically sent to the model. The Payments API does not contain every device, geography, velocity, and customer-history field required by the trained model, so FraudLens does not create a misleading score for those rows.

## How the system works

```text
Synthetic data or model-ready CSV
        |
        v
FastAPI scoring service
        |
        +--> chronological feature calculation
        +--> XGBoost probability
        +--> saved review threshold
        +--> SHAP-backed reasons and evidence
        |
        v
Streamlit reviewer dashboard
        |
        +--> investigate transaction
        +--> create or update case
        +--> add analyst notes
        +--> generate grounded explanation
```

The optional Razorpay flow is separate:

```text
Razorpay Test Mode authorization
        |
        v
Verified payment.authorized webhook
        |
        v
Pending human review
        |
        +--> approve and capture
        +--> confirm fraud and withhold capture
        +--> refund if already captured
```

There is no Razorpay API operation called "block payment." FraudLens demonstrates a hold by using Manual Capture and not capturing an authorization until a reviewer approves it.

## Dashboard pages

### Overview

Overview is the command center. It includes:

- A four-step review pipeline.
- Daily transaction and high-risk trends.
- A high-risk transaction queue with analyst actions.
- A fraud-spike notice when flagged volume rises above the window average.
- Held-out model metrics and threshold sensitivity.
- Global SHAP signal importance.
- A synthetic decision audit.
- Session-only payment capture, hold, and refund examples.

### Transaction explorer

Use this page to:

- Search by transaction, order, email, or contact.
- Filter by date, amount, status, payment method, currency, geography, and device state.
- Show only international payments.
- Upload a model-ready CSV.
- Run the model and inspect returned parameters, scores, statuses, and reasons.
- Download the filtered results.

Currency totals remain separate. For example, INR and USD are never added into one misleading total.

### Transaction investigation

This page focuses on one transaction. It shows:

- Transaction and customer details.
- Risk score and individual fraud signals.
- A reviewer-ready evidence report for high-priority transactions.
- Case status controls and analyst notes.
- Grounded chat that uses only the selected transaction's visible fields and evidence.

### Case management

This page lists cases created during investigation. Analysts can filter cases by status, priority, assignee, transaction ID, or customer and reopen the selected transaction for further review.

Case status and notes use the `fraud_cases` and `fraud_case_notes` Supabase tables. They are separate from Razorpay enforcement records, so marking a synthetic case does not affect a real payment.

## Model summary

The model is trained on 50,000 generated transactions. The newest 30%, or 15,000 transactions, are reserved as the held-out test window.

| Metric | Held-out result |
|---|---:|
| Model | Tuned, uncalibrated XGBoost |
| Decision threshold | 0.23 |
| Precision | 17.20% |
| Recall | 74.92% |
| F1 | 27.97% |
| False-positive rate | 7.24% |
| PR-AUC | 0.3203 |
| ROC-AUC | 0.8594 |
| Brier score | 0.0368 |

At this threshold:

| Outcome | Transactions |
|---|---:|
| True positives | 221 |
| False positives | 1,064 |
| True negatives | 13,641 |
| False negatives | 74 |

### How to interpret these results

The validation policy assigns an illustrative cost of `$5` to reviewing a legitimate payment and `$500` to missing synthetic fraud. That assumption favors recall over precision.

In practical terms, about 83 of every 100 synthetic transactions flagged at this threshold are false positives. This is why FraudLens is designed for human review and not automatic payment blocking.

These costs and results are demonstration values. They are not Razorpay business metrics and do not prove production accuracy.

### Model features and training

FraudLens calculates historical features using only transactions that happened before the row being scored. Transactions with the same timestamp cannot observe one another.

Signals include:

- Card and device transaction counts over one-hour and 24-hour windows.
- Billing-country and IP-country mismatch.
- Deviation from the customer's earlier spending pattern.
- Whether the device was previously seen.
- Time since the customer's previous transaction.
- Time-of-day behavior.
- Interactions between amount, device, geography, velocity, and rapid repeats.

Training uses a chronological 70/30 split. XGBoost parameters are selected with expanding temporal validation folds, and the operating threshold is selected on a final validation window before one evaluation on the held-out test window.

The generator and training search use seed `20260903`.

To rebuild the dataset, model, evaluation files, and global importance artifact:

```bash
python -m backend.src.train
python -m backend.src.global_importance
```

The saved model is written to `backend/models/fraud_detector.joblib`.

## Evidence reports and chat

For a flagged model-scored transaction, FraudLens:

1. Finds the strongest positive SHAP contributors.
2. Converts those values into deterministic evidence, such as unusual velocity, geography mismatch, amount deviation, or a new device.
3. Optionally asks Azure OpenAI to summarize the supplied evidence in plain language.

SHAP explains model behavior; it is not proof that fraud occurred.

Azure OpenAI is optional. Without it, scoring and deterministic evidence still work.

To enable generated reports and chat, copy `.env.example` to `.env` and configure:

```dotenv
AZURE_OPENAI_API_KEY=replace-with-your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com
AZURE_OPENAI_DEPLOYMENT_NAME=your-chat-model-deployment
AZURE_OPENAI_API_VERSION=v1
```

Use the endpoint, deployment name, and API version shown in the Azure portal. Restart FastAPI after changing `.env`.

## Supabase storage

Supabase can store:

- Razorpay review records.
- Webhook event IDs for duplicate protection.
- Authorization revocations.
- The append-only enforcement audit.
- Uploaded datasets and model outputs.
- Analyst case statuses and notes.

The synthetic fixtures and trained model artifacts remain local.

### Setup

1. Run these migrations in the Supabase SQL Editor, in order:

   ```text
   supabase/migrations/20260904145002_supabase_review_storage.sql
   supabase/migrations/20260904171009_fraud_case_management.sql
   supabase/migrations/20260905042852_dataset_scoring_storage.sql
   supabase/migrations/20260905090000_dataset_score_parameters.sql
   ```

2. Add the server-only settings to `.env`:

   ```dotenv
   FRAUDLENS_STORAGE=supabase
   SUPABASE_URL=https://your-project-ref.supabase.co
   SUPABASE_SECRET_KEY=sb_secret_...
   ```

3. Start the API and verify the connection:

   ```bash
   curl http://localhost:8000/health/storage
   ```

A working connection returns:

```json
{"status":"ok","backend":"supabase"}
```

Keep `SUPABASE_SECRET_KEY` on the server. Do not expose it in frontend JavaScript, browser environment variables, or source control.

The default `FRAUDLENS_STORAGE=sqlite` mode remains available for isolated local testing and the mock walkthrough. Case management requires the Supabase case tables.

## Optional Razorpay Test Mode integration

Use this integration only for Razorpay Test Mode. The mock demo is the simplest and safest presentation path.

### Requirements

1. Create a Razorpay Partner application in Test Mode.
2. Approve the `read_write` OAuth scope in the Partner Dashboard.
3. Configure Orders API payments with Manual Capture.
4. Configure Test Mode webhooks for `payment.authorized` and `payment.captured`.
5. Keep all credentials server-side and outside Git.

### Environment variables

```dotenv
RAZORPAY_CLIENT_ID=replace-with-your-test-client-id
RAZORPAY_CLIENT_SECRET=replace-with-your-test-client-secret
RAZORPAY_REDIRECT_URI=http://localhost:8501
RAZORPAY_MODE=test
RAZORPAY_WEBHOOK_SECRET=replace-with-your-webhook-secret
RAZORPAY_ENFORCEMENT_DB=backend/data/razorpay_enforcement.sqlite3
RAZORPAY_MOCK_AUTH=false
RAZORPAY_AUTH_DISABLED=false
```

The OAuth redirect URI and scope approval must also be configured in the Razorpay Partner Dashboard.

### Review behavior

- The OAuth client secret and access token remain on the server.
- Webhook signatures are verified against the raw request body.
- Duplicate webhook event IDs are ignored.
- Capture and refund actions re-fetch the current payment before changing it.
- Repeated actions against an already captured or refunded payment become recorded no-ops.
- Refunds use a stable idempotency key.
- Every payment mutation requires an explicit human action.

An uncaptured authorization is not immediately cancelled by FraudLens. Razorpay releases or refunds it according to the account's configured authorization timeout.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Check whether the API is running. |
| `GET` | `/health/storage` | Check the configured storage backend and schema. |
| `POST` | `/score` | Score one model-ready synthetic transaction. |
| `POST` | `/score/batch` | Score a chronological model-ready batch. |
| `POST` | `/datasets/score` | Score and optionally store an uploaded dataset. |
| `POST` | `/report/{transaction_id}` | Build evidence for a retained scored transaction. |
| `POST` | `/demo-report` | Build an evidence report for a dashboard demo transaction. |
| `POST` | `/chat/{transaction_id}` | Ask about a retained scored transaction. |
| `POST` | `/preview-chat` | Ask about fields visible in the dashboard. |
| `POST` | `/webhooks/razorpay` | Receive verified Razorpay Test Mode webhooks. |
| `GET` | `/cases` | List fraud-review cases. |
| `GET` | `/cases/{transaction_id}` | Read one case and its notes. |
| `POST` | `/cases/{transaction_id}/status` | Create or update a case status. |
| `POST` | `/cases/{transaction_id}/notes` | Add an analyst note to a case. |

Batch scoring returns reviewer-friendly fields such as `velocity`, `ip_billing`, `device`, `amount_deviation`, `hour`, `status`, and optional `actual` labels in addition to the engineered model fields.

Uploaded datasets are isolated from one another while preserving chronological feature history within each dataset. Single-transaction scoring keeps recent history in API memory.

## Run the tests

Run the full suite:

```bash
python -m pytest -q
```

Run only the dashboard tests:

```bash
python -m pytest -q tests/test_dashboard_ui.py
```

Run the mock-enforcement isolation checks:

```bash
python -m pytest -q tests/test_mock_enforcement.py \
  tests/test_dashboard_ui.py -k "mock_enforcement or fraud_confirmation or captured_edge_case"
```

These checks fail if a simulated capture or refund attempts an outbound payment request.

## Security boundaries

- `.env` is ignored by Git.
- OAuth tokens, API keys, Supabase secrets, and webhook secrets stay server-side.
- Supabase review tables deny public-role access through grants and row-level security.
- Razorpay webhooks require a valid HMAC signature.
- OAuth and enforcement reject live-mode use.
- Mock payment actions have no real enforcement or HTTP dependency.
- The model cannot trigger capture or refund.
- CSV exports are protected against spreadsheet-formula injection.
- Transaction context is treated as untrusted data when generating explanations.

Check that `.env` is not tracked:

```bash
git check-ignore -v .env
git ls-files .env
```

The second command should return no output.

## Project structure

```text
frontend/
  app.py                    Streamlit dashboard and navigation
  redesign.css              Dashboard styling
  processing.py             Filtering, reports, and API adapters
  mock_enforcement.py       Session-only payment actions
  razorpay_oauth.py         Test Mode OAuth and payment client

backend/
  api/main.py               FastAPI routes
  data/                     Synthetic data and generator
  models/                   Saved trained model
  reports/                  Metrics and evaluation charts
  src/features.py           Chronological feature engineering
  src/train.py              Model training and selection
  src/score.py              Single and batch scoring
  src/explain.py            SHAP evidence and reason codes
  src/review_store.py       SQLite and Supabase storage adapters
  src/razorpay_enforcement.py
                             Webhook, audit, capture, and refund logic

supabase/migrations/         Supabase schema and security policies
tests/                       Unit, integration, security, and UI tests
```

For separate deployments, install `frontend/requirements.txt` in the Streamlit service and `backend/requirements.txt` in the FastAPI service. The root `requirements.txt` contains both plus local test dependencies.

## Known limitations

- Training data and reported metrics are synthetic.
- Real Razorpay payment rows are not model-scored without additional enrichment data.
- There is no production webhook pipeline for real-time model scoring.
- Test Mode capture and refund actions require a human click.
- Fulfillment status is local; no warehouse or order-management system is connected.
- Supabase migrations must be applied before durable review and dataset storage can work.
- OAuth state, mock state, and online scoring history are process-local.
- API restarts clear in-memory scoring history.
- Real disputes and chargebacks are not used for retraining.
- Production use would require privacy controls, calibration review, drift and fairness monitoring, operational approval, and shared resilient infrastructure.

## Design principles

1. **Evidence before automation** — Every score should have inspectable reasons.
2. **Humans control money movement** — No model-driven capture or refund.
3. **Clear data boundaries** — Synthetic and Razorpay data stay separate.
4. **Time-aware evaluation** — Future transactions never influence earlier features.
5. **Visible trade-offs** — False positives and costs are reported honestly.

## Acknowledgements

The reporting structure was informed by the public [Financial Fraud Risk Engine](https://github.com/AmirhosseinHonardoust/Financial-Fraud-Risk-Engine), and the model comparison was inspired by [FraudGuard-ML](https://github.com/Arindam-GitH/FraudGuard-ML).

FraudLens uses its own generated dataset, chronological feature pipeline, training process, scoring logic, enforcement code, and dashboard implementation.
