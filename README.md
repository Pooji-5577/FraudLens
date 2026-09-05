# FraudLens

[![tests](https://github.com/Pooji-5577/FraudLens/actions/workflows/tests.yml/badge.svg)](https://github.com/Pooji-5577/FraudLens/actions/workflows/tests.yml)

FraudLens is a human-in-the-loop payment risk review workspace built for the Razorpay Buildathon AI Risk Manager track. Most hackathon fraud detectors stop at a score; FraudLens explains every flagged synthetic decision in plain language and can generate a full reviewer-ready evidence report grounded in the exact signals behind that decision. The AI writes only the explanation—the score, reason codes, and evidence values remain deterministic and independently inspectable.

> **Prototype limitation:** FraudLens is a hackathon prototype, not a live Razorpay fraud prevention system. Its model is trained and evaluated only on generated synthetic transactions and labels. The metrics below do not establish production accuracy, safety, calibration, or business impact.

It combines:

- An explainable fraud model trained and evaluated on synthetic payment data.
- A reviewer dashboard for inspecting transactions and evidence.
- A safe, presentation-only payment enforcement walkthrough.
- An optional Razorpay Test Mode integration for authorized-payment review, capture, and refund.

FraudLens does **not** automatically capture, block, or refund a real payment. A human reviewer must make every payment decision.

## The project in one minute

```text
Payment data
    │
    ├── Synthetic UI fixtures ──> rule-based demo score ──> review walkthrough
    │
    ├── Complete model input ──> FastAPI ──> XGBoost score + SHAP reasons
    │
    └── Razorpay Test Mode ──> authorized payment ──> review queue
                                                   ├─> approve and capture
                                                   ├─> withhold capture
                                                   └─> refund if already captured
```

These paths are intentionally separate. Synthetic model data contains the device, geography, velocity, history, and fraud-label fields required by the model. Razorpay's normal Payments API does not provide all of those fields, so FraudLens does not invent a score for real Razorpay payments.

## Full system architecture

FraudLens is made up of four related but deliberately separate flows: connecting to Razorpay, training the fraud model, holding an authorized payment for review, and answering questions about one transaction.

### 1. Razorpay connection

```text
Dashboard redirects to Razorpay /authorize
  client_id + redirect_uri + scope=read_write + state
        |
        v
Merchant approves access on Razorpay's hosted consent screen
        |
        v
Server exchanges the authorization code with the Partner app client_secret
  client_secret never reaches the browser
        |
        v
GET /v1/payments with Bearer access_token
  paginated with from / to / count / skip
        |
        v
Dashboard shows payment history
  real Razorpay rows are not sent into the fraud model
```

The current connection requests `read_write` because the same Test Mode OAuth token supports the explicit capture and refund controls. The payment-history view itself is read-only. OAuth, access tokens, and client secrets remain server-side; see `frontend/razorpay_oauth.py`.

### 2. Model training

```text
50,000 synthetic labeled transactions
  about 1.8% synthetic fraud in the seeded dataset
        |
        v
Point-in-time features
  card/device velocity, geography mismatch, amount deviation,
  device history, customer recency
        |
        v
Chronological 70/30 split
  train on earlier data, evaluate once on the newest held-out window
        |
        v
Temporally tune 145 XGBoost candidates on training/validation data
  select candidates by temporal PR-AUC, then lock the threshold by validation cost
        |
        v
Held-out result
  precision 17.20%, recall 74.92% at threshold 0.23
  SHAP provides the strongest contributing signals for each flag
```

The model is trained only on reproducible synthetic data. Real Razorpay payment rows do not receive a fabricated FraudLens score because the Payments API does not provide the complete feature set required by the model.

### 3. Blocking a payment

```text
Order is created through Razorpay Orders API with Manual Capture
  read_write access is required for later capture/refund actions
        |
        v
Customer pays
  payment becomes authorized, not captured
        |
        v
payment.authorized webhook reaches POST /webhooks/razorpay
  verify HMAC-SHA256 over the raw body with a separate webhook secret
  deduplicate x-razorpay-event-id
        |
        v
Human reviewer clicks Confirm fraud in the dashboard
        |
        v
Capture is withheld
  FraudLens makes no capture API call and stops local fulfillment
        |
        v
Razorpay releases/auto-refunds the uncaptured authorization after its
configured timeout, or a human requests POST /v1/payments/{id}/refund
if the payment was already captured
```

There is no Razorpay endpoint that directly blocks a payment. The safety control is manual capture: FraudLens holds the authorization by not calling capture. This path is restricted to Razorpay Test Mode and requires a human reviewer; the model score cannot move money automatically. See `backend/api/main.py` and `backend/src/razorpay_enforcement.py`.

### 4. Ask about a transaction

```text
Reviewer selects one transaction and asks a question
        |
        v
Backend builds context only for that transaction
  scored synthetic row: visible fields + score/reasons/evidence
  real Razorpay row: visible payment fields only, marked not scored
        |
        v
Context is sent as untrusted data, never as instructions
  the last 8 conversation messages are retained
        |
        v
Azure OpenAI configured deployment
  answers strictly from the supplied transaction context
        |
        v
Answer is shown to the reviewer
  missing information is stated instead of invented
```

Azure OpenAI is optional and the deployment name is configured with `AZURE_OPENAI_DEPLOYMENT_NAME`; the code does not assume a specific base model. The grounding and prompt-boundary logic lives in `backend/src/report.py`.

## What is real and what is simulated?

| Area | Data | What happens |
|---|---|---|
| Mock dashboard | Generated UI fixtures | Demonstrates filtering, deterministic rule-based demo scores, evidence, reviewer actions, and audit entries. These scores are not XGBoost predictions, and no Razorpay request is made. |
| Fraud model | Reproducible synthetic dataset | Trains and evaluates an XGBoost prototype using chronological, point-in-time features. |
| Optional Razorpay integration | Razorpay Test Mode only | Loads Test Mode payment records and supports explicit human-approved capture/refund actions. |
| AI explanation | Deterministic evidence plus optional Azure OpenAI text | AI summarizes existing evidence. It cannot change the score or initiate a payment action. |

> **Important:** The mock enforcement controls only change Streamlit session state. They never call Razorpay and never move money.

## Dashboard tour

The interface opens in demo mode and follows an analyst-shaped navigation rail: one merged Overview command center, then three focused fraud-review pages. Overview starts with a highlighted **Demo every case in one session** tour so a judge can see the entire product path without guessing where to click.

### Complete demo path: all cases

The deterministic mock dataset includes every labelled risk outcome: low-risk/allowed, review-band,
high-risk/report, false-positive, and false-negative rows. Use the tour shortcuts in this order:

1. Scroll Overview for the pipeline steps strip, the transaction trend chart, the high-risk queue
   (with inline **Mark legitimate** / **Confirm fraud** / **Investigate →** actions), the fraud-spike
   banner, held-out model metrics, the cost-of-fraud trade-off, and global signal importance.
2. **Investigate top alert**. This is the key moment: on Transaction investigation, click
   **Generate full evidence report** — an AI-written explanation is immediately followed by
   deterministic evidence values, so the demo shows more than a score. Then mark the case
   **Confirm fraud** (or another status) and add an analyst note.
3. **Explore all transactions** to filter the full synthetic set by date, amount, status, geography,
   and device.
4. **Open case management** to see every case an analyst has marked, filterable by status, each
   linking back into Transaction investigation.
5. Return to Overview and exercise all three session-only enforcement outcomes in the "Payments
   awaiting a capture or refund decision" panel: approve and capture, withhold capture after
   confirming fraud, and refund a payment captured before review.
6. Use the real-account path only to show payment-history review. Real Razorpay rows deliberately have
   no model score or synthetic error label until the required enrichment signals are integrated.

### Demo mode

The workspace opens directly in a clearly labelled synthetic demo session. No Razorpay credentials are required for the walkthrough.

The 80 payments in this default walkthrough come from `frontend/app.py::mock_payments`. Their display scores use a fixed rule-based formula so every UI state remains deterministic. They are separate from both `backend/data/transactions.csv` and the trained XGBoost model.

### 1. Overview

One merged fraud command center that replaces the former Review queue, Fraud overview, and Fraud
alerts pages:

- **Pipeline steps** — a four-step strip (Input data → Detect spikes → View results → Take action)
  summarizing the loaded window in real counts.
- **Transaction trend** — daily total volume vs. transactions flagged high risk.
- **High-risk transactions** — the top five open alerts, each with inline **Mark legitimate**,
  **Confirm fraud**, and **Investigate →** actions; an expander below reveals the full filterable
  queue (high-priority vs. review bands, triggering signals, recommended next step).
- **Fraud spike banner** — appears automatically when a day's flagged count materially exceeds the
  window's daily average.
- Below that: measured synthetic held-out precision, recall, F1, false-positive rate, PR-AUC, an
  explicitly descriptive threshold-sensitivity explorer, global SHAP signal importance, and the
  synthetic decision audit. If generated artifacts are unavailable, the dashboard says so instead of
  displaying fallback results.
- **Payments awaiting a capture or refund decision** — practice the payment decision using two
  scenarios:
  - **Authorized — awaiting review**
    - **Approve & capture** changes the simulated status to `Captured`.
    - **Confirm fraud & release authorization** changes it to `Capture withheld` and stops fulfillment.
  - **Captured before review**
    - **Refund & stop fulfillment** changes the simulated status to `Refunded`.

  Every mock action immediately adds a session audit entry containing the reviewer, timestamp,
  payment ID, action, and resulting status.

### 2. Transaction explorer

Browse generated or connected-account payments. You can:

- Search by payment ID, order ID, email, or contact.
- Filter by payment status, method, currency, risk status, geography match/mismatch, and device known/new.
- Show only international payments.
- Change the UTC date range.
- Download the filtered table as CSV.

Captured totals remain separated by currency; INR and USD are never added together as though they were interchangeable.

### 3. Transaction investigation

Pick any transaction and see its full fraud-signal breakdown, its risk score, the AI evidence-report generator (available once a transaction reaches the high-priority threshold), case-status controls (mark under investigation, confirmed fraud, or false positive, with analyst notes), and a grounded chat that answers questions using only that transaction's own fields.

### 4. Case management

Every transaction an analyst has marked, filterable by status, with the current status, risk score, who last updated it, and a shortcut back into Transaction investigation. Case status and notes are stored durably in Supabase (`fraud_cases` / `fraud_case_notes`), separate from the real Test Mode enforcement tables — marking a synthetic case never touches a real payment.

The connected-account state itself (demo vs. Razorpay Test Mode, account ID, a Disconnect button) lives in the top bar rather than its own nav page. A connected account exposes real payment history for review, filtering, export, and grounded chat in Transaction explorer; it does not receive a synthetic model score until the required enrichment signals are integrated.

## Quick start: safe mock demo

### Requirements

- Python 3.11
- macOS, Linux, or Windows with a Python environment
- On macOS, XGBoost may require OpenMP: `brew install libomp`

### 1. Create the environment

Run these commands from the repository directory:

```bash
cd fraud-spike-detector
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For separate deployments, install `frontend/requirements.txt` in the Streamlit service and `backend/requirements.txt` in the FastAPI service. The root requirements file combines both and adds the test runner for local development.

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Start the API

```bash
python -m uvicorn backend.api.main:app --reload --port 8000
```

API documentation will be available at [http://localhost:8000/docs](http://localhost:8000/docs).

### 3. Start the dashboard

Open another terminal in the same directory:

```bash
SCORING_API_URL=http://localhost:8000 \
streamlit run frontend/app.py --server.port 8501
```

Open [http://localhost:8501](http://localhost:8501); the synthetic demo workspace opens automatically.

No Razorpay credential is required for this walkthrough.

## Run the tests

Run everything:

```bash
python -m pytest -q
```

Expected result for the current repository:

```text
103 passed
```

Run only the dashboard tests:

```bash
python -m pytest -q tests/test_dashboard_ui.py
```

Run the simulated-enforcement isolation tests:

```bash
python -m pytest -q tests/test_mock_enforcement.py \
  tests/test_dashboard_ui.py -k "mock_enforcement or fraud_confirmation or captured_edge_case"
```

These tests patch the HTTP client and fail if a mock capture/refund action attempts an outbound request.

## Model results

The dataset contains 50,000 generated transactions. The newest 30%, or 15,000 transactions, form the untouched held-out test window.

| Metric | Result |
|---|---:|
| Model | Tuned, uncalibrated XGBoost |
| Decision threshold | 0.23, selected on validation data |
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

### Why is precision only 17.20%?

The example policy assigns a cost of `$5` to reviewing a legitimate payment and `$500` to missing synthetic fraud. Under that illustrative 100:1 assumption, the validation-selected threshold favors recall: it catches 74.92% of labelled fraud in the held-out window but produces many false alarms.

Put plainly, approximately 83 of every 100 synthetic payments flagged at this threshold are false positives. This prototype is suitable for demonstrating **human review**, not automatic payment blocking.

The cost assumptions are illustrative, not Razorpay business values. The **Overview** page lets a reviewer change them and inspect held-out threshold sensitivity. That test-set what-if view is descriptive; it does not change the saved validation-selected threshold.

At the saved `$5` / `$500` assumption, the measured synthetic held-out error cost is **$42,320**. This is a unit-cost calculation (`FP × $5 + FN × $500`), not observed financial loss or savings.

## How the model works

FraudLens computes every historical feature using transactions strictly earlier than the payment being scored. Rows with the same timestamp cannot observe one another.

Model signals include:

- Card and device transaction counts over one-hour and 24-hour windows.
- Billing-country and IP-country mismatch.
- Deviation from the customer's earlier spending pattern.
- Whether the device was previously seen.
- Time since the customer's previous transaction.
- Time-of-day behavior.
- Interactions between unusual amount, new device, geography mismatch, velocity, and rapid repeats.

Training uses a chronological 70/30 split rather than a random split. Inside the earlier 70%, expanding temporal folds select XGBoost hyperparameters and a final validation window selects the operating threshold. The threshold is then locked and applied once to the newest 30% for final evaluation. Logistic regression, random forest, and calibrated XGBoost results on that test window are descriptive benchmarks only; they do not select the saved model. The pipeline does not use SMOTE or duplicate minority rows.

The generated dataset and model search are reproducible with seed `20260903`. Training regenerates `backend/data/transactions.csv`, so that CSV is seeded development input—not captured Razorpay traffic. The generator deliberately encodes broad fraud correlations and label noise; good performance on those patterns may not transfer to real fraud, changing customer behavior, unseen merchants, or adversarial activity.

To regenerate the dataset, tune the XGBoost candidate, lock its validation threshold, evaluate it, and rebuild artifacts:

```bash
python -m backend.src.train
python -m backend.src.global_importance
```

Training evaluates 145 XGBoost candidates across two expanding temporal validation windows. It tunes class weighting, depth, tree count, learning rate, child weight, row/feature sampling, split penalty, L1/L2 regularization, and maximum weight updates, then compares the selected candidate with calibrated XGBoost, random forest, and logistic regression.

The current seeded run saved these XGBoost parameters: `max_depth=4`, `n_estimators=300`, `learning_rate=0.04`, `min_child_weight=8`, `subsample=0.85`, `colsample_bytree=0.70`, `gamma=0.3`, `reg_alpha=0`, `reg_lambda=10`, `max_delta_step=1`, and `scale_pos_weight=22.7405`. This class weight addresses label imbalance during fitting; it is not a hand-authored feature-priority table. Feature influence is learned by the trees, and the dashboard reports measured global SHAP group importance separately.

### Where predictions appear

- `backend/src/train.py` writes the fitted XGBoost artifact and its validation-selected threshold to `backend/models/fraud_detector.joblib`.
- `backend/src/score.py` loads that artifact, builds stateful point-in-time features, predicts probabilities, compares each score with the saved threshold, and attaches SHAP-backed reasons to flagged rows.
- FastAPI exposes that scorer through `POST /score` and `POST /score/batch` for inputs containing all required user, card, device, amount, time, and geography fields.
- The dashboard's **Held-out model performance** panel reads the generated evaluation and SHAP artifacts. Its default alert rows are the separate rule-based UI fixtures described above; they do not pass through `/score`.
- Connected Razorpay payment rows are not sent to XGBoost because the Payments API records do not contain all required model inputs. They display no fabricated model score.

## Evidence and AI explanations

For every scored transaction, the response states whether its score met the saved review threshold. For flagged synthetic transactions, SHAP also identifies the strongest positive model contributors, and Python converts the exact values into deterministic evidence such as elevated velocity, geography mismatch, unusual amount, or a first-seen device. SHAP is an attribution of model behavior, not a causal explanation or proof of fraud.

Azure OpenAI is optional. When configured, it writes a short explanation connecting those facts. It cannot supply evidence values, change the model result, or trigger capture/refund. If AI configuration is missing or unavailable, the score and deterministic evidence continue to work.

Copy `.env.example` to `.env` and configure these values only if narrative reports and chat are needed:

```dotenv
AZURE_OPENAI_API_KEY=replace-with-your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com
AZURE_OPENAI_DEPLOYMENT_NAME=your-chat-model-deployment
AZURE_OPENAI_API_VERSION=v1
```

Use the exact endpoint, deployment name, and supported API version shown in the Azure portal for your deployed chat model. Restart the FastAPI backend after changing `.env`. The `.env` file is ignored by Git; never paste its API key into frontend code or commit it.

## Supabase durable review storage

FraudLens stores real-payment review records, webhook de-duplication,
authorization revocations, the append-only enforcement audit, and analyst-uploaded
transaction datasets with their model outputs in the Supabase project. The bundled
synthetic demo fixtures and model artifacts remain local and reproducible.

Case management (analyst status and notes on synthetic demo transactions) uses
the same Supabase project through two additional, unrelated tables —
`fraud_cases` and `fraud_case_notes` — created by
`supabase/migrations/20260904171009_fraud_case_management.sql`. There is no
sqlite fallback for case management: `SUPABASE_URL`/`SUPABASE_SECRET_KEY` must
be set for Case management and the case-status controls on Transaction
investigation to work, independent of `FRAUDLENS_STORAGE`.

### Setup

1. Open the Supabase SQL Editor and run all migrations in order:
   `supabase/migrations/20260904145002_supabase_review_storage.sql` and
   `supabase/migrations/20260904171009_fraud_case_management.sql`, followed by
   `supabase/migrations/20260905042852_dataset_scoring_storage.sql`.
2. Add the server-only values to `.env`:

   ```dotenv
   FRAUDLENS_STORAGE=supabase
   SUPABASE_URL=https://your-project-ref.supabase.co
   SUPABASE_SECRET_KEY=sb_secret_...
   ```

   The publishable key is not needed because the Streamlit and FastAPI
   processes access Supabase server-side. Never put the secret key in frontend
   JavaScript, source control, or a browser-exposed environment variable.
3. Start the API and verify the schema:

   ```bash
   curl http://localhost:8000/health/storage
   ```

   A configured project returns `{"status":"ok","backend":"supabase"}`.
   The default `FRAUDLENS_STORAGE=sqlite` path remains available for isolated
   local tests and the mock walkthrough.

## Optional Razorpay Test Mode integration

The mock demo is the recommended presentation path. The integration below is optional and must use Test Mode credentials only.

### Safety requirements

1. Create or use a Razorpay Partner application in **Test Mode**.
2. Request and manually approve the `read_write` OAuth scope in the Partner Dashboard. Write access is needed for capture and refund.
3. Configure payments through the Razorpay Orders API with **Manual Capture**. If Razorpay captures a payment automatically, FraudLens cannot hold it for review.
4. Configure a Test Mode webhook for `payment.authorized` and `payment.captured`.
5. Keep every credential server-side and outside Git.

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

The Partner Dashboard scope approval and redirect URI configuration are manual steps outside this repository.

### Real Test Mode review flow

```text
Razorpay payment authorized
        │
        ▼
POST /webhooks/razorpay
        │
        ├─ verify X-Razorpay-Signature over the raw body
        ├─ deduplicate x-razorpay-event-id
        └─ store pending review in Supabase
                    │
                    ▼
              Human reviewer
              ├─ approve -> capture exact order amount/currency
              ├─ fraud -> withhold capture and stop fulfillment
              └─ already captured -> request full refund
```

An uncaptured authorization is not immediately voided by FraudLens. It waits for Razorpay's configured automatic-refund timeout. Razorpay documents a maximum three-day authorization hold, so the merchant must configure and understand that timeout.

Capture and refund actions re-fetch the current payment first. An already captured or refunded payment becomes a recorded no-op instead of a repeated mutation. Refund requests also use a stable idempotency key.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Service health check |
| `POST` | `/score` | Score one synthetic transaction |
| `POST` | `/score/batch` | Chronologically score a synthetic batch |
| `POST` | `/datasets/score` | Score an uploaded dataset, return results and signal influence, and persist rows in Supabase when storage is available |
| `POST` | `/report/{transaction_id}` | Build evidence for a retained scored transaction |
| `POST` | `/demo-report` | Build the dashboard's synthetic evidence report |
| `POST` | `/chat/{transaction_id}` | Ask about a retained scored transaction |
| `POST` | `/preview-chat` | Ask about dashboard-visible payment fields |
| `POST` | `/webhooks/razorpay` | Receive and verify Razorpay Test Mode webhooks |
| `GET` | `/health/storage` | Verify the configured review-storage backend and schema |

Single-transaction online scoring keeps recent history in memory. Uploaded batches are
isolated from one another while preserving chronological feature state within each dataset.

## Security and safety

- `.env` is ignored by Git; `.env.example` contains placeholders only.
- Supabase secret keys stay server-side; public roles are denied access to the review tables by RLS and grants.
- Razorpay client secrets, access tokens, and webhook secrets stay server-side.
- Webhooks are rejected unless the raw-body HMAC signature is valid.
- Duplicate webhook events are ignored.
- OAuth and enforcement reject live-mode use.
- Mock enforcement has no import of the real enforcement module and no HTTP dependency.
- Real payment mutations always require a human click and reviewer identity.
- The model score never directly triggers Razorpay capture or refund.
- CSV exports are protected against spreadsheet-formula injection.
- The explanation layer is defense-only and does not provide fraud-evasion guidance.

To check that a local `.env` is not tracked:

```bash
git check-ignore -v .env
git ls-files .env
```

The second command should produce no output.

## Known limitations / not yet implemented

- Model training and reported metrics use synthetic data, so performance will not transfer directly to real payment traffic.
- Real Razorpay Payments API records are not scored because required enrichment fields are unavailable.
- There is no production webhook integration for real-time model scoring or automatic enforcement; the narrow Test Mode handler records review lifecycle events and clears a local session when Razorpay sends `account.app.authorization_revoked`.
- There is no production real-payment blocking or model-driven capture control. Test Mode capture/refund actions are explicit, human-approved controls only.
- Fulfillment status is local; no warehouse or order-management system is connected.
- The Supabase migrations must be applied before `FRAUDLENS_STORAGE=supabase` can serve real review events or persist uploaded datasets. CSV scoring still returns model results while storage is unavailable.
- OAuth `state` and mock walkthrough state are process/session-local.
- Scoring history is held in memory and is lost on API restart.
- Real disputes and chargebacks are not fed back into training.
- A production launch would require enriched data, calibration review, drift and fairness monitoring, privacy controls, shared storage, and operational approval.

## Project structure

```text
frontend/app.py                     Streamlit reviewer interface and navigation
frontend/redesign.css               Dashboard visual theme
frontend/mock_enforcement.py        Session-only simulated payment actions
frontend/processing.py              UI-safe filtering, reports, and API adapters
frontend/razorpay_oauth.py          Test Mode OAuth and payment-history client
frontend/requirements.txt           Frontend-only runtime dependencies
backend/api/main.py                 FastAPI scoring, reports, chat, and webhook routes
backend/data/generate_synthetic.py  Reproducible synthetic dataset generator
backend/src/features.py             Chronological leakage-safe feature engineering
backend/src/train.py                Training, selection, and artifact generation
backend/src/tune.py                 Temporal XGBoost search
backend/src/evaluate.py             Metrics, thresholds, and cost curve
backend/src/explain.py              SHAP evidence and reason codes
backend/src/global_importance.py    Held-out global SHAP signal importance
backend/src/score.py                Stateful single and batch scoring
backend/src/razorpay_enforcement.py Webhook state, audit log, capture, and refund service
backend/src/review_store.py        SQLite/Supabase durable review-storage adapters
backend/requirements.txt            Backend-only runtime dependencies
supabase/migrations/                RLS-protected Supabase review-storage schema
backend/models/                     Saved trained model
backend/reports/metrics/            Saved evaluation and threshold artifacts
backend/reports/figures/            Evaluation charts
tests/                              Cross-layer unit, integration, security, and UI tests
```

## Design principles

1. **Evidence before automation.** Scores are accompanied by inspectable reasons.
2. **Human authority over money movement.** No model-driven capture or refund.
3. **Honest boundaries.** Synthetic and Razorpay data are clearly separated.
4. **Time-respecting evaluation.** Future transactions never leak into earlier features.
5. **Visible trade-offs.** False positives and their cost are reported, not hidden.

## Acknowledgements

The reporting structure was informed by the public [Financial Fraud Risk Engine](https://github.com/AmirhosseinHonardoust/Financial-Fraud-Risk-Engine), and the multi-model comparison was inspired by [FraudGuard-ML](https://github.com/Arindam-GitH/FraudGuard-ML). FraudLens uses its own generator, chronological feature pipeline, training process, scoring logic, enforcement code, and dashboard implementation.
