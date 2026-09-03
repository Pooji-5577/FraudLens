# Fraud Spike Detector

An explainable, cost-aware fraud detector and merchant review workspace, with real Razorpay transaction connectivity for account review. Fraud scoring currently runs on a synthetic dataset engineered to include the signals that real-time scoring would need; connecting the trained model to enriched real Razorpay transactions is the next integration step.

The two current paths are deliberately separate:

- **Razorpay payment history:** read-only OAuth access loads real payment records for filtering, inspection, export, and transaction-grounded chat. These records are not scored by the fraud model.
- **Synthetic risk system:** generated data includes card/device history, IP and billing geography, merchant category, and labels. That data powers model training, held-out evaluation, the scoring API, SHAP explanations, and the clearly labelled mock risk demonstration.

The scoring API's `blocked` field is an internal synthetic-policy decision. It does not decline, capture, refund, or otherwise change a Razorpay payment.

## Quick start

Python 3.11 is the supported runtime.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
pytest
```

Run those commands from this `fraud-spike-detector/` directory. `python -m src.train` uses a fixed seed, regenerates all 50,000 transactions, rebuilds features, and makes a chronological 70/30 split. Within the 70% training window, it fits 180 XGBoost candidates on the earliest 80% and selects by PR-AUC on the latest 20%. It then refits the winner on the full training window, compares four final candidates once on the untouched test window, deploys the PR-AUC winner, and runs threshold search and SHAP against that winner.

On macOS, XGBoost also needs the OpenMP runtime (`brew install libomp`, or `conda install llvm-openmp` in a Conda environment).

Start the API for the synthetic scoring endpoints and dashboard AI chat, then start the dashboard:

```bash
# Terminal 1: scoring service
uvicorn api.main:app --reload

# Terminal 2: review dashboard
SCORING_API_URL=http://localhost:8000 \
streamlit run dashboard/app.py
```

### Connect a Razorpay account

The dashboard is protected by Razorpay's authorization-code OAuth flow and requests
read-only access. Create an application under **Partner Dashboard → Applications**, add
the exact dashboard URL as a redirect URI, and set these server-side values:

```bash
export RAZORPAY_CLIENT_ID="your-development-or-production-client-id"
export RAZORPAY_CLIENT_SECRET="your-client-secret"
export RAZORPAY_REDIRECT_URI="http://localhost:8501"
export RAZORPAY_MODE="test"  # use live with the production client
```

After approval, the app exchanges the authorization code on the server and opens the
payment-history dashboard. The Razorpay section supports an inclusive UTC date range and
paginates through the matching payments. Production redirect URIs must use HTTPS. For local automated
UI tests only, `RAZORPAY_AUTH_DISABLED=true` bypasses the connection gate; never enable
that setting in a deployed environment. For demos that should retain the connection step,
leave `RAZORPAY_MOCK_AUTH=true`: when Partner credentials are absent, the button opens a
clearly labelled mock session without credentials or Razorpay account data. Set it to
`false` to require real Razorpay configuration.

`SCORING_API_URL` defaults to `http://localhost:8000` and can be changed for a remote service. The connected-account dashboard uses it for grounded transaction chat; it does not send real Razorpay payments to the trained scorer.

The synthetic API scorer keeps prior transactions in memory so velocity and history features are point-in-time correct within a running process. Single requests must arrive in increasing timestamp order. `POST /score/batch` sorts each submitted batch chronologically before applying state, and equal-timestamp transactions cannot observe one another. Separate batches must still move forward in time. A fresh process starts with cold history, while a production service would hydrate state from a feature store.

## Optional humanized evidence reports

For a blocked transaction, the API can organize the exact scored feature values into a reviewer-facing evidence report. Python builds the evidence list directly from card/device velocity, country comparison, amount baseline, device history, and transaction recency. Azure OpenAI only writes the connecting 2–3 sentence summary; it cannot supply or modify evidence values. The prompt forbids speculation about customer identity, intent, or whether fraud occurred.

Use `.env.example` as a template and export these variables in the API process (or load your `.env` through your deployment platform):

```bash
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com"
export AZURE_OPENAI_DEPLOYMENT_NAME="your-chat-model-deployment"
export AZURE_OPENAI_API_VERSION="v1"
```

`AZURE_OPENAI_API_VERSION` defaults to the current stable `v1` Azure OpenAI API. No real credentials belong in the repository. Without complete Azure configuration, ordinary scoring is unchanged and the dashboard explains that narrative reports are unavailable. Provider timeouts are retried once; authentication, rate-limit, timeout, and other generation failures return the original score/flag/reasons normally and retain deterministic evidence with a safe failure status.

Request a report during synthetic single-transaction scoring with `POST /score?include_report=true`. Scores below the saved threshold are marked for monitoring; scores at or above it return `blocked: true` and `recommended_action: auto-block`. These values describe the synthetic policy simulation only and do not enforce any action in Razorpay.

Example report shape:

```json
{
  "status": "generated",
  "summary": "This payment was sent for review because recent card activity was elevated and its billing and IP countries differed. The amount also differed materially from the customer's prior spending baseline.",
  "evidence": [
    {
      "signal": "card_velocity",
      "detail": "20 earlier card transactions in 1 hour; 20 in 24 hours.",
      "values": {"count_1h": 20, "count_24h": 20}
    },
    {
      "signal": "geography",
      "detail": "Billing country IN and IP country RU do not match.",
      "values": {"billing_country": "IN", "ip_country": "RU", "mismatch": true}
    }
  ],
  "confidence_note": "This is a model-generated risk assessment for human review, not a determination of fraud.",
  "recommended_action": "auto-block"
}
```

The summary text is AI-generated and exists only to assist a human reviewer. It does not trigger the block, replace human judgment, establish customer intent, or provide chargeback proof by itself. This layer only narrates detector evidence and provides no fraud-evasion guidance.

## Transaction-aware AI chat

In the dashboard, **Ask AI about a transaction** answers only from the selected row's visible Razorpay payment fields. For real account payments it explicitly has no fraud score or decision. The synthetic scoring API can separately answer questions about an already-scored synthetic transaction using its score, threshold, reasons, and deterministic evidence. Chat requests retain at most eight recent messages.

The assistant is instructed to answer only from that context and to say when the requested information is unavailable. Transaction values are treated as untrusted data rather than instructions. The question and selected transaction context are sent to the configured Azure OpenAI service, so production deployments should apply their normal data-governance and retention controls.

## Detection logic

Every feature is computed from transactions with a timestamp strictly earlier than the row being scored. Transactions at identical timestamps cannot observe one another. Signals include card/device counts over 1-hour and 24-hour windows, billing/IP country mismatch, deviation from the user's prior amount mean and standard deviation, first-seen device status, and time since the user's prior transaction.

Raw labels remain imbalanced at roughly 1–2%. Exactly 2.5% of labels are flipped: 2.1 percentage points model delayed or missed fraud reports (false negatives), while 0.4 points model disputed legitimate payments recorded as fraud (false positives). This asymmetric noise preserves the requested observed prevalence. Training handles imbalance with `scale_pos_weight`; it never oversamples rows.

## Held-out results

The headline metrics below are populated after the verified 50,000-row run. They are calculated only on the most recent 30% of transactions, never on the training rows.

- Production model: **tuned, uncalibrated XGBoost**
- Chosen threshold: **0.26**
- Precision: **13.35%**
- Recall: **76.61%**
- F1: **22.74%**
- Brier score: **0.0462**
- PR-AUC: **0.2982**
- ROC-AUC: **0.8549**

At that threshold the 15,000-row held-out window contained 226 true positives, 1,467 false positives, 13,238 true negatives, and 69 false negatives. The low precision is reported deliberately rather than hidden: under the illustrative 100:1 missed-fraud/review cost ratio, the cheapest policy accepts a large review queue to recover more fraud.

### Why precision looks low

At the current 100:1 false-negative-to-false-positive cost ratio, 13.35% precision means roughly **87 of every 100 blocked transactions are false alarms**. This is a deliberate consequence of prioritizing catching fraud over minimizing customer friction, not a modeling failure. Adjusting the cost ratio in `src/evaluate.py` and rerunning threshold search moves the policy toward higher precision at the cost of missing more fraud; for example, the existing curve reaches 40.00% precision and 48.14% recall at threshold 0.81.

The cost model uses **$5 per false positive** (an illustrative blocked-payment friction cost) and **$500 per false negative** (a conservative illustrative average loss from an undetected fraudulent payment). At the selected threshold, total held-out cost is **$41,835**. These are explicit policy assumptions, not universal values. Change them in `src/evaluate.py` for a different operating context and retrain before deployment. Because the PR-AUC winner is uncalibrated, its score is an operating score rather than a literal fraud probability; this is reflected by its weaker Brier score.

Artifacts are written to `reports/metrics/evaluation.json`, `reports/metrics/threshold_curve.csv`, and plots for cost, precision/recall/F1, PR/ROC ranking, calibration, and model comparison under `reports/figures/`. The audit trail includes `model_comparison_initial.*`, `xgboost_calibration_impact_initial.json`, `xgboost_temporal_search.*`, and `model_comparison_tuned.*`.

### Fair XGBoost tuning and second comparison

The first comparison evaluated only calibrated, lightly tuned XGBoost. Re-evaluating that original model before calibration produced PR-AUC **0.2870** / ROC-AUC **0.8312**, versus **0.2657** / **0.8360** after isotonic calibration. Calibration improved ROC-AUC slightly, but its tied score levels reduced PR-AUC; ranking and probability trust are different objectives.

The second comparison searched all 180 combinations of three `scale_pos_weight` multipliers, five depths, three estimator counts, and four learning rates. Search used only the nested temporal validation window and optimized PR-AUC. The winning validation configuration used `scale_pos_weight=28.85` (`0.5×` the contemporaneous class ratio), `max_depth=3`, `n_estimators=300`, and `learning_rate=0.05`. After selection, the class-weight multiplier was reapplied to the full training ratio (`scale_pos_weight=28.43`) and XGBoost was refit on all training rows.

Every row below uses the same point-in-time features, full 70% training period, and newest untouched 30% test period. No model uses SMOTE or random splitting.

| Model | PR-AUC | ROC-AUC |
|---|---:|---:|
| Tuned XGBoost, uncalibrated | **0.2982** | 0.8549 |
| Tuned XGBoost, calibrated | 0.2891 | **0.8554** |
| Random forest | 0.2873 | 0.8466 |
| Logistic regression | 0.2683 | 0.8497 |

Tuned, uncalibrated XGBoost is the production pick because it has the highest held-out PR-AUC, the selection metric most sensitive to false-positive burden under severe imbalance. The calibrated variant remains available in the comparison for applications that value probability interpretation more than the small ranking loss. The original and tuned comparisons are both retained under `reports/metrics/` and `reports/figures/`.

## Explanations

Blocked rows receive the three largest positive SHAP contributors translated into reviewer language, such as recent card velocity, country inconsistency, an amount far outside prior spending, or a first-seen device. The numeric score and reason codes preserve evidence for review; they are not a claim that a customer committed fraud.

## Known limitations

- Real Razorpay payments are shown for account review but are not scored by the trained model; the Payments API does not provide all required device, IP-geography, identity, and history signals.
- Razorpay webhooks are not integrated. The dashboard reads payment history through the Payments API.
- No live Razorpay payment is blocked, declined, captured, or refunded by this project.
- Real fraud outcomes are not learned from Razorpay disputes; model labels are synthetic.
- OAuth `state` is kept in process memory. This is sufficient for a single demo process, but production needs durable user-bound state and multi-worker-safe storage.
- Slow-drip fraud that stays below velocity signals and resembles ordinary spend may be missed.
- A genuinely travelling customer, a VPN, shared cards, or a new phone can resemble risk and create false positives.
- Cold-start users have no reliable spending baseline; the API also needs a persistent feature store across restarts in production.
- Synthetic performance will not transfer directly to real payment traffic, which changes over time and needs drift, fairness, compliance, and model-risk monitoring.
- The cost-optimal threshold is selected on this demo's held-out set. A production launch should freeze policy on a validation window, then report once on a later untouched test window.

## Project layout

```text
data/generate_synthetic.py  reproducible defensive dataset generator
src/features.py             leakage-safe chronological features
src/tune.py                 180-candidate temporal XGBoost search
src/benchmark.py            fair four-model comparison and artifacts
src/calibration.py          out-of-time isotonic score calibration
src/train.py                temporal split, winner selection, and training
src/evaluate.py             metrics, calibration, and cost curve
src/explain.py              SHAP contributions and reason codes
src/score.py                stateful single/batch scoring
api/main.py                 POST /score and POST /score/batch
dashboard/app.py            Razorpay payment-history and synthetic mock-review interface
tests/                      leakage and metric sanity tests
```

## Structural reference

The pipeline shape and cost/calibration reporting were informed by [Financial Fraud Risk Engine](https://github.com/AmirhosseinHonardoust/Financial-Fraud-Risk-Engine). The idea of including a multi-model benchmark and broader evaluation visuals was prompted by [Arindam-GitH/FraudGuard-ML](https://github.com/Arindam-GitH/FraudGuard-ML), formerly referenced as Payment Fraud Detection. We did not adopt its random split or SMOTE approach because those conflict with this project's temporal-leakage and imbalance guardrails. Its repository also has no committed license file despite an MIT badge in its README, so no source code was copied. This implementation's generator, chronological state handling, features, benchmarking, scoring, and explanations were written independently.
