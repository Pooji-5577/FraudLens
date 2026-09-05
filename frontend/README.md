# FraudLens frontend

The frontend is the Streamlit reviewer interface.

From the repository root:

```bash
pip install -r frontend/requirements.txt
SCORING_API_URL=http://localhost:8000 \
streamlit run frontend/app.py --server.port 8501
```

Open <http://localhost:8501>. The synthetic demo session starts automatically; its 80 UI fixtures use deterministic rule-based display scores rather than XGBoost predictions. Mock enforcement changes only Streamlit session state and makes no Razorpay request.

The frontend expects the backend scoring API at `SCORING_API_URL` for evidence reports and transaction chat.
The CSV tester also uses that API to run the trained model. After a run, the uploaded
scores become the active source for the transaction table, summary cards, and signal panel.
When Supabase storage is available the rows are saved there; if migrations are missing, the
UI still shows the model results and marks database saving as pending.

The scored-results table exposes backend parameters for velocity, IP/billing geography,
device history, amount deviation, hour, model status, and an optional `Actual` label.
Uploads can use friendly headers such as `Txn`, `IP/billing`, `Amt. dev.`, `Hour`, and
`Actual`; uploaded `Score` and `Status` values are ignored because the backend recomputes
the model decision.

The Overview page begins with **Demo every case in one session**. Its shortcuts cover low-risk,
review-band, high-risk, false-positive, and false-negative synthetic rows; the full evidence report;
the policy audit; grounded AI chat; and all three session-only enforcement
outcomes. The report is the key demo moment: the AI explanation is shown beside the deterministic
signals that support it.

The optional real-data boundary lives in the top bar: when Partner OAuth values are configured it
starts a Test Mode connection and shows payment history for review. Those rows are not assigned a
synthetic model score until the required enrichment signals are integrated.
