# FraudLens frontend

The frontend is the Streamlit reviewer interface.

From the repository root:

```bash
pip install -r frontend/requirements.txt
SCORING_API_URL=http://localhost:8000 \
streamlit run frontend/app.py --server.port 8501
```

Open <http://localhost:8501>. The synthetic demo session starts automatically; mock enforcement changes only Streamlit session state and makes no Razorpay request.

The frontend expects the backend scoring API at `SCORING_API_URL` for evidence reports and transaction chat.

The Review queue begins with **Demo every case in one session**. Its shortcuts cover low-risk,
review-band, high-risk, false-positive, and false-negative synthetic rows; the full evidence report;
the policy audit; grounded AI chat; manual model scoring; and all three session-only enforcement
outcomes. The report is the key demo moment: the AI explanation is shown beside the deterministic
signals that support it.

The **Razorpay account** view is the optional real-data boundary: when Partner OAuth values are
configured it starts a Test Mode connection and shows payment history for review. Those rows are not
assigned a synthetic model score until the required enrichment signals are integrated.
