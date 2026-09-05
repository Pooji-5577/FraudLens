# FraudLens backend

The backend contains the FastAPI application, model pipeline, payment-enforcement implementation, generated data, trained model, and evaluation artifacts.

The model is a hackathon prototype trained and evaluated only on seeded synthetic data. Its metrics are not production accuracy claims, and it is not connected to live Razorpay fraud prevention or automatic payment decisions.

From the repository root:

```bash
pip install -r backend/requirements.txt
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

Health check: <http://localhost:8000/health>

`POST /datasets/score` runs the trained model on an uploaded transaction dataset and
returns scores, decisions, reasons, and dataset-level signal influence. It also tries to
store the parsed rows and outputs in Supabase. If Supabase migrations are not applied yet,
the response still includes model results with `storage_status: "unavailable"`.

API documentation: <http://localhost:8000/docs>

Regenerate the model and global SHAP artifact:

```bash
python -m backend.src.train
python -m backend.src.global_importance
```

Razorpay enforcement is restricted to Test Mode and requires explicit human actions from the reviewer interface.
