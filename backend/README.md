# FraudLens backend

The backend contains the FastAPI application, model pipeline, payment-enforcement implementation, generated data, trained model, and evaluation artifacts.

From the repository root:

```bash
pip install -r backend/requirements.txt
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

Health check: <http://localhost:8000/health>

API documentation: <http://localhost:8000/docs>

Regenerate the model and global SHAP artifact:

```bash
python -m backend.src.train
python -m backend.src.global_importance
```

Razorpay enforcement is restricted to Test Mode and requires explicit human actions from the reviewer interface.
