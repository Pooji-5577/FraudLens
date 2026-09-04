# FraudLens domain context

FraudLens is a merchant review workspace for Razorpay payments and a separately evaluated synthetic fraud model.

## Application layout

- **Frontend**: the Streamlit reviewer interface and its presentation-only mock workflow under `frontend/`.
- **Backend**: the FastAPI routes, scoring pipeline, enforcement implementation, generated data, model, and evaluation artifacts under `backend/`.
- **Cross-layer tests**: interface and integration tests under `tests/` exercise the seam between both sides.

## Core terms

- **Pending review**: a real Test Mode Razorpay payment received as `authorized` and held without capture until a human decides.
- **Approve and capture**: a human decision to capture the exact authorized amount and currency.
- **Confirm fraud**: a human decision not to capture an authorization and to stop fulfillment within FraudLens. Razorpay releases or auto-refunds the uncaptured authorization according to the merchant's configured timeout.
- **Refund and stop fulfillment**: a human decision to fully refund a payment that was already captured and stop fulfillment within FraudLens.
- **Review record**: FraudLens's local durable state for a real payment, including its immutable payment facts and human decision.
- **Audit entry**: an append-only durable record of a capture, refund, or fraud-confirmation attempt and result.
- **Synthetic score**: a model output calculated only when all required model inputs are available. It must never trigger a real money action automatically.
- **Simulated enforcement walkthrough**: a presentation-only, Streamlit-session workflow whose statuses and audit entries never call Razorpay or reuse the real enforcement module.

## Safety invariants

- Real enforcement is Test Mode only.
- Every capture, fraud confirmation, and refund is explicitly initiated by a human reviewer.
- Real Razorpay payments without the required enrichment fields display no model score or fabricated evidence.
- OAuth secrets, webhook secrets, and access tokens remain server-side and are never committed.
- Webhooks are verified over the raw request body before parsing or processing.
- Duplicate and out-of-order events must not repeat money movement or overwrite a completed decision.
- Simulated enforcement has no HTTP dependency and must remain visibly distinct from OAuth-connected review.
- Hosting and deployment are outside this repository's current scope.
