# Amazon Purchases — Churn & Agentic Recommendation Project

Interview-prep project on the [Amazon Purchases](https://github.com) research dataset:
1,850,717 transactions from 5,027 users (2018–2024), linked 1:1 to a demographic/behavioral survey.

## Dataset

- `../Amazon/amazon-purchases.csv` — one row per line item: Order Date, Price, Quantity, State, Title, ASIN, Category, Survey ResponseID
- `../Amazon/survey.csv` — one row per user: demographics + Amazon usage behavior + sensitive attributes (health, substance use, sexual orientation)
- `../Amazon/fields.csv` — data dictionary for survey questions
- Joined via `Survey ResponseID`

## Part 1: Churn Prediction (supervised, scikit-learn)

Files: `churn_features.py` → `churn_model.py`

**Churn definition and why it's non-trivial**: most users' last recorded purchase
clusters tightly around Dec 2022–Mar 2023 — almost certainly when each person's
data extraction/survey consent happened, not organic churn. Defining churn as "no
purchase after a user's own last date" would be a right-censoring tautology (there's
no data after collection stopped for them, by definition). Instead we use a fixed
global time split shared across all users:

- Feature window: purchases on/before **2022-06-01** (`R1`)
- Label window: purchases in **(2022-06-01, 2023-03-15]** (`R2`, ~90th percentile of
  last-purchase dates, so almost every user's data extends at least that far)
- `churn = 1` if a user has zero purchases in the label window, else `0`
- Eligibility gate: ≥5 orders and ≥90 days of tenure before `R1` (established
  customers only — a brand-new account never had a chance to churn)

Result: 4,930 eligible users, 4.65% churn rate.

**Features**: RFM-style (recency, frequency, monetary), category diversity, avg
order value, tenure, and a 90-day order-count "momentum" signal — all computed
strictly from data ≤ `R1`. Demographics from the survey (age, income, education,
household size, Amazon usage frequency) are merged in. **Sensitive survey fields
(substance use, health conditions, sexual orientation) are deliberately excluded**
as churn predictors — statistically useful or not, they're unrelated to the
business question and shouldn't be baked into the model.

**Models**: Logistic Regression and Random Forest, both `class_weight="balanced"`,
preprocessing (`StandardScaler` / `OneHotEncoder`) fit on train split only.

**Results** (test set, n=1,233):

| Model | PR-AUC | ROC-AUC | Macro-F1 |
|---|---|---|---|
| Logistic Regression | 0.485 | 0.928 | 0.621 |
| Random Forest | 0.358 | 0.923 | 0.692 |

(Random baseline PR-AUC ≈ churn rate ≈ 0.046, so both models show strong lift.)
Top features by importance: 90-day order count, recency, prior-90-day order count,
avg days between orders — the model leans on real behavioral signal, not demographics.

## Part 2: RAG + Recommendation Agent

Files: `build_product_index.py` → `recommend_tools.py` → `agent.py`

**Retrieval**: local `sentence-transformers` (`all-MiniLM-L6-v2`) embeddings over
unique products (title + category), normalized so cosine similarity == dot
product at query time. Two scope cuts kept the one-time build tractable on a
laptop CPU: only products with ≥2 historical purchases are indexed (876K →
~219K — singleton purchases carry the weakest similarity signal anyway), and
text is capped at 64 tokens (a handful of outlier titles ran past 1,000 words
and were dominating wall-clock time via attention cost).

**Recommendation**: content-based, not collaborative filtering — per-user
purchase history is too sparse against an ~219K-product catalog for item-item
CF to have enough signal. Instead each user's "taste vector" is the
recency-weighted centroid (180-day half-life) of the embeddings of things
they've already bought; recommendations are the nearest unseen products to
that centroid.

**Agent**: Claude Agent SDK tools (`search_products`, `similar_products`,
`get_user_purchase_history`, `recommend_for_user`, `category_stats`) wrap
`recommend_tools.py` so the agent is grounded in retrieved data, never
inventing ASINs/prices. The retrieval/recommendation logic itself works and is
tested independent of the agent (`test_recommend_tools.py`, no API key
needed); running `agent.py` requires `ANTHROPIC_API_KEY` in the environment.

## Part 3: FastAPI Backend

File: `api.py` — REST endpoints for a Lovable (or any) frontend:

| Endpoint | Purpose |
|---|---|
| `GET /users/{id}/churn-risk` | Precomputed churn probability for a scored user |
| `GET /users/{id}/history` | Recent purchase history |
| `GET /users/{id}/recommendations` | Personalized recommendations |
| `GET /search?q=...` | Free-text semantic product search |
| `GET /products/{asin}/similar` | Nearest-neighbor products |
| `POST /chat` | Conversational agent (needs `ANTHROPIC_API_KEY`) |

CORS is wide open (`allow_origins=["*"]`) for Lovable's dynamic preview origin —
tighten to the deployed frontend's real origin before this goes anywhere near
production.

## Part 4: Analytics Dashboard (Streamlit)

Files: `build_dashboard_aggregates.py` → `dashboard_data.py` → `dashboard_qa.py` → `streamlit_app.py`

Three fixed views (Overview, Churn Analysis, Customer & Transactions — all
plotly charts, no API calls) plus an **Ask a Question** tab backed by Claude.

**Why this app never loads the raw 1.85M-row purchases file**: Streamlit
Community Cloud's free tier has ~1GB RAM, and pandas typically uses 3-5x a
CSV's on-disk size once loaded — the 300MB raw file would blow that budget.
`build_dashboard_aggregates.py` precomputes three small rollup tables
(category/state/monthly, <100KB total) once, locally; the deployed app only
ever touches those plus the 1MB `churn_dataset.csv`.

**Ask a Question**: a plain Anthropic API tool-use loop (`dashboard_qa.py`),
not the Claude Agent SDK — the Agent SDK shells out to the `claude` CLI, built
for interactive coding sessions, not a stateless one-shot "answer this
question with a validated chart spec" call. Claude gets one tool
(`query_table`, a safe parameterized pandas aggregation — never arbitrary code
execution) and must return output conforming to a JSON schema
(`chart_type`/`title`/`categories`/`values`/`insight`) via
`output_config.format`, so the app never has to guess at the response shape.

### Deploying to Streamlit Community Cloud

1. Push this `Code files` folder to a GitHub repo (see Setup below — `git init`,
   commit, create a repo on github.com, push).
2. At [share.streamlit.io](https://share.streamlit.io), sign in with GitHub → **New app**
   → pick the repo/branch → set **Main file path** to `streamlit_app.py`.
3. Under **Advanced settings**, set **Python dependencies file** to
   `requirements-dashboard.txt` (a lighter subset — skips `torch`/`sentence-transformers`,
   which this app doesn't use and which would slow the build).
4. Under **Advanced settings → Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
   (Set a usage/spend limit on this key in the Anthropic Console first — see
   the project chat history for how.)
5. Deploy. You'll get a permanent `https://<app-name>.streamlit.app` URL.

## Setup

```
# uses the shared venv at ../../.venv
../../.venv/Scripts/python.exe -m pip install -r requirements.txt

../../.venv/Scripts/python.exe churn_features.py            # -> churn_dataset.csv
../../.venv/Scripts/python.exe churn_model.py                # -> churn_model.joblib
../../.venv/Scripts/python.exe build_dashboard_aggregates.py # -> dashboard_data/*.csv
../../.venv/Scripts/python.exe build_product_index.py        # -> product_embeddings.npy, product_metadata.parquet (~30-40 min)
../../.venv/Scripts/python.exe test_recommend_tools.py       # smoke test, no API key needed

set ANTHROPIC_API_KEY=...                                     # required for agent.py / POST /chat / Ask a Question
../../.venv/Scripts/python.exe -m uvicorn api:app --reload
../../.venv/Scripts/python.exe -m streamlit run streamlit_app.py
```
