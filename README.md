# Agentic RAG & AI-Powered Analytics for Amazon Purchase Data

An **agentic AI / RAG portfolio project** built on the Amazon Purchases research
dataset: 1,850,717 line-item transactions from 5,027 survey respondents (2018–2024),
linked 1:1 to a demographic and behavioral survey. Two independent agentic systems sit
on top of a supervised churn model and a lightweight retrieval index — no hand-coded
if/else chatbots, no hallucinated numbers.

## Agentic systems at a glance

| System | Framework | Pattern | Grounded via |
|---|---|---|---|
| **Product recommendation agent** (`agent.py`) | Claude Agent SDK (MCP tool server) | Multi-turn tool-calling agent that decides which of 5 tools to call, in what order, from a free-text request | Local semantic vector index (sentence-transformers, 219K products) + per-user purchase history — the agent can *only* answer from retrieved data, never invents ASINs/prices |
| **Analytics Q&A agent** (`dashboard_qa.py`) | Anthropic API, tool use + structured outputs | Stateless one-shot agent: calls a parameterized `query_table` tool as many times as needed, then must emit a response conforming to a JSON schema | Live pandas aggregation over the churn dataset — every number in an answer traces back to an actual tool call, not model memory |

Both are **RAG in the general sense** (retrieval-augmented generation: the model's
answer is conditioned on data fetched at request time, not parametric knowledge) even
though only the first uses vector embeddings — the second retrieves via structured
query rather than similarity search, which is the right retrieval mechanism when the
underlying data is already tabular instead of unstructured text.

The project has four parts that build on each other:

1. **Churn prediction** — supervised binary classification (scikit-learn), the labeled
   ground truth that both the API and the analytics agent serve
2. **RAG + recommendation agent** — semantic search and personalized recommendations,
   orchestrated by the Claude Agent SDK (agentic system #1 above)
3. **FastAPI backend** — REST layer wrapping (1) and (2) for a frontend to consume
4. **Streamlit analytics dashboard** — charts + the Claude-powered natural-language
   Q&A agent over the churn/customer data (agentic system #2 above), deployed as a
   public web app

---

## Dataset

| File | Rows | Description |
|---|---|---|
| `../Amazon/amazon-purchases.csv` | 1,850,717 | One row per line item: Order Date, Purchase Price Per Unit, Quantity, Shipping Address State, Title, ASIN/ISBN, Category, Survey ResponseID |
| `../Amazon/survey.csv` | 5,027 | One row per respondent: demographics (age, income, education, gender, state), Amazon usage behavior, and sensitive attributes (substance use, health conditions, sexual orientation) |
| `../Amazon/fields.csv` | 23 | Data dictionary for the survey questions |

The two main tables join perfectly on `Survey ResponseID` — every purchaser has a survey
response and vice versa (5,027 users in both, 0 orphans either direction).

---

## Part 1 — Churn Prediction

**Files:** `churn_features.py` → `churn_model.py`
**Type:** supervised binary classification, scikit-learn

### The label problem (why this isn't a trivial "day since last order" cutoff)

Plotting each user's *last recorded purchase date* reveals it clusters tightly in a
Dec 2022–Mar 2023 window almost regardless of when that user started buying. This is
almost certainly when each person's data extraction / survey consent happened — not
organic disengagement. If churn were defined as "no purchase after this user's own
last date," every user would trivially "churn," because by construction there is no
data past the point their feed was cut off. That's a right-censoring tautology, a
classic pitfall in survival-analysis-style problems.

**Fix — use one fixed clock for every user instead of each user's own end date:**

- `R1 = 2022-06-01` (feature cutoff) — nothing after this date may be used as a feature
- `R2 = 2023-03-15` (label horizon) — the ~90th percentile of last-purchase dates, so
  almost every user's data extends at least this far, minimizing how many people get
  mislabeled "churned" purely because their personal data feed ended early
- `churn = 1` if a user has **zero** purchases in `(R1, R2]`, else `0`
- **Eligibility gate:** ≥5 orders and ≥90 days of tenure before `R1` — excludes
  brand-new accounts that never had a fair chance to churn

**Result:** 4,930 / 5,027 users eligible, **4.65% churn rate** (a believable,
moderately imbalanced rate — not the near-100% the naive definition would produce).

### Features (all computed strictly from data on/before `R1` — no leakage)

| Feature | What it captures |
|---|---|
| `n_orders`, `total_spend`, `avg_order_value` | Frequency / monetary (RFM) |
| `recency_days` | Days since last order as of `R1` |
| `tenure_days` | Account age as of `R1` |
| `n_categories` | Category diversity |
| `avg_days_between_orders` | Purchase cadence |
| `orders_last_90d`, `orders_prior_90d`, `momentum` | 90-day order-count trend — a cheap substitute for real time-series modeling (see below) |
| `Q-demos-age`, `-income`, `-education`, `Q-amazon-use-hh-size`, `-howmany`, `-how-oft` | Demographics + usage behavior, merged from the survey |

**Sensitive survey fields (substance use, health conditions, sexual orientation) are
deliberately excluded**, even though they might be statistically predictive — they're
unrelated to the business question and using them would bake discrimination into the
model. Only benign demographics + purchase behavior are used.

### Why not a time-series model (LSTM, ARIMA, sequence model)?

This is worth being explicit about, since the data has timestamps:

1. **Different data shape.** Purchase histories are sparse, irregularly-spaced *event*
   streams (median 232 orders over 4-5 years with gaps from days to months) — closer to
   a point process / survival-analysis problem than a dense regularly-sampled signal.
2. **Different label shape.** The label isn't "predict the next value in a sequence,"
   it's "will an event occur in a future window" — a binary time-to-event question,
   the domain of survival analysis (Cox proportional hazards, discrete-time hazard
   models) or aggregate-feature classification, not sequence forecasting.
3. **Sample size.** Only 229 positive (churned) examples out of 4,930. A
   sequence model has far more parameters than 229 positives can reliably fit without
   overfitting — a tree/linear model on well-engineered aggregate features is the right
   complexity for this data volume.

The `momentum` feature (recent vs. prior 90-day order counts) is a deliberate, cheap
stand-in for temporal awareness without needing a sequence architecture. (The
textbook-correct upgrade path, if this were pushed further, is a discrete-time hazard
model / Cox regression that models time-to-churn directly instead of one binary
snapshot.)

### Models, and why

Two models are trained and compared, both `class_weight="balanced"` (the data is
imbalanced — 4.65% positive — so class weighting matters more than resampling schemes
like SMOTE for a first pass), with preprocessing (`StandardScaler` for numeric,
`OneHotEncoder` for categorical) fit **only on the train split**:

- **Logistic Regression** — simple, interpretable baseline, cheap to explain to a
  non-technical stakeholder
- **Random Forest** — handles nonlinearity and feature interactions without needing
  manual interaction terms, robust to the mix of scaled numeric + one-hot categorical
  features, a standard strong baseline for tabular churn problems

### Results (test set, n = 1,233, 25% stratified holdout)

| Model | PR-AUC | ROC-AUC | Macro-F1 |
|---|---|---|---|
| Logistic Regression | 0.485 | 0.928 | 0.621 |
| Random Forest | 0.358 | 0.923 | 0.692 |

**Why PR-AUC and macro-F1, not accuracy:** a model that predicted "nobody churns" would
already score ~95% accuracy while being completely useless. Random-guess PR-AUC on this
data is ≈ the churn rate itself (≈0.046), so both models show a real 8–10x lift over
baseline. ROC-AUC ≈0.92-0.93 indicates strong discrimination between churners and
retained customers.

**Why Logistic Regression has *higher* PR-AUC but *lower* macro-F1 than Random
Forest** (this looks contradictory but isn't):
- **PR-AUC** is a ranking metric integrated across *every* threshold — "if I sorted
  everyone by predicted risk, how good is that ordering." LR ranks slightly better
  overall.
- **Macro-F1** is evaluated at one fixed operating point (the default 0.5 threshold).
  At that cutoff LR has recall 0.82 but precision only 0.21 (many false positives);
  Random Forest is more conservative (recall 0.56, precision 0.34), giving it a
  better-balanced single operating point even though its full ranking curve is
  slightly worse. In production you'd pick the threshold based on the actual business
  cost of a false positive vs. a missed churner, not just accept the 0.5 default.

**Feature importances** (Random Forest) confirm the model leans on real behavioral
signal, not demographics: `orders_last_90d`, `recency_days`, `orders_prior_90d`, and
`avg_days_between_orders` dominate; one-hot demographic features barely register. This
is a useful sanity check that the model isn't secretly a demographics classifier.

### Deployment note

The train/test split exists to *validate* the approach. The model actually shipped
(`churn_model.joblib`, used by the API and dashboard) is a Random Forest **refit on
all 4,930 eligible users** — the split's job was done once it confirmed the approach
generalizes; the deployed model uses every available labeled example.

---

## Part 2 — RAG + Recommendation Agent (Agentic System #1)

**Files:** `build_product_index.py` → `recommend_tools.py` → `agent.py`

### Retrieval

Local `sentence-transformers` embeddings (`all-MiniLM-L6-v2`, 384-dim) over unique
products (title + category text), L2-normalized so cosine similarity reduces to a dot
product at query time — chosen over a paid embeddings API because re-embedding a large
catalog on every rebuild through an API is wasteful for something that only needs to
happen once, offline, on a laptop CPU.

Two scope cuts made the one-time build tractable (the first, naive attempt projected
to ~5.7 hours; these brought it down to ~30 minutes):

- **Only products with ≥2 historical purchases are indexed** (876K unique products →
  ~219K). Singleton one-off purchases carry the weakest similarity signal anyway —
  there's no "people who bought X also bought Y" pattern to validate against a single
  sale.
- **Text is capped at 64 tokens** (`model.max_seq_length = 64`). A handful of outlier
  titles ran past 1,000 words; transformer attention cost scales with sequence length,
  so those pathological titles were dominating wall-clock time out of proportion to
  their number. 64 tokens comfortably covers the 95th-percentile title (~32 words).

### Recommendation logic — content-based, not collaborative filtering

With ~219K products and each user only touching a tiny slice of the catalog, there's
too little purchase overlap between users for item-item collaborative filtering to have
real signal (the interaction matrix is too sparse). Instead, each user's "taste vector"
is the **recency-weighted centroid** (180-day exponential half-life) of the embeddings
of everything they've bought; recommendations are the nearest unseen products to that
centroid. Recency weighting matters because tastes drift over a multi-year purchase
history — a 2018 purchase shouldn't count as much as one from last month.

### Agent

`agent.py` wraps `recommend_tools.py`'s functions as Claude Agent SDK `@tool`s
(`search_products`, `similar_products`, `get_user_purchase_history`,
`recommend_for_user`, `category_stats`), bundled via `create_sdk_mcp_server`, with a
system prompt instructing the model to ground every product claim in tool results
rather than inventing ASINs or prices. Model: `claude-sonnet-5` (near-Opus quality on
tool-calling at a fraction of the cost — appropriate for a low-volume personal
project), with a `max_budget_usd` hard cap per session as a safety net against runaway
loops. The retrieval/recommendation logic is tested independent of the agent
(`test_recommend_tools.py`, no API key needed); running `agent.py` requires
`ANTHROPIC_API_KEY`.

---

## Part 3 — FastAPI Backend

**File:** `api.py` — REST endpoints for a frontend to consume:

| Endpoint | Purpose |
|---|---|
| `GET /users/{id}/churn-risk` | Precomputed churn probability for a scored user |
| `GET /users/{id}/history` | Recent purchase history |
| `GET /users/{id}/recommendations` | Personalized recommendations |
| `GET /search?q=...` | Free-text semantic product search |
| `GET /products/{asin}/similar` | Nearest-neighbor products |
| `POST /chat` | Conversational agent (needs `ANTHROPIC_API_KEY`) |

CORS is wide open (`allow_origins=["*"]`) to accommodate a dynamic frontend preview
origin — tighten to the deployed frontend's real origin before this goes anywhere
near production.

---

## Part 4 — Analytics Dashboard (Streamlit) (Agentic System #2)

**Files:** `build_dashboard_aggregates.py` → `dashboard_data.py` → `dashboard_qa.py` → `streamlit_app.py`

Three fixed views (Overview, Churn Analysis, Customer & Transactions — all Plotly
charts, computed locally, no API cost) plus an **Ask a Question** tab backed by Claude.

**Why this app never loads the raw 1.85M-row purchases file:** Streamlit Community
Cloud's free tier has ~1GB RAM, and pandas typically uses 3-5x a CSV's on-disk size
once loaded — the 300MB raw file would blow that budget. `build_dashboard_aggregates.py`
precomputes three small rollup tables (category / state / monthly, <100KB total) once,
locally; the deployed app only ever touches those plus the 1MB `churn_dataset.csv`.

**Ask a Question** is a plain Anthropic API tool-use loop (`dashboard_qa.py`), not the
Claude Agent SDK — the Agent SDK shells out to the `claude` CLI, built for interactive
coding sessions, not a stateless one-shot "answer this question with a validated chart
spec" call. Design:

- Claude gets one tool, `query_table` — a **safe, parameterized pandas aggregation**
  (groupby/metric/agg/filter over one of four known tables), never arbitrary code
  execution.
- The final answer must conform to a **JSON schema** (`chart_type`, `title`,
  `categories`, `values`, `insight`) via `output_config.format` (Claude's structured
  outputs feature), so the app never has to guess at the response shape or risk a
  malformed answer breaking the UI.
- Verified live: asking *"Which income bracket has the highest churn rate?"* correctly
  returned real, grounded numbers (8.2% for "Prefer not to say", down to 1.8% for
  "$100,000–$149,999") with a valid bar-chart spec — not a hallucinated answer.

### Deploying to Streamlit Community Cloud

1. Push this `Code files` folder to a GitHub repo.
2. At [share.streamlit.io](https://share.streamlit.io), sign in with GitHub → **New app**
   → pick the repo/branch → set **Main file path** to `streamlit_app.py`.
3. Under **Advanced settings**, set **Python dependencies file** to
   `requirements-dashboard.txt` (a lighter subset — skips `torch` / `sentence-transformers`,
   which this app doesn't use and which would slow the build).
4. Under **Advanced settings → Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
   (Set a usage/spend limit on this key in the Anthropic Console first.)
5. Deploy. You'll get a permanent `https://<app-name>.streamlit.app` URL.

---

## Repository structure

```
churn_features.py             # Part 1: builds the leakage-free churn feature table
churn_model.py                 # Part 1: trains + evaluates + persists the churn model
churn_dataset.csv               # Part 1: engineered features + labels (4,930 rows)
churn_model.joblib               # Part 1: deployed Random Forest pipeline

build_product_index.py         # Part 2: builds the product embedding index
recommend_tools.py              # Part 2: retrieval/recommendation engine
agent.py                        # Part 2: Claude Agent SDK wiring
test_recommend_tools.py         # Part 2: smoke test (no API key needed)

api.py                          # Part 3: FastAPI backend

build_dashboard_aggregates.py   # Part 4: precomputes small rollup tables
dashboard_data.py               # Part 4: loads dashboard data tables
dashboard_qa.py                 # Part 4: Claude tool-use loop for NL Q&A
streamlit_app.py                # Part 4: the dashboard app itself

requirements.txt                 # full project dependencies
requirements-dashboard.txt       # lightweight subset for Streamlit Cloud deploy
explore_data.py                  # initial data exploration script
```

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
