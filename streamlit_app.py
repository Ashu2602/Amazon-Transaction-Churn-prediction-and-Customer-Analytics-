"""
Amazon Purchases -- Churn & Customer Analytics Dashboard.

Three fixed dashboard views (fast, no API cost) plus an open-ended "Ask a
Question" tab backed by Claude (dashboard_qa.py). Deliberately only loads the
small precomputed aggregates + the churn dataset -- see dashboard_data.py and
build_dashboard_aggregates.py for why the raw 1.85M-row purchases file and the
product embedding index are excluded from this app.
"""
import os

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from dashboard_data import load_tables
from dashboard_qa import answer_question

st.set_page_config(page_title="Amazon Purchases Analytics", layout="wide")

load_dotenv()  # local dev: reads .env if present; no-op on Streamlit Cloud (no .env is deployed there)

# Qualitative palette used across every chart -- one place to change the look
COLORS = px.colors.qualitative.Bold
HEAT = "Viridis"


def get_api_key() -> str | None:
    """Resolve ANTHROPIC_API_KEY from Streamlit secrets (hosted) first, then
    the environment (local .env). Split out from a bare `st.secrets[...]`
    lookup so a genuinely-missing secret produces a clear on-page message
    instead of the SDK's generic "could not resolve authentication" error."""
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass  # no secrets.toml at all (e.g. local run with only a .env) -- fall through
    return os.environ.get("ANTHROPIC_API_KEY")


@st.cache_data
def get_tables():
    return load_tables()


@st.cache_resource
def get_churn_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return joblib.load(os.path.join(base_dir, "churn_model.joblib"))


tables = get_tables()
category_df = tables["category"]
state_df = tables["state"]
monthly_df = tables["monthly"]
churn_df = tables["churn_users"]

st.title("Amazon Purchases -- Churn & Customer Analytics")

page = st.sidebar.radio("View", ["Overview", "Churn Analysis", "Customer & Transactions", "Ask a Question"])

# ---------------------------------------------------------------- Overview
if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers scored", f"{len(churn_df):,}")
    c2.metric("Churn rate", f"{churn_df['churn'].mean() * 100:.1f}%")
    c3.metric("Product categories", f"{len(category_df):,}")
    c4.metric("Avg order value", f"${churn_df['avg_order_value'].mean():.2f}")

    col1, col2 = st.columns(2)
    with col1:
        top_cat = category_df.head(10)
        fig = px.bar(
            top_cat, x="total_spend", y="Category", orientation="h", title="Top 10 categories by total spend",
            color="Category", color_discrete_sequence=COLORS,
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.line(monthly_df, x="year_month", y="n_orders", title="Monthly order volume", markers=True)
        fig.update_traces(line_color=COLORS[0])
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------ Churn Analysis
elif page == "Churn Analysis":
    st.subheader("Churn rate by customer segment")

    demo_col = st.selectbox(
        "Segment by",
        ["Q-demos-income", "Q-demos-age", "Q-demos-education", "Q-amazon-use-how-oft", "Q-amazon-use-hh-size"],
        format_func=lambda c: c.replace("Q-demos-", "").replace("Q-amazon-use-", "").replace("-", " ").title(),
    )
    seg = churn_df.groupby(demo_col)["churn"].mean().sort_values(ascending=False).reset_index()
    seg["churn"] = seg["churn"] * 100
    fig = px.bar(
        seg, x=demo_col, y="churn", title=f"Churn rate (%) by {demo_col}",
        color="churn", color_continuous_scale="Reds",
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        model = get_churn_model()
        feature_names = model.named_steps["prep"].get_feature_names_out()
        importances = model.named_steps["clf"].feature_importances_
        top_feat = (
            pd.Series(importances, index=feature_names)
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        top_feat.columns = ["feature", "importance"]
        fig = px.bar(
            top_feat, x="importance", y="feature", orientation="h", title="Top churn model features",
            color="importance", color_continuous_scale=HEAT,
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        proba = model.predict_proba(churn_df.drop(columns=["churn", "orders_in_label_window"]))[:, 1]
        fig = px.histogram(x=proba, nbins=40, title="Predicted churn probability distribution", color_discrete_sequence=[COLORS[3]])
        fig.update_layout(xaxis_title="predicted churn probability", yaxis_title="customers")
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------- Customer & Transactions
elif page == "Customer & Transactions":
    col1, col2 = st.columns(2)
    with col1:
        fig = px.choropleth(
            state_df, locations="Shipping Address State", locationmode="USA-states", scope="usa",
            color="total_spend", title="Total spend by state", color_continuous_scale="Plasma",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(
            state_df.head(10), x="total_spend", y="Shipping Address State", orientation="h", title="Top 10 states by spend",
            color="Shipping Address State", color_discrete_sequence=COLORS,
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    fig = px.line(monthly_df, x="year_month", y="total_spend", title="Monthly total spend", markers=True)
    fig.update_traces(line_color=COLORS[2])
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------- Ask a Question
else:
    st.subheader("Ask a question about churn or customer behavior")
    st.caption("Examples: \"Which income bracket churns the most?\" / \"What are the top 5 categories by spend?\" / \"How many customers have churned?\"")

    api_key = get_api_key()
    if not api_key:
        st.error(
            "ANTHROPIC_API_KEY is not configured for this app. "
            "Go to the app's ⋮ menu → Settings → Secrets, add `ANTHROPIC_API_KEY = \"sk-ant-...\"`, "
            "save, then Reboot the app."
        )

    question = st.text_input("Your question")
    if st.button("Ask", disabled=not api_key) and question:
        with st.spinner("Thinking..."):
            try:
                spec = answer_question(question, tables, api_key=api_key)
            except Exception as e:
                st.error(f"Couldn't answer that: {e}")
                spec = None

        if spec:
            st.markdown(f"**{spec['insight']}**")
            if spec["chart_type"] == "number" and spec["values"]:
                st.metric(spec["title"], spec["values"][0])
            elif spec["chart_type"] == "table" and spec["categories"]:
                st.dataframe(pd.DataFrame({spec["x_label"] or "category": spec["categories"], spec["y_label"] or "value": spec["values"]}))
            elif spec["categories"] and spec["values"]:
                chart_df = pd.DataFrame({spec["x_label"] or "x": spec["categories"], spec["y_label"] or "y": spec["values"]})
                x_col, y_col = spec["x_label"] or "x", spec["y_label"] or "y"
                if spec["chart_type"] == "pie":
                    fig = px.pie(chart_df, names=x_col, values=y_col, title=spec["title"], color_discrete_sequence=COLORS)
                elif spec["chart_type"] == "line":
                    fig = px.line(chart_df, x=x_col, y=y_col, title=spec["title"], markers=True)
                    fig.update_traces(line_color=COLORS[0])
                else:
                    fig = px.bar(chart_df, x=x_col, y=y_col, title=spec["title"], color=x_col, color_discrete_sequence=COLORS)
                    fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
