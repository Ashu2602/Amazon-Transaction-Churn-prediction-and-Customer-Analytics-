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

from dashboard_data import load_tables
from dashboard_qa import answer_question

st.set_page_config(page_title="Amazon Purchases Analytics", layout="wide")

# Resolve ANTHROPIC_API_KEY from Streamlit secrets (hosted) or the environment
# (local .env via python-dotenv) -- st.secrets raises if no secrets.toml exists
# at all, so probe safely rather than assuming one is present.
try:
    _secret_key = st.secrets.get("ANTHROPIC_API_KEY")
except Exception:
    _secret_key = None
if _secret_key:
    os.environ["ANTHROPIC_API_KEY"] = _secret_key
else:
    from dotenv import load_dotenv
    load_dotenv()


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
        fig = px.bar(top_cat, x="total_spend", y="Category", orientation="h", title="Top 10 categories by total spend")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.line(monthly_df, x="year_month", y="n_orders", title="Monthly order volume")
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
    fig = px.bar(seg, x=demo_col, y="churn", title=f"Churn rate (%) by {demo_col}")
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
        fig = px.bar(top_feat, x="importance", y="feature", orientation="h", title="Top churn model features")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        proba = model.predict_proba(churn_df.drop(columns=["churn", "orders_in_label_window"]))[:, 1]
        fig = px.histogram(x=proba, nbins=40, title="Predicted churn probability distribution")
        fig.update_layout(xaxis_title="predicted churn probability", yaxis_title="customers")
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------- Customer & Transactions
elif page == "Customer & Transactions":
    col1, col2 = st.columns(2)
    with col1:
        fig = px.choropleth(
            state_df, locations="Shipping Address State", locationmode="USA-states", scope="usa",
            color="total_spend", title="Total spend by state",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(state_df.head(10), x="total_spend", y="Shipping Address State", orientation="h", title="Top 10 states by spend")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    fig = px.line(monthly_df, x="year_month", y="total_spend", title="Monthly total spend")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------- Ask a Question
else:
    st.subheader("Ask a question about churn or customer behavior")
    st.caption("Examples: \"Which income bracket churns the most?\" / \"What are the top 5 categories by spend?\" / \"How many customers have churned?\"")

    question = st.text_input("Your question")
    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            try:
                spec = answer_question(question, tables)
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
                if spec["chart_type"] == "pie":
                    fig = px.pie(chart_df, names=spec["x_label"] or "x", values=spec["y_label"] or "y", title=spec["title"])
                elif spec["chart_type"] == "line":
                    fig = px.line(chart_df, x=spec["x_label"] or "x", y=spec["y_label"] or "y", title=spec["title"])
                else:
                    fig = px.bar(chart_df, x=spec["x_label"] or "x", y=spec["y_label"] or "y", title=spec["title"])
                st.plotly_chart(fig, use_container_width=True)
