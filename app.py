# app.py  – Streamlit multi-tab analytics dashboard
# -------------------------------------------------
# Robust imports: first try the utils/ package layout,
# fall back to a single utils.py if the package isn’t present.
try:
    from utils.data_loader import load_data
    from utils.classifiers  import run_classifiers
    from utils.clusterer    import run_kmeans
    from utils.regression   import run_regressions
    from utils.association  import run_apriori
except ModuleNotFoundError:
    # fallback: everything lives in utils.py
    from utils import (
        load_data,
        run_classifiers,
        run_kmeans,
        run_regressions,
        run_apriori,
    )

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.metrics import roc_curve

# ─────────────────────────────────────────────────────────────
# Streamlit config
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Analytics Dashboard", layout="wide")

# ─────────────────────────────────────────────────────────────
# Sidebar – data upload
# ─────────────────────────────────────────────────────────────
st.sidebar.title("Upload Data")
data_file = st.sidebar.file_uploader("Excel file", type=["xlsx"])

if data_file:
    df = pd.read_excel(data_file)
else:
    # fallback to repo-bundled placeholder
    df = load_data("data/Anirudh_data.xlsx")

st.sidebar.info("Use the tabs at the top to explore.")

# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────
tabs = st.tabs([
    "Data Visualization",
    "Classification",
    "Clustering",
    "Association Rules",
    "Regression"
])

# 1️⃣ Data Visualization
with tabs[0]:
    st.header("Data Visualization & Insights")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe(include="all"))

    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    # Numeric distributions
    if len(num_cols) > 0:
        st.subheader("Numeric Distributions")
        for col in num_cols[:3]:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

        if len(num_cols) >= 2:
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    # Top categorical counts
    if len(cat_cols) > 0:
        st.subheader("Top Category Counts")
        for col in cat_cols[:2]:
            st.write(df[col].value_counts().head(10).to_frame())

# 2️⃣ Classification
with tabs[1]:
    st.header("Classification")

    target = st.selectbox("Target column", df.columns)
    features = st.multiselect(
        "Feature columns",
        df.columns.drop(target),
        default=list(df.columns.drop(target)[:5])
    )

    if st.button("Run Classification") and features:
        X = df[features]
        y = df[target]

        clf_results = run_classifiers(X, y)

        # Metrics table
        st.subheader("Performance Metrics")
        perf_df = pd.DataFrame({
            name: {
                "Train Acc": r["train_acc"],
                "Test Acc":  r["test_acc"],
                "Precision": r["precision"],
                "Recall":    r["recall"],
                "F1":        r["f1"]
            } for name, r in clf_results.items()
        }).T
        st.dataframe(perf_df.style.format("{:.2f}"))

        # ROC curves
        st.subheader("ROC Curves")
        fig, ax = plt.subplots()
        for name, res in clf_results.items():
            model = res["model"]
            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(X)[:, 1]
            else:
                scores = model.decision_function(X)
            fpr, tpr, _ = roc_curve(y, scores)
            ax.plot(fpr, tpr, label=name)
        ax.plot([0, 1], [0, 1], "--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        sel_model = st.selectbox("Choose model", list(clf_results.keys()))
        cm = clf_results[sel_model]["confusion_matrix"]
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Predict on new data
        st.subheader("Predict on New Data")
        pred_file = st.file_uploader("Upload data (features only)", type=["xlsx"])
        if pred_file:
            new_df = pd.read_excel(pred_file)
            preds = clf_results[sel_model]["model"].predict(new_df)
            new_df["Predicted"] = preds
            st.dataframe(new_df.head())

            buf = io.BytesIO()
            new_df.to_excel(buf, index=False)
            st.download_button(
                "Download Predictions",
                buf.getvalue(),
                file_name="predictions.xlsx"
            )

# 3️⃣ Clustering
with tabs[2]:
    st.header("Clustering (K-Means)")

    num_cols = df.select_dtypes(include=[np.number]).columns
    cluster_feats = st.multiselect(
        "Select features for clustering",
        num_cols,
        default=list(num_cols[:4])
    )
    n_clusters = st.slider("Number of clusters", 2, 10, 3)

    if st.button("Run K-Means") and cluster_feats:
        labels, inertia, _ = run_kmeans(df[cluster_feats], n_clusters)
        df["Cluster"] = labels
        st.write(f"Inertia (within-cluster SSE): **{inertia:.2f}**")

        fig, ax = plt.subplots()
        sns.countplot(x="Cluster", data=df, ax=ax)
        st.pyplot(fig)

        st.subheader("Cluster Profiles (means)")
        st.dataframe(df.groupby("Cluster")[cluster_feats].mean())

        buf = io.BytesIO()
        df.to_excel(buf, index=False)
        st.download_button(
            "Download Clustered Data",
            buf.getvalue(),
            file_name="clustered_data.xlsx"
        )

# 4️⃣ Association Rules
with tabs[3]:
    st.header("Association Rule Mining")

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) >= 2:
        c1 = st.selectbox("Column 1", cat_cols)
        c2 = st.selectbox("Column 2", cat_cols, index=1)
        min_sup  = st.slider("Min support",     0.01, 1.0, 0.1)
        min_conf = st.slider("Min confidence",  0.01, 1.0, 0.3)

        if st.button("Run Apriori"):
            rules = run_apriori(df, [c1, c2], min_sup, min_conf)
            st.dataframe(rules.sort_values("lift", ascending=False).head(10))
    else:
        st.info("Need at least two categorical columns.")

# 5️⃣ Regression
with tabs[4]:
    st.header("Regression")

    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) >= 2:
        target_r = st.selectbox("Target variable", num_cols)
        feat_r = st.multiselect(
            "Feature variables",
            num_cols.drop(target_r),
            default=list(num_cols.drop(target_r)[:3])
        )

        if st.button("Run Regression") and feat_r:
            X_r = df[feat_r]
            y_r = df[target_r]
            reg_results = run_regressions(X_r, y_r)

            st.subheader("Metrics")
            metrics_df = pd.DataFrame({
                n: {
                    "Train R²": r["train_r2"],
                    "Test R²":  r["test_r2"],
                    "RMSE":     r["rmse"]
                } for n, r in reg_results.items()
            }).T
            st.dataframe(metrics_df.style.format("{:.2f}"))

            for name, res in reg_results.items():
                fig, ax = plt.subplots()
                ax.scatter(res["y_test"], res["y_pred"])
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title(f"{name} Regression")
                st.pyplot(fig)
    else:
        st.info("Need numeric columns for regression.")
