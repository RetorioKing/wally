# app.py – Streamlit dashboard for Walmart sales data (all-in-one, robust)
# -----------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io, os

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve,
    r2_score, mean_squared_error
)

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# ──────────────────────────────
# Helper functions (were utils/*)
# ──────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    """Read an Excel file and return a DataFrame."""
    return pd.read_excel(path)

def run_classifiers(X, y):
    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "GBRT": GradientBoostingClassifier(random_state=42)
    }
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    results = {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        ypred = model.predict(Xte)
        results[name] = {
            "model": model,
            "train_acc": accuracy_score(ytr, model.predict(Xtr)),
            "test_acc":  accuracy_score(yte, ypred),
            "precision": precision_score(yte, ypred, average="weighted", zero_division=0),
            "recall":    recall_score   (yte, ypred, average="weighted", zero_division=0),
            "f1":        f1_score       (yte, ypred, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(yte, ypred),
            "y_test": yte,
            "y_pred": ypred
        }
    return results

def run_kmeans(X, k):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(Xs)
    return labels, km.inertia_, km

def run_regressions(X, y):
    models = {
        "Linear": LinearRegression(),
        "Ridge":  Ridge(alpha=1.0),
        "Lasso":  Lasso(alpha=0.01),
        "Decision Tree": DecisionTreeRegressor(random_state=42)
    }
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=42)
    results = {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        ypred = model.predict(Xte)
        results[name] = {
            "train_r2": r2_score(ytr, model.predict(Xtr)),
            "test_r2":  r2_score(yte, ypred),
            "rmse":     mean_squared_error(yte, ypred, squared=False),
            "model":    model,
            "y_test":   yte,
            "y_pred":   ypred
        }
    return results

def run_apriori(df, cols, supp=0.1, conf=0.3):
    df_filt = df[cols].astype(str)
    te = TransactionEncoder()
    onehot = pd.DataFrame(
        te.fit(df_filt.values.tolist()).transform(df_filt.values.tolist()),
        columns=te.columns_
    )
    itemsets = apriori(onehot, min_support=supp, use_colnames=True)
    return association_rules(itemsets, metric="confidence", min_threshold=conf)

# ──────────────────────────────
# Streamlit configuration
# ──────────────────────────────
st.set_page_config(page_title="Walmart Sales Analytics", layout="wide")
st.title("📊 Walmart Sales Analytics Dashboard")

# ──────────────────────────────
# Load data: upload or fallback
# ──────────────────────────────
st.sidebar.header("Load data")
user_file = st.sidebar.file_uploader("Excel file (.xlsx)", type=["xlsx"])

if user_file:
    df = pd.read_excel(user_file)
    st.sidebar.success("✅ Uploaded file loaded")
else:
    try:
        df = load_data("Walmart_sales.xlsx")   # file beside app.py
        st.sidebar.success("✅ Loaded default Walmart_sales.xlsx")
    except FileNotFoundError:
        df = None
        st.sidebar.warning(
            "⚠️ No default file found.\nPlease upload an Excel file on the sidebar."
        )

# If still no DataFrame, stop the app
if df is None:
    st.stop()

# ──────────────────────────────
# Tabs
# ──────────────────────────────
tabs = st.tabs(["Visualization", "Classification",
                "Clustering", "Association Rules", "Regression"])

# 1️⃣ Visualization
with tabs[0]:
    st.subheader("Dataset preview")
    st.dataframe(df.head())

    st.subheader("Descriptive statistics")
    st.dataframe(df.describe(include="all"))

    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    if len(num_cols):
        st.subheader("Numeric distributions (first 3)")
        for col in num_cols[:3]:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f"{col} distribution")
            st.pyplot(fig)

        if len(num_cols) >= 2:
            st.subheader("Correlation heatmap")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    if len(cat_cols):
        st.subheader("Top-category value counts (first 2)")
        for col in cat_cols[:2]:
            st.write(df[col].value_counts().head(10).to_frame())

# 2️⃣ Classification
with tabs[1]:
    st.header("Classification")
    target = st.selectbox("Target (categorical/binary)", df.columns)
    features = st.multiselect(
        "Feature columns", df.columns.drop(target),
        default=list(df.columns.drop(target)[:5])
    )

    if st.button("Run classification") and features:
        X, y = df[features], df[target]
        clf_res = run_classifiers(X, y)

        st.subheader("Performance metrics")
        perf = pd.DataFrame({
            n: {"Train Acc": r["train_acc"], "Test Acc": r["test_acc"],
                "Precision": r["precision"], "Recall": r["recall"], "F1": r["f1"]}
            for n, r in clf_res.items()
        }).T
        st.dataframe(perf.style.format("{:.2f}"))

        st.subheader("Confusion matrix & ROC")
        model_sel = st.selectbox("Model", list(clf_res.keys()))
        cm = clf_res[model_sel]["confusion_matrix"]
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)

        fig, ax = plt.subplots()
        for n, r in clf_res.items():
            mdl = r["model"]
            scores = (
                mdl.predict_proba(X)[:, 1]
                if hasattr(mdl, "predict_proba")
                else mdl.decision_function(X)
            )
            fpr, tpr, _ = roc_curve(y, scores)
            ax.plot(fpr, tpr, label=n)
        ax.plot([0, 1], [0, 1], "--")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
        st.pyplot(fig)

        st.subheader("Predict on new data")
        pred_file = st.file_uploader(
            "Upload feature-only Excel", type=["xlsx"], key="pred"
        )
        if pred_file:
            new_df = pd.read_excel(pred_file)
            new_df["Predicted"] = clf_res[model_sel]["model"].predict(new_df)
            st.dataframe(new_df.head())
            buf = io.BytesIO(); new_df.to_excel(buf, index=False)
            st.download_button(
                "Download predictions", buf.getvalue(), file_name="walmart_predictions.xlsx"
            )

# 3️⃣ Clustering
with tabs[2]:
    st.header("K-Means clustering")
    sel_feats = st.multiselect(
        "Numeric features", num_cols,
        default=list(num_cols[:4])
    )
    k = st.slider("Clusters (k)", 2, 10, 3)
    if st.button("Run K-Means") and sel_feats:
        labels, inertia, _ = run_kmeans(df[sel_feats], k)
        df["Cluster"] = labels
        st.write(f"Inertia (SSE): **{inertia:.2f}**")
        fig, ax = plt.subplots()
        sns.countplot(x="Cluster", data=df, ax=ax)
        st.pyplot(fig)

        st.subheader("Cluster means")
        st.dataframe(df.groupby("Cluster")[sel_feats].mean())

        buf = io.BytesIO(); df.to_excel(buf, index=False)
        st.download_button(
            "Download cluster-labeled data", buf.getvalue(),
            file_name="walmart_clustered.xlsx"
        )

# 4️⃣ Association Rules
with tabs[3]:
    st.header("Association rule mining")
    if len(cat_cols) >= 2:
        c1 = st.selectbox("Column 1", cat_cols)
        c2 = st.selectbox("Column 2", cat_cols, index=1)
        supp = st.slider("Min support", 0.01, 1.0, 0.1)
        conf = st.slider("Min confidence", 0.01, 1.0, 0.3)
        if st.button("Run Apriori"):
            rules = run_apriori(df, [c1, c2], supp, conf)
            st.dataframe(rules.sort_values("lift", ascending=False).head(10))
    else:
        st.info("Need at least two categorical columns.")

# 5️⃣ Regression
with tabs[4]:
    st.header("Regression")
    if len(num_cols) >= 2:
        y_var = st.selectbox("Target variable (numeric)", num_cols)
        X_vars = st.multiselect(
            "Feature variables", num_cols.drop(y_var),
            default=list(num_cols.drop(y_var)[:3])
        )
        if st.button("Run regression") and X_vars:
            reg_res = run_regressions(df[X_vars], df[y_var])

            st.subheader("Model metrics")
            mtab = pd.DataFrame({
                n: {"Train R²": r["train_r2"], "Test R²": r["test_r2"], "RMSE": r["rmse"]}
                for n, r in reg_res.items()
            }).T
            st.dataframe(mtab.style.format("{:.2f}"))

            for n, r in reg_res.items():
                fig, ax = plt.subplots()
                ax.scatter(r["y_test"], r["y_pred"])
                ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
                ax.set_title(n)
                st.pyplot(fig)
    else:
        st.info("Need at least two numeric columns for regression.")
