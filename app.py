# app.py – one-file Streamlit analytics dashboard
# No external utils import required
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# ML imports
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

# Association-rule imports
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# ───────────────────────────────────────────────
# Helper functions (were in utils/* before)
# ───────────────────────────────────────────────
def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_excel(file_path)

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
            "recall":    recall_score(yte, ypred,    average="weighted", zero_division=0),
            "f1":        f1_score(yte, ypred,        average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(yte, ypred),
            "y_test": yte,
            "y_pred": ypred
        }
    return results

def run_kmeans(X, n_clusters):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)
    return labels, kmeans.inertia_, kmeans

def run_regressions(X, y):
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
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

def run_apriori(df, cols, min_support=0.1, min_conf=0.3):
    df_filt = df[cols].astype(str)
    transactions = df_filt.values.tolist()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    onehot = pd.DataFrame(te_ary, columns=te.columns_)
    itemsets = apriori(onehot, min_support=min_support, use_colnames=True)
    rules = association_rules(itemsets, metric="confidence", min_threshold=min_conf)
    return rules

# ───────────────────────────────────────────────
# Streamlit configuration & sidebar
# ───────────────────────────────────────────────
st.set_page_config(page_title="Analytics Dashboard", layout="wide")
st.sidebar.title("Upload Data")
data_file = st.sidebar.file_uploader("Excel file", type=["xlsx"])
if data_file:
    df = pd.read_excel(data_file)
else:
    # default bundled placeholder
    df = load_data("data/Anirudh_data.xlsx")

st.sidebar.info("Use the top tabs to explore the analysis.")

# ───────────────────────────────────────────────
# Tabs
# ───────────────────────────────────────────────
tabs = st.tabs(["Data Visualization", "Classification", "Clustering",
                "Association Rules", "Regression"])

# 1️⃣ Data Visualization
with tabs[0]:
    st.header("Data Visualization & Insights")
    st.subheader("Preview")
    st.dataframe(df.head())

    st.subheader("Descriptive Stats")
    st.dataframe(df.describe(include="all"))

    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    if len(num_cols):
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

    if len(cat_cols):
        st.subheader("Top Category Counts")
        for col in cat_cols[:2]:
            st.write(df[col].value_counts().head(10).to_frame())

# 2️⃣ Classification
with tabs[1]:
    st.header("Classification")
    target = st.selectbox("Target column", df.columns)
    features = st.multiselect("Feature columns", df.columns.drop(target),
                              default=list(df.columns.drop(target)[:5]))

    if st.button("Run Classification") and features:
        X, y = df[features], df[target]
        clf_res = run_classifiers(X, y)

        st.subheader("Performance")
        perf = pd.DataFrame({
            n: {"Train Acc": r["train_acc"], "Test Acc": r["test_acc"],
                "Precision": r["precision"], "Recall": r["recall"], "F1": r["f1"]}
            for n, r in clf_res.items()
        }).T
        st.dataframe(perf.style.format("{:.2f}"))

        st.subheader("ROC Curves")
        fig, ax = plt.subplots()
        for n, r in clf_res.items():
            mdl = r["model"]
            scores = mdl.predict_proba(X)[:, 1] if hasattr(mdl, "predict_proba") else mdl.decision_function(X)
            fpr, tpr, _ = roc_curve(y, scores)
            ax.plot(fpr, tpr, label=n)
        ax.plot([0, 1], [0, 1], "--")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Confusion Matrix")
        sel = st.selectbox("Choose model", list(clf_res.keys()))
        cm = clf_res[sel]["confusion_matrix"]
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        st.subheader("Predict on New Data")
        pred_file = st.file_uploader("Upload features-only Excel", type=["xlsx"])
        if pred_file:
            newdf = pd.read_excel(pred_file)
            preds = clf_res[sel]["model"].predict(newdf)
            newdf["Predicted"] = preds
            st.dataframe(newdf.head())
            buf = io.BytesIO()
            newdf.to_excel(buf, index=False)
            st.download_button("Download predictions", buf.getvalue(),
                               file_name="predictions.xlsx")

# 3️⃣ Clustering
with tabs[2]:
    st.header("Clustering (K-Means)")
    clust_feats = st.multiselect("Select features", num_cols,
                                 default=list(num_cols[:4]))
    n_clusters = st.slider("Clusters", 2, 10, 3)
    if st.button("Run K-Means") and clust_feats:
        labels, inert, _ = run_kmeans(df[clust_feats], n_clusters)
        df["Cluster"] = labels
        st.write(f"Inertia: **{inert:.2f}**")
        fig, ax = plt.subplots()
        sns.countplot(x="Cluster", data=df, ax=ax)
        st.pyplot(fig)
        st.subheader("Cluster Means")
        st.dataframe(df.groupby("Cluster")[clust_feats].mean())
        buf = io.BytesIO()
        df.to_excel(buf, index=False)
        st.download_button("Download cluster-labeled data", buf.getvalue(),
                           file_name="clustered_data.xlsx")

# 4️⃣ Association Rules
with tabs[3]:
    st.header("Association Rule Mining")
    if len(cat_cols) >= 2:
        c1 = st.selectbox("Column 1", cat_cols)
        c2 = st.selectbox("Column 2", cat_cols, index=1)
        sup  = st.slider("Min support", 0.01, 1.0, 0.1)
        conf = st.slider("Min confidence", 0.01, 1.0, 0.3)
        if st.button("Run Apriori"):
            rules = run_apriori(df, [c1, c2], sup, conf)
            st.dataframe(rules.sort_values("lift", ascending=False).head(10))
    else:
        st.info("Need at least two categorical columns.")

# 5️⃣ Regression
with tabs[4]:
    st.header("Regression")
    if len(num_cols) >= 2:
        target_r = st.selectbox("Target variable", num_cols)
        feat_r = st.multiselect(
            "Feature variables", num_cols.drop(target_r),
            default=list(num_cols.drop(target_r)[:3]))
        if st.button("Run Regression") and feat_r:
            regs = run_regressions(df[feat_r], df[target_r])
            st.subheader("Metrics")
            m_df = pd.DataFrame({
                n: {"Train R²": r["train_r2"], "Test R²": r["test_r2"], "RMSE": r["rmse"]}
                for n, r in regs.items()
            }).T
            st.dataframe(m_df.style.format("{:.2f}"))
            for n, r in regs.items():
                fig, ax = plt.subplots()
                ax.scatter(r["y_test"], r["y_pred"])
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title(n)
                st.pyplot(fig)
    else:
        st.info("Need numeric columns for regression.")
