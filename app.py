# app.py – Walmart Analytics Dashboard
# ------------------------------------------------------------------
# Works with or without Plotly; no SciPy / statsmodels dependencies.
# ------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Optional Plotly import (nice visuals if present)
try:
    import plotly.express as px
    HAS_PLOTLY = True
except ModuleNotFoundError:
    HAS_PLOTLY = False
    import warnings
    warnings.warn("Plotly not installed – using Matplotlib fallback for scatter plot.")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, silhouette_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from mlxtend.frequent_patterns import apriori, association_rules

# ------------------------------------------------------------------
# CONFIG – adjust if your file / sheet name changes
# ------------------------------------------------------------------
FILE_NAME  = "Anirudh_data.xlsx"
SHEET_NAME = "Dataset (2)"

st.set_page_config(page_title="Walmart Sales Intelligence",
                   page_icon="🛒",
                   layout="wide")

# ------------------------------------------------------------------
# DATA LOADER
# ------------------------------------------------------------------
@st.cache_data
def load_excel(path: str, sheet: str) -> pd.DataFrame:
    """Load Excel sheet; stop app if it fails."""
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except Exception as e:
        st.error(f"❌ Could not load “{path}” / sheet “{sheet}”.\n\n{e}")
        return pd.DataFrame()

df = load_excel(FILE_NAME, SHEET_NAME)
if df.empty:
    st.stop()

numeric_cols     = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

# ------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------------
st.sidebar.title("🏷️ Navigation")
page = st.sidebar.radio(
    "Choose module",
    ["📊 Descriptive Analytics",
     "🤖 Classification",
     "🎯 Clustering",
     "🛒 Association Rules",
     "📈 Regression"]
)

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def perf_row(y_true, y_pred, model_name):
    return {
        "Model":     model_name,
        "Accuracy":  round(accuracy_score(y_true, y_pred), 3),
        "Precision": round(precision_score(y_true, y_pred, average="weighted"), 3),
        "Recall":    round(recall_score(y_true, y_pred, average="weighted"), 3),
        "F1":        round(f1_score(y_true, y_pred, average="weighted"), 3),
    }

def prettify_rules(rules_df: pd.DataFrame) -> pd.DataFrame:
    for c in ("antecedents", "consequents"):
        rules_df[c] = rules_df[c].apply(lambda x: ", ".join(sorted(list(x))))
    return rules_df

# ------------------------------------------------------------------
# 1️⃣  DESCRIPTIVE ANALYTICS
# ------------------------------------------------------------------
if page == "📊 Descriptive Analytics":
    st.header("📊 Descriptive Analytics")

    # --- Interactive filters
    with st.sidebar.expander("🔎 Filters", True):
        num_col = st.selectbox("Numeric filter column", numeric_cols, 0)
        nmin, nmax = float(df[num_col].min()), float(df[num_col].max())
        num_range = st.slider(f"{num_col} range", nmin, nmax, (nmin, nmax))
        cat_filters = {}
        for c in categorical_cols[:5]:          # first few categorical cols
            cat_filters[c] = st.multiselect(
                c, df[c].dropna().unique().tolist(),
                default=df[c].dropna().unique().tolist()
            )
        show_raw = st.checkbox("Show filtered data")

    # Apply filters
    mask = df[num_col].between(*num_range)
    for c, vals in cat_filters.items():
        mask &= df[c].isin(vals)
    dff = df[mask]

    st.success(f"Filtered rows: {len(dff):,}")
    if show_raw:
        st.dataframe(dff.head())

    # ---- Visuals
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"{num_col} Distribution")
        fig, ax = plt.subplots()
        ax.hist(dff[num_col].dropna(), bins=30)
        ax.set_xlabel(num_col)
        st.pyplot(fig)

    with col2:
        st.subheader("Correlation Heatmap")
        fig2, ax2 = plt.subplots()
        corr = dff[numeric_cols].corr()
        im = ax2.imshow(corr, aspect="auto")
        ax2.set_xticks(range(len(corr))); ax2.set_xticklabels(corr.columns, rotation=90)
        ax2.set_yticks(range(len(corr))); ax2.set_yticklabels(corr.columns)
        fig2.colorbar(im)
        st.pyplot(fig2)

    if len(numeric_cols) >= 2:
        xcol, ycol = numeric_cols[:2]
        st.subheader(f"{ycol} vs {xcol}")
        if HAS_PLOTLY:
            fig3 = px.scatter(dff, x=xcol, y=ycol, opacity=0.6, height=400)
            if dff[[xcol, ycol]].dropna().shape[0] > 1:
                coef = np.polyfit(dff[xcol], dff[ycol], 1)
                xs = np.linspace(dff[xcol].min(), dff[xcol].max(), 100)
                fig3.add_scatter(x=xs, y=coef[0]*xs + coef[1],
                                 mode="lines", name="Linear fit",
                                 line=dict(dash="dash"))
            st.plotly_chart(fig3, use_container_width=True)
        else:
            fig3, ax3 = plt.subplots()
            ax3.scatter(dff[xcol], dff[ycol], alpha=0.6)
            if dff[[xcol, ycol]].dropna().shape[0] > 1:
                coef = np.polyfit(dff[xcol], dff[ycol], 1)
                xs = np.linspace(dff[xcol].min(), dff[xcol].max(), 100)
                ax3.plot(xs, coef[0]*xs + coef[1], linestyle="--")
            ax3.set_xlabel(xcol); ax3.set_ylabel(ycol)
            st.pyplot(fig3)

# ------------------------------------------------------------------
# 2️⃣  CLASSIFICATION
# ------------------------------------------------------------------
elif page == "🤖 Classification":
    st.header("🤖 Classification")

    target_col = st.selectbox("Choose a categorical target column", categorical_cols)
    if target_col:
        y = df[target_col]
        X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42
        )

        # Scale numeric features for KNN only
        scaler = StandardScaler().fit(X_tr.select_dtypes(include=np.number))
        def scale(d):
            d2 = d.copy()
            d2.loc[:, scaler.feature_names_in_] = scaler.transform(d2[scaler.feature_names_in_])
            return d2
        X_tr_sc, X_te_sc = scale(X_tr), scale(X_te)

        models = {
            "KNN":             KNeighborsClassifier(n_neighbors=7),
            "Decision Tree":   DecisionTreeClassifier(max_depth=6, random_state=42),
            "Random Forest":   RandomForestClassifier(n_estimators=300, random_state=42),
            "Gradient Boost":  GradientBoostingClassifier(random_state=42),
        }

        results, probas = [], {}
        for name, mdl in models.items():
            mdl.fit(X_tr_sc if name=="KNN" else X_tr, y_tr)
            preds = mdl.predict(X_te_sc if name=="KNN" else X_te)
            results.append(perf_row(y_te, preds, name))
            if y.nunique()==2 and hasattr(mdl, "predict_proba"):
                probas[name] = mdl.predict_proba(X_te_sc if name=="KNN" else X_te)[:, 1]

        st.subheader("Performance")
        st.dataframe(pd.DataFrame(results).set_index("Model"))

        cm_model = st.selectbox("Confusion matrix model", list(models.keys()))
        cm_pred  = models[cm_model].predict(X_te_sc if cm_model=="KNN" else X_te)
        cm       = confusion_matrix(y_te, cm_pred)
        fig_cm, ax_cm = plt.subplots()
        ax_cm.imshow(cm, cmap="Blues")
        ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(j, i, cm[i, j], ha="center", va="center")
        st.pyplot(fig_cm)

        if y.nunique()==2:
            st.subheader("ROC Curves")
            fig_roc, ax_roc = plt.subplots()
            for name, p in probas.items():
                fpr, tpr, _ = roc_curve(y_te, p)
                ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.2f})")
            ax_roc.plot([0,1],[0,1],"--", color="grey")
            ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR"); ax_roc.legend()
            st.pyplot(fig_roc)

        # Batch prediction
        st.markdown("---")
        st.subheader("🔮 Batch Prediction")
        upl = st.file_uploader("Upload CSV (same structure, no target)", type="csv")
        if upl:
            new = pd.read_csv(upl)
            new_proc = pd.get_dummies(new, drop_first=True)
            new_proc = new_proc.reindex(columns=X.columns, fill_value=0)
            best = models["Random Forest"]
            new[f"{target_col}_Pred"] = best.predict(new_proc)
            st.write(new.head())
            st.download_button("Download predictions",
                               new.to_csv(index=False).encode("utf-8"),
                               "predictions.csv",
                               "text/csv")

# ------------------------------------------------------------------
# 3️⃣  CLUSTERING
# ------------------------------------------------------------------
elif page == "🎯 Clustering":
    st.header("🎯 K-Means Clustering")

    if len(numeric_cols) < 2:
        st.warning("Need at least two numeric columns for clustering.")
        st.stop()

    k = st.slider("Number of clusters (k)", 2, 10, 4)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(df[numeric_cols])
    df["Cluster"] = km.labels_

    # Diagnostics
    inertias, silh = [], []
    for i in range(2, 11):
        km_i = KMeans(n_clusters=i, n_init=10, random_state=42).fit(df[numeric_cols])
        inertias.append(km_i.inertia_)
        silh.append(silhouette_score(df[numeric_cols], km_i.labels_))

    cA, cB = st.columns(2)
    with cA:
        fig_el, ax_el = plt.subplots()
        ax_el.plot(range(2,11), inertias, marker="o")
        ax_el.set_title("Elbow Curve"); ax_el.set_xlabel("k"); ax_el.set_ylabel("Inertia")
        st.pyplot(fig_el)
    with cB:
        fig_si, ax_si = plt.subplots()
        ax_si.plot(range(2,11), silh, marker="s")
        ax_si.set_title("Silhouette Score"); ax_si.set_xlabel("k"); ax_si.set_ylabel("Score")
        st.pyplot(fig_si)

    st.subheader("Cluster Centroids (numeric means)")
    cent = pd.DataFrame(km.cluster_centers_, columns=numeric_cols).round(2)
    st.dataframe(cent)

    st.download_button("Download data with cluster labels",
                       df.to_csv(index=False).encode("utf-8"),
                       "clustered_data.csv",
                       "text/csv")

# ------------------------------------------------------------------
# 4️⃣  ASSOCIATION RULES
# ------------------------------------------------------------------
elif page == "🛒 Association Rules":
    st.header("🛒 Association Rule Mining")

    # Binary columns (0/1 or True/False)
    bin_cols = [c for c in df.columns if df[c].dropna().isin([0,1,True,False]).all()]
    selectable = bin_cols + categorical_cols

    use_cols = st.multiselect("Columns to include", selectable,
                              default=bin_cols[:20] if bin_cols else selectable[:20])

    min_sup  = st.slider("Min support",    0.01, 0.5, 0.05, 0.01)
    min_conf = st.slider("Min confidence", 0.10, 0.9, 0.60, 0.05)
    min_lift = st.slider("Min lift",       1.00, 5.0, 1.20, 0.10)

    if st.button("Run Apriori"):
        if not use_cols:
            st.warning("Select at least one column.")
            st.stop()
        basket = pd.get_dummies(df[use_cols].astype(str), prefix=use_cols)
        frequent = apriori(basket.astype(bool), min_support=min_sup, use_colnames=True)
        if frequent.empty:
            st.warning("No frequent itemsets at this support.")
        else:
            rules = association_rules(frequent,
                                      metric="confidence",
                                      min_threshold=min_conf)
            rules = rules[rules["lift"] >= min_lift]
            if rules.empty:
                st.warning("No rules meet confidence/lift thresholds.")
            else:
                rules = prettify_rules(rules).sort_values("lift", ascending=False).head(10)
                st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]]
                             .style.format({"support":"{:.3f}",
                                            "confidence":"{:.2f}",
                                            "lift":"{:.2f}"}))

# ------------------------------------------------------------------
# 5️⃣  REGRESSION
# ------------------------------------------------------------------
else:  # page == "📈 Regression"
    st.header("📈 Regression")

    target_num = st.selectbox("Numeric target column", numeric_cols)
    if target_num:
        y = df[target_num]
        X = pd.get_dummies(df.drop(columns=[target_num]), drop_first=True)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "Linear": LinearRegression(),
            "Ridge":  Ridge(alpha=1.0),
            "Lasso":  Lasso(alpha=0.001),
            "DTReg":  DecisionTreeRegressor(max_depth=6, random_state=42)
        }

        rows = []
        for name, reg in models.items():
            reg.fit(X_tr, y_tr)
            preds = reg.predict(X_te)
            rows.append({
                "Model": name,
                "R²":   round(reg.score(X_te, y_te), 3),
                "RMSE": int(np.sqrt(((y_te - preds) ** 2).mean())),
                "MAE":  int(np.abs(y_te - preds).mean())
            })

        st.dataframe(pd.DataFrame(rows).set_index("Model"))

        st.markdown("""
        **Interpretation notes**  
        • Ridge mitigates multicollinearity vs. OLS.  
        • Lasso shrinks irrelevant features to zero (driver screening).  
        • Decision-Tree captures non-linear effects—monitor depth to avoid over-fit.  
        """)

