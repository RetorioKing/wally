import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from utils.data_loader import load_data
from utils.classifiers import run_classifiers
from utils.clusterer import run_kmeans
from utils.regression import run_regressions
from utils.association import run_apriori
from sklearn.metrics import roc_curve

st.set_page_config(layout='wide', page_title='Multi-Tab Analytics Dashboard')

# Sidebar
st.sidebar.title('Upload Data')
data_file = st.sidebar.file_uploader('Upload Excel File', type=['xlsx'])
if data_file:
    df = pd.read_excel(data_file)
else:
    df = load_data('data/Anirudh_data.xlsx')

st.sidebar.info('Navigate using the tabs above.')

# Tabs
tabs = st.tabs(['Data Visualization', 'Classification', 'Clustering', 'Association Rules', 'Regression'])

# 1. Data Visualization
with tabs[0]:
    st.header('Data Visualization & Insights')
    st.subheader('Dataset Overview')
    st.dataframe(df.head())

    st.subheader('Descriptive Statistics')
    st.dataframe(df.describe(include='all'))

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    st.subheader('Numeric Distributions')
    for col in numeric_cols[:3]:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f'Distribution of {col}')
        st.pyplot(fig)

    st.subheader('Correlation Heatmap')
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    st.subheader('Top Categories')
    for col in cat_cols[:2]:
        st.write(df[col].value_counts().head(10).to_frame())

# 2. Classification
with tabs[1]:
    st.header('Classification')
    all_cols = df.columns
    target = st.selectbox('Select target column', all_cols)
    features = st.multiselect('Select feature columns', all_cols.drop(target), default=list(all_cols.drop(target)[:5]))
    if st.button('Run Classification') and len(features) > 0:
        X = df[features]
        y = df[target]
        results = run_classifiers(X, y)

        st.subheader('Performance Metrics')
        perf_df = pd.DataFrame({name: {
            'Train Acc': res['train_acc'],
            'Test Acc': res['test_acc'],
            'Precision': res['precision'],
            'Recall': res['recall'],
            'F1': res['f1']
        } for name, res in results.items()}).T
        st.dataframe(perf_df.style.format('{:.2f}'))

        st.subheader('ROC Curves')
        fig, ax = plt.subplots()
        for name, res in results.items():
            model = res['model']
            if hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(X)[:,1]
            else:
                y_score = model.decision_function(X)
            fpr, tpr, _ = roc_curve(y, y_score)
            ax.plot(fpr, tpr, label=name)
        ax.plot([0,1],[0,1],'--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        st.pyplot(fig)

        st.subheader('Confusion Matrix')
        sel_model = st.selectbox('Choose model', list(results.keys()))
        cm = results[sel_model]['confusion_matrix']
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

        st.subheader('Predict on New Data')
        pred_file = st.file_uploader('Upload new data (features only)', type=['xlsx'])
        if pred_file:
            pred_df = pd.read_excel(pred_file)
            preds = results[sel_model]['model'].predict(pred_df)
            pred_df['Predicted'] = preds
            st.dataframe(pred_df.head())
            output = io.BytesIO()
            pred_df.to_excel(output, index=False)
            st.download_button('Download Predictions', output.getvalue(), file_name='predictions.xlsx')

# 3. Clustering
with tabs[2]:
    st.header('Clustering (K‑Means)')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cluster_features = st.multiselect('Select features for clustering', numeric_cols, default=list(numeric_cols[:4]))
    n_clusters = st.slider('Number of clusters', 2, 10, 3)
    if st.button('Run Clustering') and len(cluster_features) > 0:
        labels, inertia, _ = run_kmeans(df[cluster_features], n_clusters)
        df['Cluster'] = labels
        st.write(f'Inertia (within‑cluster SSE): {inertia:.2f}')

        fig, ax = plt.subplots()
        sns.countplot(x='Cluster', data=df, ax=ax)
        st.pyplot(fig)

        st.subheader('Cluster Profiles')
        st.dataframe(df.groupby('Cluster')[cluster_features].mean())

        output = io.BytesIO()
        df.to_excel(output, index=False)
        st.download_button('Download Clustered Data', output.getvalue(), file_name='clustered_data.xlsx')

# 4. Association Rules
with tabs[3]:
    st.header('Association Rule Mining')
    if len(cat_cols) >= 2:
        col1 = st.selectbox('First column', cat_cols, key='col1')
        col2 = st.selectbox('Second column', cat_cols, key='col2')
        min_sup = st.slider('Min Support', 0.01, 1.0, 0.1)
        min_conf = st.slider('Min Confidence', 0.01, 1.0, 0.3)
        if st.button('Run Apriori'):
            rules = run_apriori(df, [col1, col2], min_sup, min_conf)
            st.dataframe(rules.sort_values('lift', ascending=False).head(10))
    else:
        st.info('Need at least 2 categorical columns for Apriori.')

# 5. Regression
with tabs[4]:
    st.header('Regression')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 2:
        target_reg = st.selectbox('Target variable', numeric_cols, key='target_reg')
        features_reg = st.multiselect('Feature variables', numeric_cols.drop(target_reg), default=list(numeric_cols.drop(target_reg)[:3]))
        if st.button('Run Regression') and len(features_reg) > 0:
            X_reg = df[features_reg]
            y_reg = df[target_reg]
            reg_results = run_regressions(X_reg, y_reg)

            st.subheader('Regression Metrics')
            reg_df = pd.DataFrame({name: {
                'Train R2': res['train_r2'],
                'Test R2': res['test_r2'],
                'RMSE': res['rmse']
            } for name, res in reg_results.items()}).T
            st.dataframe(reg_df.style.format('{:.2f}'))

            for name, res in reg_results.items():
                fig, ax = plt.subplots()
                ax.scatter(res['y_test'], res['y_pred'])
                ax.set_xlabel('Actual')
                ax.set_ylabel('Predicted')
                ax.set_title(f'{name} Regression')
                st.pyplot(fig)
    else:
        st.info('Need numeric columns for regression analysis.')
