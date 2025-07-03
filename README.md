# Streamlit Dashboard: Multi-Tab Analytics

## Features
- **Data Visualization**: 10+ interactive insights with filters and sliders.
- **Classification**: KNN, Decision Tree, Random Forest, GBRT; metrics table, confusion matrix, ROC curve, upload & download predictions.
- **Clustering**: K‑Means with dynamic cluster slider, elbow plot, personas, download labeled data.
- **Association Rules**: Apriori with adjustable support & confidence; top‑10 rules.
- **Regression**: Linear, Lasso, Ridge & Decision Tree regressors with performance comparison.

## Quick Start
1. Install requirements  
   ```bash
   pip install -r requirements.txt
   ```
2. Place your dataset in `data/Anirudh_data.xlsx` (replace the placeholder).
3. Launch the dashboard  
   ```bash
   streamlit run app.py
   ```

Deploy on Streamlit Cloud by pushing this repo to GitHub and selecting `app.py` as the entry point.
