import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

def run_apriori(df, cols, min_support=0.1, min_confidence=0.3):
    df_filt = df[cols].astype(str)
    transactions = df_filt.values.tolist()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_te = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df_te, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
    return rules
