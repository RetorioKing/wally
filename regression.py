from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def run_regressions(X, y):
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.01),
        'Decision Tree': DecisionTreeRegressor(random_state=42)
    }
    results = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'train_r2': r2_score(y_train, model.predict(X_train)),
            'test_r2': r2_score(y_test, y_pred),
            'rmse': mean_squared_error(y_test, y_pred, squared=False),
            'model': model,
            'y_test': y_test,
            'y_pred': y_pred
        }
    return results
