import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """Load Excel data from the given path."""
    return pd.read_excel(file_path)
