import pandas as pd
import numpy as np

def engineer_stateless_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all stateless feature engineering transformations."""
    df = df.copy()
    
    # 1. Target Transformation
    if 'AskPrice' in df.columns:
        df['AskPrice_Log'] = np.log1p(df['AskPrice'])
        
    # 2. Clean FuelType
    if 'FuelType' in df.columns:
        df['FuelType'] = df['FuelType'].str.lower()
        
    # 3. Drop redundant / useless columns
    cols_to_drop = ['Year', 'AdditionInfo', 'PostedDate', 'model']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        
    # 4. Cap kmDriven outliers
    if 'kmDriven' in df.columns:
        df['kmDriven'] = df['kmDriven'].clip(upper=300000)
        
    return df