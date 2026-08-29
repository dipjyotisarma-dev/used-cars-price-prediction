import pandas as pd
import numpy as np

def clean_used_cars_data(filepath: str) -> pd.DataFrame:
    """
    Reads the raw used cars dataset and applies all Phase 3 cleaning steps.
    """
    # 1. Load data
    df = pd.read_csv(filepath)
    
    # 2. Deduplication
    df = df.drop_duplicates(keep='first')
    
    # 3. Data Type Correction (Price & kmDriven)
    df['AskPrice'] = df['AskPrice'].astype(str).str.replace(r'\D', '', regex=True)
    df['AskPrice'] = pd.to_numeric(df['AskPrice'], errors='coerce').astype('Int64')
    
    df['kmDriven'] = df['kmDriven'].astype(str).str.replace(r'\D', '', regex=True)
    df['kmDriven'] = pd.to_numeric(df['kmDriven'], errors='coerce')
    
    # 4. Handle Implicit Missing Data (Impossible values)
    df = df[df['Year'] != 1900]
    df.loc[(df['kmDriven'] == 0) | (df['kmDriven'] > 800000), 'kmDriven'] = np.nan
    
    # 5. Handle Explicit Missing Data
    df = df.dropna(subset=['kmDriven'])
    
    return df