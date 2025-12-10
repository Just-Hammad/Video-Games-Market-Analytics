import pandas as pd

try:
    df = pd.read_parquet('data/mainDS.parquet')
    print("Columns in mainDS.parquet:")
    print(df.columns.tolist())
    
    required_columns = [
        'Estimated owners', 'AppID', 'Name', 'Release date', 'Positive', 'Negative',
        'Price', 'DLC count', 'Achievements', 'Genres', 'Tags', 'Steam Score',
        'Peak CCU', 'Average playtime forever'
    ]
    
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        print(f"Missing columns: {missing}")
    else:
        print("All required columns present.")
        
except Exception as e:
    print(f"Error reading parquet: {e}")
