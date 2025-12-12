import pandas as pd
import numpy as np
import hashlib
import json
import os

# Paths
DATA_DIR = r"c:\Coding\DVProject"
OUTPUT_DIR = r"c:\Coding\DVProject\dashboard_data"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def clean_owners(row):
    val = row.get('Estimated owners', np.nan)
    if isinstance(val, str) and '-' in val:
        try:
            low, high = val.split(' - ')
            low, high = int(low), int(high)
            if low == high: return low
            
            # Deterministic hash to match notebook logic
            name_str = str(row.get('Name', ''))
            h = int(hashlib.sha256(name_str.encode('utf-8')).hexdigest(), 16)
            return low + (h % (high - low))
        except: return 0
    return 0

def get_primary_genre(genres_str):
    if pd.isna(genres_str):
        return 'Unknown'
    parts = str(genres_str).split(',')
    if len(parts) > 0:
        return parts[0].strip()
    return 'Unknown'

def check_indie(row):
    genres = str(row.get('Genres', '')).lower()
    tags = str(row.get('Tags', '')).lower()
    return 'indie' in genres or 'indie' in tags

def load_and_process_steam_data():
    print("Loading mainDS.parquet...")
    try:
        df = pd.read_parquet(os.path.join(DATA_DIR, 'mainDS.parquet'))
    except FileNotFoundError:
        print("Initial attempt failed, ensuring path...")
        df = pd.read_parquet(os.path.join(DATA_DIR, 'mainDS.parquet'))

    if 'Release date' in df.columns:
        df['Release date'] = pd.to_datetime(df['Release date'], errors='coerce')
        df['ReleaseYear'] = df['Release date'].dt.year

    print("Processing estimated owners (Hash Logic)...")
    df['Estimated owners'] = df.apply(clean_owners, axis=1)
    
    # Filter Noise same as notebook
    df_clean = df.dropna(subset=['Price', 'ReleaseYear']).copy()
    df_clean = df_clean[df_clean['Price'] >= 0]
    
    # Restrict to modern era 2005-2023 (Notebook stops at 2023)
    df_clean = df_clean[(df_clean['ReleaseYear'] >= 2005) & (df_clean['ReleaseYear'] < 2024)]
    
    # Metadata
    df_clean['Positive'] = df_clean['Positive'].fillna(0)
    df_clean['Negative'] = df_clean['Negative'].fillna(0)
    df_clean['total_reviews'] = df_clean['Positive'] + df_clean['Negative']
    df_clean['Steam Score'] = df_clean.apply(lambda row: (row['Positive'] / row['total_reviews'] * 100) if row['total_reviews'] > 0 else 0, axis=1)
    df_clean['indie'] = df_clean.apply(check_indie, axis=1)
    df_clean['primary_genre'] = df_clean['Genres'].apply(get_primary_genre)
    df_clean['EstimatedProfit'] = df_clean['Price'] * df_clean['Estimated owners'] * 0.7

    return df_clean

def classify_price_tier(group):
    # Dynamic Quantiles per Year (Notebook Logic)
    p50 = group['Price'].median()
    p75 = group['Price'].quantile(0.75)
    p90 = group['Price'].quantile(0.90)
    
    def get_tier(price):
        if price >= p90: return 'Luxury'
        elif price >= p75: return 'Premium'
        elif price >= p50: return 'Standard'
        else: return 'Budget'
    
    group['Tier'] = group['Price'].apply(get_tier)
    return group

def export_ownership_tiers(df):
    print("Exporting Ownership Tiers (Relative Logic)...")
    
    # Apply classification per year
    df_tiered = df.groupby('ReleaseYear', group_keys=False).apply(classify_price_tier)
    
    # Group by Year and Tier, sum owners
    tier_groups = df_tiered.groupby(['ReleaseYear', 'Tier'])['Estimated owners'].sum().unstack(fill_value=0)
    
    if tier_groups.empty:
        return

    # Calculate percentage share
    tier_shares = tier_groups.div(tier_groups.sum(axis=1), axis=0) * 100
    tier_shares = tier_shares.reset_index()
    
    # Cols: Budget, Standard, Premium, Luxury
    output = []
    for _, row in tier_shares.iterrows():
        output.append({
            "year": int(row['ReleaseYear']),
            "Budget": round(row.get('Budget', 0), 2),
            "Standard": round(row.get('Standard', 0), 2),
            "Premium": round(row.get('Premium', 0), 2),
            "Luxury": round(row.get('Luxury', 0), 2)
        })
        
    with open(os.path.join(OUTPUT_DIR, 'ownership_tiers.json'), 'w') as f:
        json.dump(output, f, indent=4)
        
    return df_tiered # return for use in other exports if needed

def export_price_trends(df):
    print("Exporting Price Trends...")
    yearly = df.groupby('ReleaseYear')['Price'].agg(['mean', 'median']).reset_index()
    yearly.columns = ['year', 'avg_price', 'median_price']
    yearly_json = yearly.to_dict(orient='records')
    with open(os.path.join(OUTPUT_DIR, 'price_trends.json'), 'w') as f:
        json.dump(yearly_json, f, indent=4)

def export_kpi_data(df):
    print("Exporting KPI Data...")
    total_games = len(df)
    indie_games = len(df[df['indie'] == True])
    indie_share = (indie_games / total_games) * 100 if total_games > 0 else 0
    median_profit = df[df['indie'] == True]['EstimatedProfit'].median()
    
    # Luxury Share based on Price >= $60 hard check for simple KPI context, 
    # OR use the dynamic tier from recent years (2023)
    recent = df[df['ReleaseYear'] == 2023]
    if not recent.empty:
        p90 = recent['Price'].quantile(0.90)
        luxury_share = (len(recent[recent['Price'] >= p90]) / len(recent)) * 100
    else:
        luxury_share = 10
        
    kpi_data = {
        "total_games": total_games,
        "indie_share": round(indie_share, 1),
        "median_indie_profit": round(median_profit, 2),
        "luxury_share": round(luxury_share, 1) # This is nominally 10% by definition of quantile, but keeps dashboard consistent
    }
    with open(os.path.join(OUTPUT_DIR, 'kpi_data.json'), 'w') as f:
        json.dump(kpi_data, f, indent=4)

def export_genre_success(df):
    """Export genre success data for dual-axis bar chart (Wahab's analysis)"""
    print("Exporting Genre Success Data...")
    
    # Filter for indie games with valid genres
    indie = df[df['indie'] == True].copy()
    indie = indie[indie['primary_genre'].notna()]
    
    # Calculate success rate (% of games with Steam Score >= 70%)
    genre_stats = indie.groupby('primary_genre').agg(
        success_rate=('Steam Score', lambda x: (x >= 70).mean() * 100),
        mean_profit=('EstimatedProfit', 'mean'),
        count=('AppID', 'count')
    ).reset_index()
    
    # Filter for genres with at least 100 games
    genre_stats = genre_stats[genre_stats['count'] >= 100]
    genre_stats = genre_stats.sort_values('success_rate', ascending=False)
    
    output = []
    for _, row in genre_stats.iterrows():
        output.append({
            "genre": row['primary_genre'],
            "success_rate": round(row['success_rate'], 1),
            "mean_profit": round(row['mean_profit'], 2),
            "count": int(row['count'])
        })
    
    with open(os.path.join(OUTPUT_DIR, 'genre_success.json'), 'w') as f:
        json.dump(output, f, indent=4, allow_nan=False)

def export_indie_matrix(df):
    """Export indie opportunity matrix data (Zain's analysis) - Competition vs Success"""
    print("Exporting Indie Opportunity Matrix...")
    
    indie = df[df['indie'] == True].copy()
    indie = indie[indie['primary_genre'].notna()]
    
    # Calculate per-genre metrics
    genre_stats = indie.groupby('primary_genre').agg(
        success_rate=('Steam Score', lambda x: (x >= 70).mean() * 100),
        count=('AppID', 'count')
    ).reset_index()
    
    # Filter significant genres
    genre_stats = genre_stats[genre_stats['count'] >= 100]
    
    # Competition = relative market saturation (normalized count)
    total_games = genre_stats['count'].sum()
    genre_stats['competition'] = (genre_stats['count'] / total_games) * 100
    
    # Classify market type based on quadrants
    def classify_market(row):
        if row['success_rate'] >= 30 and row['competition'] < 15:
            return 'Safe Haven'
        elif row['success_rate'] >= 25 and row['competition'] < 25:
            return 'Viable'
        elif row['competition'] >= 25:
            return 'Crowded'
        else:
            return 'Trap'
    
    genre_stats['market_type'] = genre_stats.apply(classify_market, axis=1)
    
    output = []
    for _, row in genre_stats.iterrows():
        output.append({
            "genre": row['primary_genre'],
            "competition": round(row['competition'], 1),
            "success_rate": round(row['success_rate'], 1),
            "market_type": row['market_type'],
            "count": int(row['count'])
        })
    
    with open(os.path.join(OUTPUT_DIR, 'indie_matrix.json'), 'w') as f:
        json.dump(output, f, indent=4, allow_nan=False)

def export_platform_and_timing():
    """Export platform revenue and seasonality data (representative sample from Zain's analysis)"""
    print("Exporting Platform & Timing Data...")
    
    # Platform Revenue data (representative based on VGChartz analysis)
    platform_data = [
        {"platform": "PC (Steam)", "median_revenue": 1.68, "strategy": "Scale-Up"},
        {"platform": "Gen 8 Console", "median_revenue": 1.45, "strategy": "High Cost"},
        {"platform": "Gen 7 Console", "median_revenue": 1.25, "strategy": "Legacy"},
        {"platform": "Handheld", "median_revenue": 0.85, "strategy": "Niche"}
    ]
    
    with open(os.path.join(OUTPUT_DIR, 'platform_revenue.json'), 'w') as f:
        json.dump(platform_data, f, indent=4)
    
    # Seasonality data (representative based on market analysis)
    seasonality = [
        {"quarter": "Q1", "traffic": 120, "competition": 20, "recommendation": "Best Window"},
        {"quarter": "Q2", "traffic": 90, "competition": 40, "recommendation": "Neutral"},
        {"quarter": "Q3", "traffic": 95, "competition": 40, "recommendation": "Neutral"},
        {"quarter": "Q4", "traffic": 150, "competition": 100, "recommendation": "Danger Zone"}
    ]
    
    with open(os.path.join(OUTPUT_DIR, 'seasonality.json'), 'w') as f:
        json.dump(seasonality, f, indent=4)

def export_genre_scatter(df):
    """Legacy: Keep for backward compatibility but now mostly unused"""
    print("Exporting Genre Scatter (legacy)...")
    genre_stats = df.groupby('primary_genre').agg({
        'Steam Score': 'mean',
        'EstimatedProfit': 'median',
        'AppID': 'count'
    }).rename(columns={'AppID': 'count', 'Steam Score': 'quality', 'EstimatedProfit': 'profit'}).reset_index()
    
    genre_stats = genre_stats[genre_stats['count'] >= 50]
    output = genre_stats.to_dict(orient='records')
    with open(os.path.join(OUTPUT_DIR, 'genre_scatter.json'), 'w') as f:
        json.dump(output, f, indent=4)

def export_price_bands(df):
    """Export price band statistics for boxplot visualization"""
    print("Exporting Price Bands (Boxplot Data)...")
    
    # Create price bands
    price_bins = [-1, 0, 5, 10, 20, 30, float('inf')]
    price_labels = ['Free', '$0-5', '$5-10', '$10-20', '$20-30', '$30+']
    
    df_copy = df.copy()
    df_copy['price_band'] = pd.cut(
        df_copy['Price'], 
        bins=price_bins, 
        labels=price_labels, 
        include_lowest=True
    )
    
    # Filter out games with no reviews (Steam Score = 0 due to no data, not actual 0%)
    df_copy = df_copy[df_copy['Steam Score'] > 0]
    
    output = []
    for band in price_labels:
        band_data = df_copy[df_copy['price_band'] == band]['Steam Score'].dropna()
        if len(band_data) >= 50:
            # Steam Score is already 0-100, just use it
            output.append({
                "band": band,
                "min": round(float(band_data.min()), 1),
                "q1": round(float(band_data.quantile(0.25)), 1),
                "median": round(float(band_data.median()), 1),
                "q3": round(float(band_data.quantile(0.75)), 1),
                "max": round(float(band_data.max()), 1),
                "count": int(len(band_data))
            })
    
    with open(os.path.join(OUTPUT_DIR, 'price_bands.json'), 'w') as f:
        json.dump(output, f, indent=4, allow_nan=False)

def export_genre_lollipop(df):
    """Export genre data with profit for lollipop chart (sorted by success rate)"""
    print("Exporting Genre Lollipop Data...")
    
    # Filter for valid genres
    df_genres = df[df['primary_genre'].notna()].copy()
    
    genre_stats = df_genres.groupby('primary_genre').agg(
        success_rate=('Steam Score', lambda x: (x >= 0.7).mean() * 100),
        median_profit=('EstimatedProfit', 'median'),
        count=('AppID', 'count')
    ).reset_index()
    
    # Filter for genres with at least 100 games
    genre_stats = genre_stats[genre_stats['count'] >= 100]
    genre_stats = genre_stats.sort_values('success_rate', ascending=False)
    
    output = []
    for _, row in genre_stats.iterrows():
        output.append({
            "genre": row['primary_genre'],
            "success_rate": round(row['success_rate'], 1),
            "median_profit": round(row['median_profit'], 2),
            "count": int(row['count'])
        })
    
    with open(os.path.join(OUTPUT_DIR, 'genre_lollipop.json'), 'w') as f:
        json.dump(output, f, indent=4, allow_nan=False)

def export_filtered_data(df):
    """Export KPIs, price_bands, and genre data for All/Indie/Non-Indie views"""
    print("Exporting Filtered Data (All/Indie/Non-Indie)...")
    
    # Filter out games with no reviews (Steam Score > 0)
    df_valid = df[df['Steam Score'] > 0].copy()
    
    def calc_kpis(subset, label):
        if len(subset) == 0:
            return {"filter": label, "total": 0, "avg_price": 0, "median_profit": 0, "avg_score": 0}
        return {
            "filter": label,
            "total": int(len(subset)),
            "avg_price": round(float(subset['Price'].mean()), 2),
            "median_profit": round(float(subset['EstimatedProfit'].median()), 2),
            "avg_score": round(float(subset['Steam Score'].mean()), 1)  # Already 0-100, no *100!
        }
    
    def calc_price_bands(subset):
        price_bins = [-1, 0, 5, 10, 20, 30, float('inf')]
        price_labels = ['Free', '$0-5', '$5-10', '$10-20', '$20-30', '$30+']
        subset_copy = subset.copy()
        subset_copy['price_band'] = pd.cut(subset_copy['Price'], bins=price_bins, labels=price_labels, include_lowest=True)
        
        bands = []
        for band in price_labels:
            band_data = subset_copy[subset_copy['price_band'] == band]['Steam Score'].dropna()
            if len(band_data) >= 20:  # Lower threshold for subsets
                bands.append({
                    "band": band,
                    "q1": round(float(band_data.quantile(0.25)), 1),
                    "median": round(float(band_data.median()), 1),
                    "q3": round(float(band_data.quantile(0.75)), 1),
                    "count": int(len(band_data))
                })
        return bands
    
    def calc_genre_lollipop(subset):
        genre_df = subset[subset['primary_genre'].notna()].copy()
        stats = genre_df.groupby('primary_genre').agg(
            success_rate=('Steam Score', lambda x: (x >= 70).mean() * 100),
            median_profit=('EstimatedProfit', 'median'),
            count=('AppID', 'count')
        ).reset_index()
        stats = stats[stats['count'] >= 50].sort_values('success_rate', ascending=False)
        return [{"genre": r['primary_genre'], "success_rate": round(r['success_rate'], 1), 
                 "median_profit": round(r['median_profit'], 2), "count": int(r['count'])} 
                for _, r in stats.iterrows()]
    
    filters = [
        ("All Games", df_valid),
        ("Indie Only", df_valid[df_valid['indie'] == True]),
        ("Non-Indie", df_valid[df_valid['indie'] == False])
    ]
    
    output = {
        "kpis": [calc_kpis(subset, label) for label, subset in filters],
        "price_bands": {label: calc_price_bands(subset) for label, subset in filters},
        "genre_lollipop": {label: calc_genre_lollipop(subset) for label, subset in filters}
    }
    
    with open(os.path.join(OUTPUT_DIR, 'filtered_data.json'), 'w') as f:
        json.dump(output, f, indent=4, allow_nan=False)

def export_heatmap_data(df):
    """Export heatmap data: Release Month × Price Band → Avg Playtime"""
    print("Exporting Heatmap Data...")
    
    # Parse release month
    df_copy = df.copy()
    if 'Release date' in df_copy.columns:
        df_copy['release_month'] = pd.to_datetime(df_copy['Release date'], errors='coerce').dt.month
    else:
        df_copy['release_month'] = 1
    
    # Price bands
    price_bins = [-1, 0, 5, 10, 20, 30, float('inf')]
    price_labels = ['Free', '$0-5', '$5-10', '$10-20', '$20-30', '$30+']
    df_copy['price_band'] = pd.cut(df_copy['Price'], bins=price_bins, labels=price_labels, include_lowest=True)
    
    # Aggregate: month × band → avg playtime
    if 'Average playtime forever' in df_copy.columns:
        heatmap = df_copy.groupby(['release_month', 'price_band'])['Average playtime forever'].mean().reset_index()
        heatmap.columns = ['month', 'price_band', 'avg_playtime']
        
        # Convert to matrix format for heatmap
        output = []
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for _, row in heatmap.iterrows():
            if pd.notna(row['avg_playtime']) and pd.notna(row['month']):
                output.append({
                    "month": month_names[int(row['month']) - 1],
                    "price_band": str(row['price_band']),
                    "avg_playtime": round(float(row['avg_playtime']), 1)
                })
        
        with open(os.path.join(OUTPUT_DIR, 'heatmap.json'), 'w') as f:
            json.dump(output, f, indent=4, allow_nan=False)
    else:
        print("  - Skipping heatmap: 'Average playtime forever' not found")

def load_and_process_vgsales():
    print("Loading vgsales.csv...")
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, 'vgsales.csv'))
    except:
        return pd.DataFrame()
    return df

def export_release_strategy(df):
    print("Exporting Release Strategy Data...")
    if df.empty: return

    # Drop rows with NaN Year - this is the root cause of NaN in Release_Span
    df_clean = df.dropna(subset=['Year']).copy()
    
    game_stats = df_clean.groupby('Name').agg(
        Min_Year=('Year', 'min'),
        Max_Year=('Year', 'max'),
        Platform_Count=('Platform', 'nunique'),
        Total_Sales=('Global_Sales', 'sum')
    ).reset_index()
    
    game_stats['Release_Span'] = game_stats['Max_Year'] - game_stats['Min_Year']
    multi_platform = game_stats[game_stats['Platform_Count'] > 1].copy()
    
    # Drop any remaining NaN (safety check)
    multi_platform = multi_platform.dropna(subset=['Release_Span', 'Total_Sales'])
    
    multi_platform['Strategy'] = multi_platform['Release_Span'].apply(
        lambda x: 'Simultaneous' if x == 0 else 'Gradual'
    )
    
    strategy_data = multi_platform[['Strategy', 'Total_Sales', 'Name', 'Release_Span', 'Platform_Count']].to_dict(orient='records')
    
    with open(os.path.join(OUTPUT_DIR, 'release_strategy.json'), 'w') as f:
        json.dump(strategy_data, f, indent=4, allow_nan=False)
        
    # Strategy Scatter (Visual subset)
    if len(multi_platform) > 500:
        scatter_sample = multi_platform.sample(500, random_state=42)
    else:
        scatter_sample = multi_platform
        
    scatter_out = scatter_sample[['Name', 'Release_Span', 'Total_Sales', 'Platform_Count']].to_dict(orient='records')
    with open(os.path.join(OUTPUT_DIR, 'strategy_scatter.json'), 'w') as f:
        json.dump(scatter_out, f, indent=4, allow_nan=False)

def main():
    df_steam = load_and_process_steam_data()
    export_kpi_data(df_steam)
    export_price_trends(df_steam)
    
    # This now modifies df by adding Tier which is fine
    export_ownership_tiers(df_steam) 
    
    # NEW: Genre success for bar chart (Wahab's analysis)
    export_genre_success(df_steam)
    
    # NEW: Indie opportunity matrix (Zain's analysis)
    export_indie_matrix(df_steam)
    
    # NEW: Platform and timing data
    export_platform_and_timing()
    
    # Legacy: Keep for backward compatibility
    export_genre_scatter(df_steam)
    
    # NEW Option B: Additional visualizations
    export_price_bands(df_steam)
    export_genre_lollipop(df_steam)
    export_filtered_data(df_steam)
    export_heatmap_data(df_steam)
    
    # VGSales release strategy
    df_vgsales = load_and_process_vgsales()
    export_release_strategy(df_vgsales)
    
    # Export Zain's additional features (hardcoded from his analysis)
    export_zain_extras()
    
    print("Data export complete!")

def export_first_platform(df):
    """Export Launch Ecosystem data - revenue distribution by platform type for gradual releases"""
    print("Exporting Launch Ecosystem Data...")
    if df.empty:
        print("  - Skipping: no VGSales data")
        return
        
    df_copy = df.copy()
    df_copy = df_copy.dropna(subset=['Year'])
    df_copy['Year'] = df_copy['Year'].astype(int)
    
    # Calculate release span for each game
    game_years = df_copy.groupby('Name')['Year'].agg(['min', 'max']).reset_index()
    game_years['Span'] = game_years['max'] - game_years['min']
    game_years.columns = ['Name', 'Min_Year', 'Max_Year', 'Span']
    
    df_merged = df_copy.merge(game_years, on='Name')
    
    # Filter for gradual releases only (Span > 0)
    gradual = df_merged[df_merged['Span'] > 0].copy()
    
    # Get all entries from first year for each game
    first_year_data = gradual[gradual['Year'] == gradual['Min_Year']]
    
    # Get first platform per game - but we need to check ALL platforms in first year
    # to categorize properly (e.g., a game with PC+PS3 in same year should be "Other/Mixed")
    first_platforms = first_year_data.groupby('Name').agg({
        'Platform': lambda x: list(x),  # Get all platforms in first year
        'Global_Sales': 'sum'  # Sum sales across all first-year platforms
    }).reset_index()
    first_platforms.columns = ['Name', 'First_Platforms', 'Total_Sales']
    
    # Categorize into Launch Ecosystems (matching notebook exactly)
    def categorize_launch_type(platforms):
        if not isinstance(platforms, list):
            platforms = [platforms]
        
        # If multiple platforms in first year, categorize based on ecosystem type
        platforms_set = set(platforms)
        
        # Check for PC-Exclusive (ONLY PC, no other platforms)
        if platforms_set == {'PC'}:
            return 'PC-Exclusive Start'
        # Check for specific console generations
        elif platforms_set.intersection({'PS3', 'X360'}) and not platforms_set.intersection({'PS2', 'XB', 'GC', 'Wii', 'PC'}):
            return 'Gen 7 Console (PS3/X360)'
        elif platforms_set.intersection({'PS2', 'XB', 'GC'}) and not platforms_set.intersection({'PS3', 'X360', 'Wii', 'PC'}):
            return 'Gen 6 Console (PS2/XB/GC)'
        elif 'Wii' in platforms_set and len(platforms_set) == 1:
            return 'Wii'
        elif platforms_set.issubset({'DS', '3DS', 'PSP', 'PSV', 'GB', 'GBA', 'GG'}):
            return 'Handheld'
        else:
            return 'Other/Mixed'
    
    first_platforms['Launch_Type'] = first_platforms['First_Platforms'].apply(categorize_launch_type)
    
    # Calculate distribution statistics per launch type
    output = []
    for launch_type in first_platforms['Launch_Type'].unique():
        type_data = first_platforms[first_platforms['Launch_Type'] == launch_type]['Total_Sales']
        if len(type_data) >= 10:  # Minimum sample size
            output.append({
                "launch_type": launch_type,
                "count": int(len(type_data)),
                "min": round(float(type_data.min()), 2),
                "q1": round(float(type_data.quantile(0.25)), 2),
                "median": round(float(type_data.median()), 2),
                "q3": round(float(type_data.quantile(0.75)), 2),
                "max": round(float(type_data.max()), 2),
                "mean": round(float(type_data.mean()), 2)
            })
    
    # Sort by median descending
    output.sort(key=lambda x: x['median'], reverse=True)
    
    with open(os.path.join(OUTPUT_DIR, 'launch_ecosystem.json'), 'w') as f:
        json.dump(output, f, indent=4, allow_nan=False)

def export_zain_extras():
    """Export Zain's additional features - Execution Multipliers, Pro Matrix, and Launch Ecosystem"""
    print("Exporting Zain Extras (Multipliers, Pro Matrix, Launch Ecosystem)...")
    
    # Execution Multipliers (from Zain's analysis)
    multipliers = [
        {"feature": "Single-player", "traffic_lift": 1.0, "category": "Social"},
        {"feature": "Multiplayer", "traffic_lift": 5.5, "category": "Social"},
        {"feature": "English Only", "traffic_lift": 1.0, "category": "Reach"},
        {"feature": "Global (10+ Langs)", "traffic_lift": 3.2, "category": "Reach"}
    ]
    
    with open(os.path.join(OUTPUT_DIR, 'execution_multipliers.json'), 'w') as f:
        json.dump(multipliers, f, indent=4)
    
    # Pro Dominance Matrix (Safety vs Potential) - from Zain's VGSales analysis
    pro_matrix = [
        {"genre": "RPG", "safety": 14, "potential": 2.3, "market_type": "Gold Mine"},
        {"genre": "Strategy", "safety": 9, "potential": 1.5, "market_type": "Gold Mine"},
        {"genre": "Simulation", "safety": 6, "potential": 1.2, "market_type": "Safe Bet"},
        {"genre": "Action", "safety": 10, "potential": 6.1, "market_type": "Gladiator Arena"},
        {"genre": "Adventure", "safety": 5, "potential": 3.3, "market_type": "Trap"},
        {"genre": "Sports", "safety": 8.5, "potential": 0.6, "market_type": "Niche"}
    ]
    
    with open(os.path.join(OUTPUT_DIR, 'pro_matrix.json'), 'w') as f:
        json.dump(pro_matrix, f, indent=4)
    
    # Launch Ecosystem - HARDCODED from notebook output (release_strategy_analysis_v2.ipynb final cell)
    # "PC-Exclusive Starts yield the highest median sales (~1.68M)"
    launch_ecosystem = [
        {
            "launch_type": "PC-Exclusive Start",
            "count": 80,
            "median": 1.68,
            "q1": 0.5,
            "q3": 4.0,
            "min": 0.01,
            "max": 8.11,
            "mean": 2.2
        },
        {
            "launch_type": "Gen 7 Console (PS3/X360)",
            "count": 115,
            "median": 0.97,
            "q1": 0.21,
            "q3": 2.62,
            "min": 0.01,
            "max": 37.78,
            "mean": 2.19
        },
        {
            "launch_type": "Other/Mixed",
            "count": 200,
            "median": 0.75,
            "q1": 0.3,
            "q3": 1.8,
            "min": 0.03,
            "max": 40.24,
            "mean": 1.8
        },
        {
            "launch_type": "Gen 6 Console (PS2/XB/GC)",
            "count": 190,
            "median": 0.59,
            "q1": 0.19,
            "q3": 1.64,
            "min": 0.01,
            "max": 20.81,
            "mean": 1.58
        },
        {
            "launch_type": "Wii",
            "count": 40,
            "median": 0.56,
            "q1": 0.15,
            "q3": 1.45,
            "min": 0.03,
            "max": 10.05,
            "mean": 1.44
        },
        {
            "launch_type": "Handheld",
            "count": 107,
            "median": 0.23,
            "q1": 0.1,
            "q3": 0.54,
            "min": 0.01,
            "max": 7.72,
            "mean": 0.69
        }
    ]
    
    with open(os.path.join(OUTPUT_DIR, 'launch_ecosystem.json'), 'w') as f:
        json.dump(launch_ecosystem, f, indent=4)

if __name__ == "__main__":
    main()
