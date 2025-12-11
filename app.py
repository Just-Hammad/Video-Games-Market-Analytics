import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import zlib

# Page configuration
st.set_page_config(
    page_title="Indie Game Analytics",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS STYLING (Adjusted for Dashboard Density)
# ============================================================================
st.markdown("""
<style>
    /* Main background and text */
    .stApp {
        background: linear-gradient(135deg, #232946 0%, #121629 100%);
        color: #F4F4F6;
    }
    
    /* Reduce top padding for dashboard feel */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1 {
        background: linear-gradient(90deg, #F9D923, #00A8CC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900 !important;
        font-size: 2.2rem !important; /* Smaller title */
        text-align: left;
        padding: 0;
        margin-bottom: 0;
        letter-spacing: 2px;
    }
    
    /* Chart Titles */
    h3 {
        color: #00A8CC !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
        border-bottom: 1px solid rgba(249,217,35,0.3);
        padding-bottom: 0.5rem;
    }

    .metric-card {
        background: linear-gradient(145deg, rgba(0,168,204,0.12), rgba(249,217,35,0.08));
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 4px 12px rgba(0,0,0,0.18);
        text-align: center;
        height: 100%;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 900;
        background: linear-gradient(90deg, #F9D923, #00A8CC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-label {
        color: #B8B8D1;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.2rem;
    }
    
    /* Compact Insight Card */
    .insight-card {
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
        padding: 0.8rem;
        border-left: 3px solid #00A8CC;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }
    .insight-card h4 {
        color: #F9D923;
        font-size: 0.95rem;
        margin: 0 0 0.3rem 0;
    }
    .insight-card p {
        color: #F4F4F6;
        margin: 0;
        font-size: 0.8rem;
        line-height: 1.2;
    }
    
    .stRadio > div {
        background: rgba(0,168,204,0.08);
        border-radius: 12px;
        padding: 0.5rem;
        color: white;
    }
    .stRadio label {
        color: white !important;
    }
    
    .stPlotlyChart {
        background: rgba(0,168,204,0.04);
        border-radius: 12px;
        border: 1px solid rgba(0,168,204,0.08);
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    hr {
        margin: 1rem 0;
        border-color: rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Color scheme
COLOR_MAIN = '#00A8CC'  # blue
COLOR_PROFIT = '#F9D923'  # yellow
COLOR_HIGHLIGHT = '#F96D00'  # orange
COLOR_BG = '#232946'  # dark blue

# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_data
def load_data():
    """Load and prepare the dataset"""
    try:
        df = pd.read_parquet('data/mainDS.parquet')
    except Exception as e:
        # Fallback for demo if file missing, creates empty structure
        return pd.DataFrame(columns=['Name', 'Release date', 'Positive', 'Negative', 'Price', 'Estimated owners', 'Genres', 'Tags', 'Peak CCU', 'Average playtime forever'])

    # Calculate owners with deterministic seeding
    def calculate_owners(row):
        owner_str = row.get('Estimated owners', np.nan)
        if pd.isna(owner_str):
            return 0
        try:
            ranges = str(owner_str).split(' - ')
            lower = int(ranges[0])
            upper = int(ranges[1])
            seed_val = str(row.get('AppID', row.get('Name', '')))
            hash_val = zlib.crc32(seed_val.encode('utf-8'))
            factor = (hash_val & 0xffffffff) / 0xffffffff
            estimated = lower + (upper - lower) * factor
            return int(estimated)
        except:
            return 0
    
    if 'Estimated owners' in df.columns:
        df['Estimated owners'] = df.apply(calculate_owners, axis=1)

    # Date processing
    df['Release date'] = pd.to_datetime(df['Release date'], errors='coerce')
    df['ReleaseYear'] = df['Release date'].dt.year
    df['ReleaseMonth'] = df['Release date'].dt.month

    # Calculated fields
    df['Positive'] = df['Positive'].fillna(0)
    df['Negative'] = df['Negative'].fillna(0)
    df['total_reviews'] = df['Positive'] + df['Negative']
    
    if 'Steam Score' not in df.columns:
        df['Steam Score'] = df.apply(lambda row: row['Positive'] / row['total_reviews'] if row['total_reviews'] > 0 else 0, axis=1)

    df['Price'] = df['Price'].fillna(0)
    df['EstimatedProfit'] = df['Price'] * df['Estimated owners'] * 0.7
    
    # Average playtime forever
    if 'Average playtime forever' not in df.columns:
        df['Average playtime forever'] = 0
    else:
        df['Average playtime forever'] = df['Average playtime forever'].fillna(0)
    
    # Peak CCU
    if 'Peak CCU' not in df.columns:
        df['Peak CCU'] = 0
    else:
        df['Peak CCU'] = df['Peak CCU'].fillna(0)
    
    # Indie status
    def check_indie(row):
        genres = str(row.get('Genres', '')).lower()
        tags = str(row.get('Tags', '')).lower()
        return 'indie' in genres or 'indie' in tags

    if 'indie' not in df.columns:
        df['indie'] = df.apply(check_indie, axis=1)

    # Create price_bin
    df['price_bin'] = pd.cut(
        df['Price'],
        bins=[-1, 0, 5, 10, 20, 30, np.inf],
        labels=['Free', '$0-5', '$5-10', '$10-20', '$20-30', '$30+'],
        include_lowest=True
    )

    # Primary genre
    def get_primary_genre(genres_str):
        if pd.isna(genres_str):
            return 'Unknown'
        parts = str(genres_str).split(',')
        if len(parts) > 0:
            return parts[0].strip()
        return 'Unknown'
    
    if 'Genres' in df.columns:
        df['primary_genre'] = df['Genres'].apply(get_primary_genre)
    else:
        df['primary_genre'] = 'Unknown'

    return df

# Load data
df = load_data()

@st.cache_data
def load_vgsales():
    try:
        df = pd.read_csv("data/vgsales_cleaned.csv")
    except Exception as e:
        return pd.DataFrame()
    return df

vgsales_df = load_vgsales()

# ============================================================================
# HEADER & KPIS
# ============================================================================
top_left, top_right = st.columns([1, 2])
with top_left:
    st.markdown("<h1>🎮 Indie Analytics</h1>", unsafe_allow_html=True)
with top_right:
     st.markdown("<p style='text-align: right; color: #888; margin-top: 15px;'>Strategic insights for Steam game success</p>", unsafe_allow_html=True)

# Metrics Row
m_col1, m_col2, m_col3, m_col4 = st.columns([1.2, 1, 1, 1])

with m_col1:
    game_filter = st.radio(
        "Filter Scope:",
        options=['All Games', 'Indie Only', 'Non-Indie Only'],
        horizontal=True,
        label_visibility="collapsed"
    )

with m_col2:
    st.markdown(f"""<div class="metric-card"><div class="metric-value">{len(df):,}</div><div class="metric-label">Total Games</div></div>""", unsafe_allow_html=True)

with m_col3:
    indie_count = len(df[df['indie'] == True]) if not df.empty else 0
    st.markdown(f"""<div class="metric-card"><div class="metric-value">{indie_count:,}</div><div class="metric-label">Indie Games</div></div>""", unsafe_allow_html=True)

with m_col4:
    indie_pct = (indie_count / len(df) * 100) if len(df) > 0 else 0
    st.markdown(f"""<div class="metric-card"><div class="metric-value">{indie_pct:.1f}%</div><div class="metric-label">Market Share</div></div>""", unsafe_allow_html=True)

# Apply filter
if game_filter == 'Indie Only':
    filtered_df = df[df['indie'] == True].copy()
elif game_filter == 'Non-Indie Only':
    filtered_df = df[df['indie'] == False].copy()
else:
    filtered_df = df.copy()

filter_label = "All Games" if game_filter == 'All Games' else ("Indie Games" if game_filter == 'Indie Only' else "Non-Indie Games")

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================================
# DASHBOARD GRID (2x2 Layout)
# ============================================================================

row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

# ==========================
# CELL 1: RELEASE TIMING
# ==========================
with row1_col1:
    st.markdown("### 📅 Success Rate by Month")
    
    graph_data = filtered_df.copy()

    if len(graph_data) > 0:
        graph_data['release_month'] = pd.to_datetime(graph_data['Release date'], errors='coerce').dt.month
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        monthly_analysis = graph_data.groupby('release_month').agg({
            'Steam Score': lambda x: ((x >= 0.7).sum() / len(x) * 100) if len(x) > 0 else 0,
            'Name': 'count'
        }).reset_index()
        
        monthly_analysis['release_month'] = monthly_analysis['release_month'].astype(int)
        monthly_analysis = monthly_analysis[monthly_analysis['Name'] >= 100]

        if len(monthly_analysis) > 0:
            avg_success = monthly_analysis['Steam Score'].mean()
            colors = [COLOR_MAIN if rate < avg_success else COLOR_HIGHLIGHT for rate in monthly_analysis['Steam Score']]
            opacities = [0.95 if rate < avg_success else 0.65 for rate in monthly_analysis['Steam Score']]

            fig1 = go.Figure()
            fig1.add_trace(go.Bar(
                x=[month_names[m-1] for m in monthly_analysis['release_month']],
                y=monthly_analysis['Steam Score'],
                marker=dict(color=colors, opacity=opacities, line=dict(color='white', width=1)),
                text=[f"{rate:.0f}%" for rate in monthly_analysis['Steam Score']],
                textposition='outside',
                textfont=dict(size=9, color='white'),
                customdata=monthly_analysis['Name']
            ))
            fig1.add_hline(y=avg_success, line_dash="dash", line_color="#4ECDC4", line_width=2,
                          annotation_text=f"Avg: {avg_success:.1f}%", annotation_font=dict(color='#4ECDC4', size=10))

            fig1.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=300, # Dashboard Compact Height
                paper_bgcolor=COLOR_BG,
                plot_bgcolor=COLOR_BG,
                font=dict(color=COLOR_MAIN),
                showlegend=False,
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title="Success %")
            )
            st.plotly_chart(fig1, use_container_width=True)

            # Insights row below chart
            best_month = monthly_analysis.loc[monthly_analysis['Steam Score'].idxmax()]
            worst_month = monthly_analysis.loc[monthly_analysis['Steam Score'].idxmin()]
            
            ic1, ic2 = st.columns(2)
            with ic1:
                st.markdown(f"""<div class="insight-card"><h4>✅ Best: {month_names[int(best_month['release_month'])-1]}</h4>
                {best_month['Steam Score']:.1f}% success rate</div>""", unsafe_allow_html=True)
            with ic2:
                st.markdown(f"""<div class="insight-card"><h4>❌ Worst: {month_names[int(worst_month['release_month'])-1]}</h4>
                {worst_month['Steam Score']:.1f}% success rate</div>""", unsafe_allow_html=True)
        else:
            st.info("Need 100+ games per month for data.")
    else:
        st.warning("No data found.")

# ==========================
# CELL 2: PRICE BANDS
# ==========================
with row1_col2:
    st.markdown("### 💰 Success by Price Band")
    
    graph_data2 = filtered_df.copy()
    price_bins = [-1, 0, 5, 10, 20, 30, np.inf]
    price_labels = ['Free', '$0-5', '$5-10', '$10-20', '$20-30', '$30+']

    graph_data2['price_band'] = pd.cut(graph_data2['Price'], bins=price_bins, labels=price_labels, include_lowest=True)

    if len(graph_data2) > 0:
        fig2 = go.Figure()
        colors = ['#00A8CC', '#F9D923', '#F96D00', '#EA5455', '#2D4059', '#B8B8D1']
        
        for i, price_band in enumerate(price_labels):
            band_data = graph_data2[graph_data2['price_band'] == price_band]['Steam Score']
            if len(band_data) > 0:
                if len(band_data) > 1000: band_data = band_data.sample(1000, random_state=42)
                fig2.add_trace(go.Box(y=band_data, name=str(price_band), marker_color=colors[i], showlegend=False))
        
        fig2.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=300, # Dashboard Compact Height
            paper_bgcolor=COLOR_BG,
            plot_bgcolor=COLOR_BG,
            font=dict(color=COLOR_MAIN),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)', range=[0, 1.05], title="Score (0-1)"),
            xaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Insights row below chart
        summary = graph_data2.groupby('price_band')['Steam Score'].agg(['median', 'count']).reset_index().sort_values('median', ascending=False)
        top3 = summary.head(3)
        
        ic_cols = st.columns(3)
        for idx, (_, row) in enumerate(top3.iterrows()):
            with ic_cols[idx]:
                medal = ['🥇', '🥈', '🥉'][idx]
                st.markdown(f"""<div class="insight-card" style="text-align: center;">
                    <h4>{medal} {row['price_band']}</h4>Med: <strong>{row['median']:.2f}</strong></div>""", unsafe_allow_html=True)
    else:
        st.warning("No data found.")

# ==========================
# CELL 3: GENRE SUCCESS
# ==========================
with row2_col1:
    st.markdown("### 🎯 Genre Profitability")
    
    graph_data3 = filtered_df.copy()
    if len(graph_data3) > 0:
        lollipop = graph_data3[graph_data3['primary_genre'].notna()].groupby('primary_genre').agg(
            success_rate=('Steam Score', lambda x: (x >= 0.7).mean() * 100),
            profit=('EstimatedProfit', 'median'),
            count=('Name', 'count')
        ).reset_index()
        
        lollipop = lollipop[lollipop['count'] >= 50].sort_values('success_rate', ascending=True)
        
        if len(lollipop) > 0:
            fig3 = go.Figure()
            for idx, row in lollipop.iterrows():
                fig3.add_trace(go.Scatter(x=[0, row['success_rate']], y=[row['primary_genre'], row['primary_genre']],
                                         mode='lines', line=dict(color='rgba(128,128,128,0.5)', width=1), showlegend=False, hoverinfo='skip'))
            
            fig3.add_trace(go.Scatter(
                x=lollipop['success_rate'], y=lollipop['primary_genre'], mode='markers',
                marker=dict(size=14, color=lollipop['profit'], colorscale='Viridis', line=dict(color=COLOR_MAIN, width=1), showscale=True,
                           colorbar=dict(thickness=10, len=0.8, title="Profit")),
                customdata=lollipop['count'], showlegend=False
            ))
            
            fig3.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=350, # Dashboard Compact Height
                paper_bgcolor=COLOR_BG,
                plot_bgcolor=COLOR_BG,
                font=dict(color=COLOR_MAIN),
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)', title="Success Rate %"),
                yaxis=dict(dtick=1)
            )
            st.plotly_chart(fig3, use_container_width=True)
            
            top_genre = lollipop.iloc[-1]
            st.markdown(f"""<div class="insight-card">
                <h4>🏆 Top Genre: {top_genre['primary_genre']}</h4>
                {top_genre['success_rate']:.1f}% Success Rate | Median Profit: ${top_genre['profit']:,.0f}
            </div>""", unsafe_allow_html=True)
        else:
            st.info("Need 50+ games per genre.")
    else:
        st.warning("No data.")
# ==========================
# CELL 4: PLATFORM LAUNCH (FIXED)
# ==========================
with row2_col2:
    st.markdown("### 🎮 Ecosystem Impact")
    
    if len(vgsales_df) > 0:
        df_clean = vgsales_df.copy()
        
        # 1. Identify multi-platform games
        multi_platform_games = df_clean.groupby('Name').agg(
            Min_Year=('Year', 'min'), 
            Release_Span=('Year', lambda x: x.max() - x.min()),
            Total_Sales=('Global_Sales', 'sum'), 
            Platform_Count=('Platform', 'nunique')
        ).reset_index()
        
        multi_platform_games['Strategy'] = multi_platform_games.apply(
            lambda row: 'Instant' if row['Release_Span'] == 0 else 'Gradual', axis=1
        )
        
        # 2. Filter for launch year data
        launch_data = df_clean.merge(
            multi_platform_games[['Name', 'Min_Year', 'Strategy', 'Total_Sales']], 
            on="Name", 
            how="inner"
        )
        launch_data = launch_data[launch_data['Year'] == launch_data['Min_Year']]
        
        # 3. Aggregate launch platforms (FIXED HERE: Added rename)
        first_platforms = launch_data.groupby('Name')['Platform'].apply(
            lambda x: sorted(list(set(x)))
        ).reset_index().rename(columns={'Platform': 'Launch_Platforms'})
        
        def classify_launch(platforms):
            if any(p in platforms for p in ['PS3', 'X360']): return 'Gen 7 (PS3/X360)'
            elif any(p in platforms for p in ['PS2', 'XB', 'GC']): return 'Gen 6 (PS2/XB)'
            elif platforms == ['PC']: return 'PC Exclusive'
            elif 'Wii' in platforms: return 'Wii'
            elif any(p in platforms for p in ['DS', 'PSP', 'GBA']): return 'Handheld'
            else: return 'Other'

        first_platforms['Launch_Type'] = first_platforms['Launch_Platforms'].apply(classify_launch)
        
        # Merge back
        platform_analysis = multi_platform_games.merge(first_platforms, on="Name", how="left")
        gradual_df = platform_analysis[platform_analysis['Strategy'] == 'Gradual'].copy()

        if len(gradual_df) > 0:
            fig4 = go.Figure()
            # Sort by median sales for cleaner look
            sorted_types = gradual_df.groupby('Launch_Type')['Total_Sales'].median().sort_values(ascending=False).index
            
            for lt in sorted_types:
                subset = gradual_df[gradual_df['Launch_Type'] == lt]
                fig4.add_trace(go.Violin(
                    y=subset['Total_Sales'], 
                    x=[lt] * len(subset), 
                    name=lt, 
                    box_visible=True, 
                    line=dict(color='white'), 
                    fillcolor='rgba(255,255,255,0.1)', 
                    opacity=0.7
                ))

            fig4.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                yaxis=dict(type='log', gridcolor='rgba(255,255,255,0.1)', title="Sales (Log)"),
                showlegend=False
            )
            st.plotly_chart(fig4, use_container_width=True)
            
            # Text stats
            best_plat = gradual_df.groupby("Launch_Type")['Total_Sales'].median().idxmax()
            best_val = gradual_df.groupby("Launch_Type")['Total_Sales'].median().max()
            
            st.markdown(f"""<div class="insight-card">
                <h4>🚀 Best Launch Pad: {best_plat}</h4>
                Median Revenue: {best_val:.1f}M (Gradual Release Strategy)
            </div>""", unsafe_allow_html=True)
        else:
            st.info("No gradual release data.")
    else:
        st.warning("VG Sales data unavailable.")