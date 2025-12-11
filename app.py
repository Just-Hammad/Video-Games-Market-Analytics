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
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
<style>
    /* Main background and text */
    .stApp {
        background: linear-gradient(135deg, #232946 0%, #121629 100%);
        color: #F4F4F6;
    }
    h1 {
        background: linear-gradient(90deg, #F9D923, #00A8CC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900 !important;
        font-size: 3rem !important;
        text-align: center;
        padding: 1.2rem 0;
        letter-spacing: 2px;
    }
    h2 {
        color: #00A8CC !important;
        font-weight: 800 !important;
        border-left: 4px solid #F9D923;
        padding-left: 1rem;
        margin-top: 2rem !important;
    }
    .metric-card {
        background: linear-gradient(145deg, rgba(0,168,204,0.12), rgba(249,217,35,0.08));
        border-radius: 18px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 6px 24px rgba(0,0,0,0.18);
        text-align: center;
    }
    .metric-value {
        font-size: 2.7rem;
        font-weight: 900;
        background: linear-gradient(90deg, #F9D923, #00A8CC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-label {
        color: #B8B8D1;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 0.5rem;
    }
    .insight-card {
        background: linear-gradient(145deg, rgba(0,168,204,0.10), rgba(249,217,35,0.05));
        border-radius: 14px;
        padding: 1rem 1.5rem;
        border-left: 4px solid #00A8CC;
        margin: 1rem 0;
    }
    .insight-card h4 {
        color: #F9D923;
        margin: 0 0 0.5rem 0;
    }
    .insight-card p {
        color: #F4F4F6;
        margin: 0;
    }
    .stRadio > div {
        background: rgba(0,168,204,0.08);
        border-radius: 12px;
        padding: 0.5rem 1rem;
    }
    .stPlotlyChart {
        background: rgba(0,168,204,0.04);
        border-radius: 18px;
        padding: 1rem;
        border: 1px solid rgba(0,168,204,0.08);
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #F9D923, transparent);
        margin: 2rem 0;
    }
    .stCaption {
        color: #B8B8D1 !important;
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
        st.error(f"Error loading mainDS.parquet: {e}")
        return pd.DataFrame()

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
        st.error(f"Error loading vgsales_cleaned.csv: {e}")
        return pd.DataFrame()
    return df

vgsales_df = load_vgsales()


# ============================================================================
# HEADER
# ============================================================================
st.markdown("<h1>🎮 Indie Game Analytics Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888; margin-bottom: 2rem;'>Discover insights for indie game success on Steam</p>", unsafe_allow_html=True)

# ============================================================================
# METRICS AND FILTERS
# ============================================================================
col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])

with col1:
    game_filter = st.radio(
        "📊 Filter Dataset:",
        options=['All Games', 'Indie Only', 'Non-Indie Only'],
        horizontal=True,
        index=0
    )

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(df):,}</div>
        <div class="metric-label">Total Games</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    indie_count = len(df[df['indie'] == True])
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{indie_count:,}</div>
        <div class="metric-label">Indie Games</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    indie_pct = (indie_count / len(df) * 100) if len(df) > 0 else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{indie_pct:.1f}%</div>
        <div class="metric-label">Market Share</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Apply filter
if game_filter == 'Indie Only':
    filtered_df = df[df['indie'] == True].copy()
elif game_filter == 'Non-Indie Only':
    filtered_df = df[df['indie'] == False].copy()
else:
    filtered_df = df.copy()

# Get filter label for display
filter_label = "All Games" if game_filter == 'All Games' else ("Indie Games" if game_filter == 'Indie Only' else "Non-Indie Games")
# ============================================================================
# GRAPH 1: RELEASE TIMING SUCCESS RATE
# ============================================================================
st.markdown("## 📅 Avoid Summer Releases - Success Rate by Month")

graph_data = filtered_df.copy()

if len(graph_data) > 0:

    # Calculate release month
    graph_data['release_month'] = pd.to_datetime(
        graph_data['Release date'], errors='coerce'
    ).dt.month

    # Month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Monthly analysis
    monthly_analysis = graph_data.groupby('release_month').agg({
        'Steam Score': lambda x: ((x >= 0.7).sum() / len(x) * 100) if len(x) > 0 else 0,
        'Name': 'count'
    }).reset_index()

    # Force month to integer (fix float → index error)
    monthly_analysis['release_month'] = monthly_analysis['release_month'].astype(int)

    # Filter months with at least 100 releases
    monthly_analysis = monthly_analysis[monthly_analysis['Name'] >= 100]

    if len(monthly_analysis) > 0:

        # Average success rate
        avg_success = monthly_analysis['Steam Score'].mean()

        # Auto-coloring:
        # Blue = below average, Orange = above average
        colors = [
            COLOR_MAIN if rate < avg_success else COLOR_HIGHLIGHT
            for rate in monthly_analysis['Steam Score']
        ]
        opacities = [
            0.95 if rate < avg_success else 0.65
            for rate in monthly_analysis['Steam Score']
        ]

        # Create bar chart
        fig1 = go.Figure()

        fig1.add_trace(go.Bar(
            x=[month_names[m-1] for m in monthly_analysis['release_month']],
            y=monthly_analysis['Steam Score'],
            marker=dict(
                color=colors,
                opacity=opacities,
                line=dict(color='white', width=1.5)
            ),
            text=[f"{rate:.1f}%" for rate in monthly_analysis['Steam Score']],
            textposition='outside',
            textfont=dict(size=10, color='white', family='Arial Black'),
            hovertemplate='<b>%{x}</b><br>' +
                          'Success Rate: %{y:.1f}%<br>' +
                          'Games: %{customdata:,}<extra></extra>',
            customdata=monthly_analysis['Name']
        ))

        # Add average line
        fig1.add_hline(
            y=avg_success,
            line_dash="dash",
            line_color="#4ECDC4",
            line_width=2,
            annotation_text=f"Average: {avg_success:.1f}%",
            annotation_position="right",
            annotation_font=dict(color='#4ECDC4', size=11)
        )

        fig1.update_layout(
            xaxis=dict(
                title="Release Month",
                tickfont=dict(color=COLOR_MAIN, size=12),
                gridcolor=COLOR_BG
            ),
            yaxis=dict(
                title="Success Rate (%)",
                gridcolor=COLOR_BG,
                tickfont=dict(color=COLOR_MAIN)
            ),
            height=500,
            paper_bgcolor=COLOR_BG,
            plot_bgcolor=COLOR_BG,
            font=dict(color=COLOR_MAIN),
            showlegend=False
        )

        # Render chart
        st.plotly_chart(fig1, use_container_width=True)

        # Best & worst months
        best_month = monthly_analysis.loc[monthly_analysis['Steam Score'].idxmax()]
        worst_month = monthly_analysis.loc[monthly_analysis['Steam Score'].idxmin()]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="insight-card">
                <h4>✅ Best Month: {month_names[int(best_month['release_month'])-1]}</h4>
                <p><strong>{best_month['Steam Score']:.1f}%</strong> success rate  
                with <strong>{int(best_month['Name']):,}</strong> games</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="insight-card">
                <h4>❌ Worst Month: {month_names[int(worst_month['release_month'])-1]}</h4>
                <p><strong>{worst_month['Steam Score']:.1f}%</strong> success rate  
                with <strong>{int(worst_month['Name']):,}</strong> games</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="insight-card">
            <h4>💡 Strategic Insight</h4>
            <p>Bars automatically turn <strong>dark red</strong> when their success rate falls
            <strong>below the monthly average</strong>.  
            Based on <strong>{len(graph_data):,}</strong> {filter_label.lower()} 
            with at least 100 releases per month.</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("Not enough data: Need at least 100 games per month.")

else:
    st.warning(f"No data found for {filter_label}")

st.markdown("---")

# ============================================================================
# GRAPH 2: BOXPLOT
# ============================================================================
st.markdown("## 💰 Price Band Success Distribution")

graph_data2 = filtered_df.copy()

price_bins = [-1, 0, 5, 10, 20, 30, np.inf]
price_labels = ['Free', '$0-5', '$5-10', '$10-20', '$20-30', '$30+']

graph_data2['price_band'] = pd.cut(
    graph_data2['Price'],
    bins=price_bins,
    labels=price_labels,
    include_lowest=True
)

if len(graph_data2) > 0:
    fig2 = go.Figure()
    
    # Color gradient for price bands
    # Colorblind-friendly palette
    colors = ['#00A8CC', '#F9D923', '#F96D00', '#EA5455', '#2D4059', '#B8B8D1']
    
    for i, price_band in enumerate(price_labels):
        band_data = graph_data2[graph_data2['price_band'] == price_band]['Steam Score']
        if len(band_data) > 0:
            if len(band_data) > 1000:
                band_data = band_data.sample(1000, random_state=42)
            
            fig2.add_trace(go.Box(
                y=band_data,
                name=str(price_band),
                marker_color=colors[i],
                line=dict(color=colors[i], width=2),
                fillcolor=colors[i],
                opacity=0.8,
                boxpoints=False
            ))
    
    fig2.update_layout(
        xaxis_title="Price Band",
        yaxis_title="Steam Score (0-1)",
        height=500,
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        font=dict(color=COLOR_MAIN),
        yaxis=dict(
            gridcolor=COLOR_BG,
            range=[0, 1.05]
        ),
        xaxis=dict(gridcolor=COLOR_BG),
        showlegend=False
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Calculate median summary
    summary = (
        graph_data2.groupby('price_band')['Steam Score']
        .agg(['median', 'count'])
        .reset_index()
        .sort_values('median', ascending=False)
    )
    
    top3 = summary.head(3)
    
    cols = st.columns(3)
    for idx, (_, row) in enumerate(top3.iterrows()):
        with cols[idx]:
            medal = ['🥇', '🥈', '🥉'][idx]
            st.markdown(f"""
            <div class="insight-card" style="text-align: center;">
                <h4>{medal} {row['price_band']}</h4>
                <p>Median Score: <strong>{row['median']:.2f}</strong><br>({int(row['count']):,} games)</p>
            </div>
            """, unsafe_allow_html=True)
else:
    st.warning(f"No data found for {filter_label}")

st.markdown("---")

# ============================================================================
# GRAPH 3: LOLLIPOP CHART - GENRE SUCCESS + PROFIT
# ============================================================================
st.markdown("## 🎯 Genre Success Rates & Profitability")

graph_data3 = filtered_df.copy()

if len(graph_data3) > 0:
    # Prepare lollipop data
    lollipop = (
        graph_data3[graph_data3['primary_genre'].notna()]
        .groupby('primary_genre')
        .agg(
            success_rate=('Steam Score', lambda x: (x >= 0.7).mean() * 100),
            profit=('EstimatedProfit', 'median'),
            count=('Name', 'count')
        )
        .reset_index()
    )
    
    # Filter genres with at least 50 games
    lollipop = lollipop[lollipop['count'] >= 50]
    
    # Sort by success rate for better visualization
    lollipop = lollipop.sort_values('success_rate', ascending=True)
    
    if len(lollipop) > 0:
        fig3 = go.Figure()
        
        # Add stem lines (from 0 to success rate)
        for idx, row in lollipop.iterrows():
            fig3.add_trace(go.Scatter(
                x=[0, row['success_rate']],
                y=[row['primary_genre'], row['primary_genre']],
                mode='lines',
                line=dict(color='rgba(128,128,128,0.5)', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add dots colored by profit
            fig3.add_trace(go.Scatter(
                x=lollipop['success_rate'],
                y=lollipop['primary_genre'],
                mode='markers',
                marker=dict(
                    size=22,  # Increased size for better visibility
                    color=lollipop['profit'],
                    colorscale='Viridis',
                    line=dict(color=COLOR_MAIN, width=2),
                    colorbar=dict(
                        title=dict(text="Median<br>Profit ($)", side="right"),
                        tickfont=dict(color=COLOR_MAIN),
                        tickformat='$,.0f'
                    ),
                    showscale=True
                ),
                hovertemplate='<b>%{y}</b><br>' +
                             'Success Rate: %{x:.1f}%<br>' +
                             'Median Profit: $%{marker.color:,.0f}<br>' +
                             'Games: %{customdata:,}<extra></extra>',
                customdata=lollipop['count'],
                showlegend=False
            ))
        
        fig3.update_layout(
            xaxis=dict(
                title="Success Rate (% with Steam Score ≥ 0.7)",
                gridcolor=COLOR_BG,
                tickfont=dict(color=COLOR_MAIN),
                range=[0, max(lollipop['success_rate']) * 1.05]
            ),
            yaxis=dict(
                title="",
                tickfont=dict(color=COLOR_MAIN, size=12),
                gridcolor=COLOR_BG
            ),
            height=max(600, len(lollipop) * 25),
            paper_bgcolor=COLOR_BG,
            plot_bgcolor=COLOR_BG,
            font=dict(color=COLOR_MAIN),
            margin=dict(l=150, r=20, t=40, b=60)
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # Top 3 performers
        top3 = lollipop.nlargest(3, 'success_rate')
        
        cols = st.columns(3)
        for idx, (_, row) in enumerate(top3.iterrows()):
            with cols[idx]:
                medal = ['🥇', '🥈', '🥉'][idx]
                st.markdown(f"""
                <div class="insight-card" style="text-align: center;">
                    <h4>{medal} {row['primary_genre']}</h4>
                    <p><strong>{row['success_rate']:.1f}%</strong> Success Rate<br>
                    Median Profit: <strong>${row['profit']:,.0f}</strong><br>
                    ({int(row['count']):,} games)</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="insight-card">
            <h4>💡 Reading the Chart</h4>
            <p>Each dot represents a genre's success rate. Dot color indicates profitability (cooler colors = lower profit, warmer = higher profit).
            Analyzing <strong>{len(lollipop)}</strong> genres from <strong>{len(graph_data3):,}</strong> {filter_label.lower()}.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info(f"Not enough genre data: Need genres with 50+ games in {filter_label.lower()}")
else:
    st.warning(f"No data found for {filter_label}")

st.markdown("---")
# ============================================================================
# GRAPH X: PLATFORM LAUNCH STRATEGY ANALYSIS (VGCHARTZ)
# ============================================================================
st.markdown("## 🎮 Platform Launch Ecosystem Impact on Sales")

if len(vgsales_df) > 0:

    # Prepare copy
    df_clean = vgsales_df.copy()

    # Step 1: Extract earliest launch year per game
    multi_platform_games = (
        df_clean.groupby('Name')
        .agg(
            Min_Year=('Year', 'min'),
            Release_Span=('Year', lambda x: x.max() - x.min()),
            Total_Sales=('Global_Sales', 'sum'),
            Platform_Count=('Platform', 'nunique')
        )
        .reset_index()
    )

    # Strategy column (Instant vs Gradual)
    multi_platform_games['Strategy'] = multi_platform_games.apply(
        lambda row: 'Instant' if row['Release_Span'] == 0 else 'Gradual',
        axis=1
    )

    # Step 2: Merge years back to full DF
    launch_data = df_clean.merge(
        multi_platform_games[['Name', 'Min_Year', 'Strategy', 'Total_Sales']],
        on="Name",
        how="inner"
    )

    # Only rows on launch year
    launch_data = launch_data[launch_data['Year'] == launch_data['Min_Year']]

    # Step 3: Collect first-launch platforms per game
    first_platforms = (
        launch_data.groupby('Name')['Platform']
        .apply(lambda x: sorted(list(set(x))))
        .reset_index()
        .rename(columns={'Platform': 'Launch_Platforms'})
    )

    # Classification function
    def classify_launch(platforms):
        if any(p in platforms for p in ['PS3', 'X360']):
            return 'Gen 7 Console (PS3/X360)'
        elif any(p in platforms for p in ['PS2', 'XB', 'GC']):
            return 'Gen 6 Console (PS2/XB/GC)'
        elif platforms == ['PC']:
            return 'PC Exclusive Start'
        elif 'Wii' in platforms:
            return 'Wii'
        elif any(p in platforms for p in ['DS', 'PSP', 'GBA']):
            return 'Handheld'
        else:
            return 'Other/Mixed'

    first_platforms['Launch_Type'] = first_platforms['Launch_Platforms'].apply(classify_launch)

    # Merge back
    platform_analysis = multi_platform_games.merge(first_platforms, on="Name", how="left")

    # Only gradual releases
    gradual_df = platform_analysis[platform_analysis['Strategy'] == 'Gradual'].copy()

    if len(gradual_df) > 0:

        st.markdown("### 📈 Revenue Distribution by Launch Ecosystem (Gradual Releases Only)")

        # Prepare Plotly violin chart
        fig = go.Figure()

        for lt in gradual_df['Launch_Type'].unique():
            subset = gradual_df[gradual_df['Launch_Type'] == lt]
            fig.add_trace(go.Violin(
                y=subset['Total_Sales'],
                x=[lt] * len(subset),
                name=lt,
                box_visible=True,
                meanline_visible=True,
                spanmode="hard",
                line=dict(color='white'),
                fillcolor='rgba(255,255,255,0.1)',
                opacity=0.7
            ))

        fig.update_layout(
            height=600,
            title="Revenue by Launch Ecosystem",
            xaxis_title="Launch Type",
            yaxis_title="Total Global Sales (Millions, Log Scale)",
            yaxis=dict(type='log', gridcolor='rgba(255,255,255,0.1)'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # Insights
        summary = (
            gradual_df.groupby("Launch_Type")['Total_Sales']
            .median()
            .sort_values(ascending=False)
        )

        st.markdown("### 🏆 Median Revenue by Launch Ecosystem")
        
        for lt, med in summary.items():
            st.markdown(
                f"- **{lt}** → Median Revenue: **{med:.1f}M**"
            )

    else:
        st.info("No gradual multi-platform titles found in vgsales dataset.")

else:
    st.warning("VG sales dataset could not be loaded.")

# ============================================================================
# GRAPH X: RELEASE STRATEGY — GRADUAL VS SIMULTANEOUS (VGCHARTZ)
# ============================================================================
st.markdown("## 🕒 Release Strategy: Gradual vs Simultaneous Launches")

if len(vgsales_df) > 0:

    df_clean = vgsales_df.copy()

    # ----------------------------------------------------
    # 1. Aggregate platform-level rows → per-game stats
    # ----------------------------------------------------
    game_stats = df_clean.groupby('Name').agg(
        Min_Year=('Year', 'min'),
        Max_Year=('Year', 'max'),
        Platform_Count=('Platform', 'nunique'),
        Total_Sales=('Global_Sales', 'sum')
    ).reset_index()

    # Compute release span
    game_stats['Release_Span'] = game_stats['Max_Year'] - game_stats['Min_Year']

    # Multi-platform only
    multi_platform_games = game_stats[game_stats['Platform_Count'] > 1].copy()

    # Classify strategy
    multi_platform_games['Strategy'] = multi_platform_games['Release_Span'].apply(
        lambda s: 'Simultaneous' if s == 0 else 'Gradual'
    )

    # ----------------------------------------------------
    # 2. Plotly Violin + Strip equivalent
    # ----------------------------------------------------
    st.markdown("### 🎻 Global Sales Distribution (Log Scale)")

    fig = go.Figure()

    # VIOLIN LAYERS
    fig.add_trace(go.Violin(
        x=multi_platform_games['Strategy'],
        y=multi_platform_games['Total_Sales'],
        name="Distribution",
        box_visible=True,
        meanline_visible=True,
        points=False,
        line=dict(color='white'),
        fillcolor='rgba(70,130,180,0.3)'
    ))

    # STRIP POINTS (limit to 1000 for readability)
    strip_sample = (
        multi_platform_games.sample(1000, random_state=42)
        if len(multi_platform_games) > 1000
        else multi_platform_games
    )

    fig.add_trace(go.Scatter(
        x=strip_sample['Strategy'],
        y=strip_sample['Total_Sales'],
        mode='markers',
        marker=dict(color='white', size=4, opacity=0.3),
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>Sales: %{y:.2f}M<extra></extra>"
    ))

    fig.update_layout(
        height=600,
        yaxis=dict(type='log', title="Total Sales (Millions, Log Scale)",
                   gridcolor='rgba(255,255,255,0.1)'),
        xaxis_title="Release Strategy",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------
    # 3. Strategy Metrics Table
    # ----------------------------------------------------
    st.markdown("### 📊 Strategy Performance Metrics")

    strategy_metrics = (
        multi_platform_games
        .groupby('Strategy')['Total_Sales']
        .agg(['mean', 'median', 'count'])
        .sort_values('median', ascending=False)
    )

    st.dataframe(strategy_metrics.style.format({
        "mean": "{:.2f}",
        "median": "{:.2f}",
        "count": "{:,.0f}"
    }))

    # ----------------------------------------------------
    # 4. Quick counts for strategies
    # ----------------------------------------------------
    st.markdown("### 📦 Strategy Distribution")

    strategy_counts = multi_platform_games['Strategy'].value_counts()

    for strategy, count in strategy_counts.items():
        st.markdown(f"- **{strategy}** → {count:,} games")

else:
    st.warning("Could not load vgsales_cleaned.csv — ensure the file exists.")


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <p style="font-size: 0.9rem;">
        📊 <strong>Data Visualization Project</strong> | Analyzing Steam Game Market Trends
    </p>
    <p style="font-size: 0.8rem; margin-top: 1rem;">
        Success Rate = % of games with Steam Score ≥ 0.7 | Profit estimates based on estimated owners × price × 70%
    </p>
</div>
""", unsafe_allow_html=True)