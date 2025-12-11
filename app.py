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
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f0f23 100%);
    }
    
    /* Headers */
    h1 {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800 !important;
        font-size: 2.8rem !important;
        text-align: center;
        padding: 1rem 0;
    }
    
    h2 {
        color: #4ECDC4 !important;
        font-weight: 700 !important;
        border-left: 4px solid #FF6B6B;
        padding-left: 1rem;
        margin-top: 2rem !important;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF6B6B, #FFD93D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        color: #a0a0a0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }
    
    /* Insight cards */
    .insight-card {
        background: linear-gradient(145deg, rgba(78, 205, 196, 0.15), rgba(78, 205, 196, 0.05));
        border-radius: 12px;
        padding: 1rem 1.5rem;
        border-left: 4px solid #4ECDC4;
        margin: 1rem 0;
    }
    
    .insight-card h4 {
        color: #4ECDC4;
        margin: 0 0 0.5rem 0;
    }
    
    .insight-card p {
        color: #e0e0e0;
        margin: 0;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 0.5rem 1rem;
    }
    
    /* Plotly charts container */
    .stPlotlyChart {
        background: rgba(255,255,255,0.02);
        border-radius: 16px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        margin: 2rem 0;
    }
    
    /* Caption styling */
    .stCaption {
        color: #888 !important;
    }
</style>
""", unsafe_allow_html=True)

# Color scheme
COLOR_MAIN = '#FF6B6B'
COLOR_PROFIT = '#FFD93D'
COLOR_HIGHLIGHT = '#4ECDC4'
COLOR_BG = '#1a1a2e'

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
        # Dark red = below average
        # Light red = above average
        colors = [
            '#B22222' if rate < avg_success else '#FF6B6B'
            for rate in monthly_analysis['Steam Score']
        ]

        opacities = [
            0.95 if rate < avg_success else 0.45
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
                tickfont=dict(color='white', size=11),
                gridcolor='rgba(255,255,255,0.05)'
            ),
            yaxis=dict(
                title="Success Rate (%)",
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='white')
            ),
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
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
    colors = ['#4ECDC4', '#45B7AA', '#3CA18E', '#338B72', '#2A7556', '#215F3A']
    
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
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            range=[0, 1.05]
        ),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
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
                size=14,
                color=lollipop['profit'],
                colorscale=[[0, '#FF6B6B'], [0.5, '#FFD93D'], [1, '#4ECDC4']],
                line=dict(color='white', width=2),
                colorbar=dict(
                    title=dict(text="Median<br>Profit ($)", side="right"),
                    tickfont=dict(color='white'),
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
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='white'),
                range=[0, max(lollipop['success_rate']) * 1.05]
            ),
            yaxis=dict(
                title="",
                tickfont=dict(color='white', size=10),
                gridcolor='rgba(255,255,255,0.05)'
            ),
            height=max(600, len(lollipop) * 25),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
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