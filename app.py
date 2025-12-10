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
# GRAPH 1: HEATMAP
# ============================================================================
st.markdown("## 📅 Release Timing & Pricing Analysis")

graph_data = filtered_df.copy()

if len(graph_data) > 0:
    # Create heatmap data
    playtime_heatmap = graph_data.groupby(['ReleaseMonth', 'price_bin'])['Average playtime forever'].mean().unstack()
    
    # Ensure all months 1-12 are present
    all_months = list(range(1, 13))
    playtime_heatmap = playtime_heatmap.reindex(all_months)
    
    # Price band order
    price_order = ['Free', '$0-5', '$5-10', '$10-20', '$20-30', '$30+']
    playtime_heatmap = playtime_heatmap.reindex(columns=price_order)
    
    # Create Plotly heatmap with dark theme
    fig1 = go.Figure(data=go.Heatmap(
        z=playtime_heatmap.values,
        x=playtime_heatmap.columns.tolist(),
        y=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        colorscale=[
            [0, '#1a1a2e'],
            [0.25, '#16213e'],
            [0.5, '#4ECDC4'],
            [0.75, '#FFD93D'],
            [1, '#FF6B6B']
        ],
        hovertemplate='<b>%{y}</b> | <b>%{x}</b><br>Avg Playtime: %{z:.1f} min<extra></extra>',
        colorbar=dict(
            title=dict(text="Playtime (min)", side="right"),
            tickfont=dict(color='white')
        ),
        zmin=0
    ))
    
    fig1.update_layout(
        xaxis_title="Price Band",
        yaxis_title="Release Month",
        yaxis=dict(autorange='reversed'),
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
    )
    
    st.plotly_chart(fig1, width="stretch")
    
    st.markdown(f"""
    <div class="insight-card">
        <h4>💡 Key Insight</h4>
        <p>Analyzing <strong>{len(graph_data):,}</strong> {filter_label.lower()}. Darker/warmer colors indicate higher average playtime. 
        Look for patterns in release timing and pricing that lead to better player engagement.</p>
    </div>
    """, unsafe_allow_html=True)
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
    
    st.plotly_chart(fig2, width="stretch")
    
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
# GRAPH 3: GENRE ANALYSIS
# ============================================================================
st.markdown("## 🎯 Genre Performance Analysis")

graph_data3 = filtered_df.copy()

if len(graph_data3) > 0:
    genre_stats = (
        graph_data3[graph_data3['primary_genre'].notna()]
        .groupby('primary_genre')
        .agg(
            avg_success=('Steam Score', lambda x: (x >= 0.7).mean() * 100),
            count=('Name', 'count'),
            EstimatedProfit=('EstimatedProfit', 'mean')
        )
        .reset_index()
    )
    
    genre_stats = genre_stats[genre_stats['count'] >= 100]
    genre_stats = genre_stats.sort_values('avg_success', ascending=False)
    
    if len(genre_stats) > 0:
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Bar: Success Rate with gradient effect
        fig3.add_trace(
            go.Bar(
                x=genre_stats['primary_genre'],
                y=genre_stats['avg_success'],
                name='Success Rate (%)',
                marker=dict(
                    color=genre_stats['avg_success'],
                    colorscale=[[0, '#FF6B6B'], [0.5, '#FFD93D'], [1, '#4ECDC4']],
                    line=dict(color='rgba(255,255,255,0.3)', width=1)
                ),
                opacity=0.9,
                hovertemplate='<b>%{x}</b><br>Success Rate: %{y:.1f}%<br>Games: %{customdata}<extra></extra>',
                customdata=genre_stats['count']
            ),
            secondary_y=False
        )
        
        # Line: Profit
        fig3.add_trace(
            go.Scatter(
                x=genre_stats['primary_genre'],
                y=genre_stats['EstimatedProfit'],
                name='Avg Profit ($)',
                mode='lines+markers',
                marker=dict(color='#FFD93D', size=10, line=dict(color='white', width=2)),
                line=dict(color='#FFD93D', width=3),
                hovertemplate='<b>%{x}</b><br>Avg Profit: $%{y:,.0f}<extra></extra>'
            ),
            secondary_y=True
        )
        
        fig3.update_layout(
            xaxis=dict(
                tickangle=45,
                tickfont=dict(size=11, color='white'),
                gridcolor='rgba(255,255,255,0.05)'
            ),
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                x=0.01, y=0.99,
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='rgba(255,255,255,0.1)',
                borderwidth=1
            )
        )
        
        fig3.update_yaxes(
            title_text="Success Rate (%)", 
            secondary_y=False, 
            showgrid=True, 
            gridcolor='rgba(255,255,255,0.1)'
        )
        fig3.update_yaxes(
            title_text="Average Profit ($)", 
            showgrid=False
        )
        
        st.plotly_chart(fig3, width="stretch")
        
        # Top performers
        top_genre = genre_stats.iloc[0]
        st.markdown(f"""
        <div class="insight-card">
            <h4>🏆 Top Performer: {top_genre['primary_genre']}</h4>
            <p><strong>{top_genre['avg_success']:.1f}%</strong> success rate with <strong>{int(top_genre['count']):,}</strong> games 
            | Average profit: <strong>${top_genre['EstimatedProfit']:,.0f}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Not enough data: Need genres with 100+ indie games")
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