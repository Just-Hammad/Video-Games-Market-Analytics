import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Steam Games Analytics Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #A23B72;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }
    .insight-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 6px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.8rem;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    # To use your actual data, replace this function with:
    # return pd.read_csv('cleaned_data.csv')
    
    np.random.seed(42)
    n_games = 8000
    release_probs = np.array([0.03]*5 + [0.05]*5 + [0.08]*5)
    release_probs = release_probs / release_probs.sum()
    release_years = np.random.choice(range(2010, 2025), n_games, p=release_probs)

    
    is_indie = np.random.choice([True, False], n_games, p=[0.72, 0.28])
    
    df = pd.DataFrame({
        'AppID': range(n_games),
        'Name': [f'Game_{i}' for i in range(n_games)],
        'Price': np.random.choice([0, 4.99, 9.99, 14.99, 19.99, 29.99, 39.99, 49.99, 59.99], n_games),
        'Estimated owners': np.random.choice([2500, 15000, 75000, 250000, 750000, 2500000, 7500000], n_games),
        'Steam Score': np.random.beta(7, 2, n_games),
        'Positive': np.random.randint(10, 15000, n_games),
        'Negative': np.random.randint(5, 3000, n_games),
        'Peak CCU': np.random.randint(5, 100000, n_games),
        'Achievements': np.random.choice([0, 8, 18, 35, 65, 120, 200], n_games),
        'DLC count': np.random.choice([0, 1, 2, 3, 4, 6, 12], n_games),
        'platform_count': np.random.choice([1, 2, 3], n_games, p=[0.35, 0.45, 0.2]),
        'indie': is_indie,
        'primary_genre': np.random.choice(['Action', 'Adventure', 'RPG', 'Strategy', 'Simulation', 
                                           'Indie', 'Casual', 'Sports', 'Racing'], n_games),
        'ReleaseYear': release_years,
        'release_month': np.random.randint(1, 13, n_games),
        'Average playtime forever': np.random.randint(30, 5000, n_games),
    })
    
    df['total_reviews'] = df['Positive'] + df['Negative']
    df['EstimatedProfit'] = df['Price'] * df['Estimated owners'] * 0.7
    df['is_free'] = df['Price'] == 0
    df['indie_status'] = df['indie'].map({True: 'Indie', False: 'Non-Indie'})
    df['has_dlc'] = df['DLC count'] > 0
    df['has_achievements'] = df['Achievements'] > 0
    
    return df

# Load data
df = load_data()

# Sidebar
st.sidebar.markdown("# 🎮 Filters & Controls")
st.sidebar.markdown("---")

# Year filter
year_range = st.sidebar.slider(
    "📅 Release Year Range",
    int(df['ReleaseYear'].min()),
    int(df['ReleaseYear'].max()),
    (2015, 2024)
)

# Game type filter
game_type = st.sidebar.multiselect(
    "🎯 Game Type",
    options=['Indie', 'Non-Indie'],
    default=['Indie', 'Non-Indie']
)

# Price filter
price_range = st.sidebar.slider(
    "💰 Price Range ($)",
    0.0,
    float(df['Price'].max()),
    (0.0, 60.0)
)

# Genre filter
selected_genres = st.sidebar.multiselect(
    "🎨 Genres",
    options=sorted(df['primary_genre'].unique()),
    default=sorted(df['primary_genre'].unique())
)

# Platform filter
platform_filter = st.sidebar.multiselect(
    "💻 Platform Count",
    options=[1, 2, 3],
    default=[1, 2, 3]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 About")
st.sidebar.info("""
This dashboard provides interactive analysis of Steam games market data.
Use the filters above to explore different segments and trends.
""")

# Apply filters
filtered_df = df[
    (df['ReleaseYear'] >= year_range[0]) &
    (df['ReleaseYear'] <= year_range[1]) &
    (df['indie_status'].isin(game_type)) &
    (df['Price'] >= price_range[0]) &
    (df['Price'] <= price_range[1]) &
    (df['primary_genre'].isin(selected_genres)) &
    (df['platform_count'].isin(platform_filter))
]

# Main header
st.markdown('<h1 class="main-header">🎮 Steam Games Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Interactive analysis of Steam gaming market trends and indie game success factors")
st.markdown("---")

# Key Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("📊 Total Games", f"{len(filtered_df):,}")
with col2:
    indie_pct = (filtered_df['indie'].sum() / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    st.metric("🎨 Indie %", f"{indie_pct:.1f}%")
with col3:
    avg_price = filtered_df['Price'].mean()
    st.metric("💵 Avg Price", f"${avg_price:.2f}")
with col4:
    avg_score = filtered_df['Steam Score'].mean()
    st.metric("⭐ Avg Score", f"{avg_score:.2%}")
with col5:
    median_profit = filtered_df['EstimatedProfit'].median()
    st.metric("💰 Median Profit", f"${median_profit:,.0f}")

st.markdown("---")

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Market Overview", 
    "💰 Pricing Strategy", 
    "🎯 Success Factors", 
    "🎨 Genre Analysis",
    "🔍 Deep Dive",
    "📊 Insights Summary"
])

# TAB 1: Market Overview
with tab1:
    st.markdown('<p class="sub-header">Market Structure & Trends</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Releases over time
        release_counts = filtered_df.groupby(['ReleaseYear', 'indie_status']).size().reset_index(name='count')
        fig1 = px.line(release_counts, x='ReleaseYear', y='count', color='indie_status',
                      title='📅 Game Releases Over Time',
                      labels={'count': 'Number of Games', 'ReleaseYear': 'Year'},
                      color_discrete_map={'Indie': '#FF6B6B', 'Non-Indie': '#4ECDC4'})
        fig1.update_traces(mode='lines+markers', line=dict(width=3), marker=dict(size=8))
        fig1.update_layout(hovermode='x unified', height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Market share
        yearly_data = filtered_df.groupby('ReleaseYear').agg({
            'indie': lambda x: (x.sum() / len(x) * 100)
        }).reset_index()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=yearly_data['ReleaseYear'], y=yearly_data['indie'],
                                 mode='lines+markers', name='Indie Market Share',
                                 line=dict(color='#FF6B6B', width=4),
                                 marker=dict(size=10),
                                 fill='tozeroy'))
        fig2.add_hline(y=70, line_dash="dash", line_color="red", 
                      annotation_text="70% Threshold")
        fig2.update_layout(title='📊 Indie Market Share Over Time',
                          xaxis_title='Year', yaxis_title='Market Share (%)',
                          height=400, hovermode='x')
        st.plotly_chart(fig2, use_container_width=True)
    
    # Platform distribution
    st.markdown("### 💻 Platform Distribution")
    platform_data = filtered_df.groupby(['platform_count', 'indie_status']).size().reset_index(name='count')
    fig3 = px.bar(platform_data, x='platform_count', y='count', color='indie_status',
                 title='Games by Platform Count',
                 labels={'platform_count': 'Number of Platforms', 'count': 'Number of Games'},
                 barmode='group',
                 color_discrete_map={'Indie': '#FF6B6B', 'Non-Indie': '#4ECDC4'})
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)
    
    # Additional insights
    col1, col2, col3 = st.columns(3)
    with col1:
        total_revenue = filtered_df['EstimatedProfit'].sum()
        st.metric("💵 Total Est. Revenue", f"${total_revenue/1e9:.2f}B")
    with col2:
        avg_owners = filtered_df['Estimated owners'].mean()
        st.metric("👥 Avg Owners", f"{avg_owners:,.0f}")
    with col3:
        avg_ccu = filtered_df['Peak CCU'].mean()
        st.metric("🔥 Avg Peak CCU", f"{avg_ccu:,.0f}")

# TAB 2: Pricing Strategy
with tab2:
    st.markdown('<p class="sub-header">Pricing Analysis & Revenue</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price distribution
        paid_games = filtered_df[filtered_df['is_free'] == False]
        fig4 = go.Figure()
        for status, color in [('Indie', '#FF6B6B'), ('Non-Indie', '#4ECDC4')]:
            data = paid_games[paid_games['indie_status'] == status]['Price']
            fig4.add_trace(go.Histogram(x=data, name=status, opacity=0.7,
                                       marker_color=color, nbinsx=30))
        fig4.update_layout(title='💵 Price Distribution (Paid Games)',
                          xaxis_title='Price (USD)', yaxis_title='Count',
                          barmode='overlay', height=400, hovermode='x')
        st.plotly_chart(fig4, use_container_width=True)
    
    with col2:
        # Success by price
        paid_games['price_bracket'] = pd.cut(paid_games['Price'],
                                             bins=[0, 5, 10, 15, 20, 30, 50, 100],
                                             labels=['$0-5', '$5-10', '$10-15', '$15-20', '$20-30', '$30-50', '$50+'])
        price_success = paid_games.groupby(['price_bracket', 'indie_status'])['Steam Score'].mean().reset_index()
        fig5 = px.bar(price_success, x='price_bracket', y='Steam Score', color='indie_status',
                     title='⭐ Success Rate by Price Band',
                     labels={'Steam Score': 'Average Steam Score'},
                     barmode='group',
                     color_discrete_map={'Indie': '#FF6B6B', 'Non-Indie': '#4ECDC4'})
        fig5.add_hline(y=0.7, line_dash="dash", line_color="green",
                      annotation_text="Success Threshold (70%)")
        fig5.update_layout(height=400)
        st.plotly_chart(fig5, use_container_width=True)
    
    # Profit analysis
    st.markdown("### 💰 Profit Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        profit_games = filtered_df[filtered_df['EstimatedProfit'] > 0]
        fig6 = go.Figure()
        for status, color in [('Indie', '#FF6B6B'), ('Non-Indie', '#4ECDC4')]:
            data = profit_games[profit_games['indie_status'] == status]['EstimatedProfit']
            fig6.add_trace(go.Histogram(x=data, name=status, opacity=0.7,
                                       marker_color=color))
        fig6.update_xaxes(type="log")
        fig6.update_layout(title='Estimated Profit (Log Scale)',
                          xaxis_title='Profit (USD)', yaxis_title='Count',
                          barmode='overlay', height=400)
        st.plotly_chart(fig6, use_container_width=True)
    
    with col2:
        # Profit per game by price bracket
        paid_games_profit = paid_games[paid_games['EstimatedProfit'] > 0].copy()
        profit_by_price = paid_games_profit.groupby('price_bracket').agg({
            'EstimatedProfit': 'median',
            'Name': 'count'
        }).reset_index()
        profit_by_price['ProfitPerGame'] = profit_by_price['EstimatedProfit'] / profit_by_price['Name']
        
        fig7 = go.Figure(go.Bar(
            x=profit_by_price['price_bracket'],
            y=profit_by_price['ProfitPerGame'],
            marker_color='#2ecc71',
            text=profit_by_price['ProfitPerGame'].apply(lambda x: f'${x:,.0f}'),
            textposition='outside'
        ))
        fig7.update_layout(title='💎 Profit Per Game by Price Point',
                          xaxis_title='Price Band', yaxis_title='Profit Per Game (USD)',
                          height=400)
        st.plotly_chart(fig7, use_container_width=True)
    
    # Free vs Paid comparison
    st.markdown("### 🆓 Free vs Paid Games")
    free_paid = filtered_df.groupby('is_free').agg({
        'Steam Score': 'mean',
        'Estimated owners': 'median',
        'Name': 'count'
    }).reset_index()
    free_paid['Type'] = free_paid['is_free'].map({True: 'Free', False: 'Paid'})
    
    fig8 = make_subplots(rows=1, cols=2, subplot_titles=('Success Rate', 'Median Owners'))
    fig8.add_trace(go.Bar(x=free_paid['Type'], y=free_paid['Steam Score'], 
                          marker_color=['#e74c3c', '#2ecc71'], name='Steam Score'), row=1, col=1)
    fig8.add_trace(go.Bar(x=free_paid['Type'], y=free_paid['Estimated owners'],
                          marker_color=['#e74c3c', '#2ecc71'], name='Owners', showlegend=False), row=1, col=2)
    fig8.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig8, use_container_width=True)

# TAB 3: Success Factors
with tab3:
    st.markdown('<p class="sub-header">Key Success Factors</p>', unsafe_allow_html=True)
    
    indie_games = filtered_df[filtered_df['indie'] == True]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Platform impact
        platform_success = filtered_df.groupby('platform_count').agg({
            'Steam Score': lambda x: ((x >= 0.7).sum() / len(x) * 100),
            'EstimatedProfit': 'median'
        }).reset_index()
        
        fig9 = go.Figure()
        fig9.add_trace(go.Bar(x=platform_success['platform_count'], 
                             y=platform_success['Steam Score'],
                             name='Success Rate',
                             marker_color=['#e74c3c', '#f39c12', '#2ecc71'],
                             text=platform_success['Steam Score'].apply(lambda x: f'{x:.1f}%'),
                             textposition='outside'))
        fig9.update_layout(title='💻 Success by Platform Count',
                          xaxis_title='Number of Platforms',
                          yaxis_title='Success Rate (%)',
                          height=400)
        st.plotly_chart(fig9, use_container_width=True)
    
    with col2:
        # DLC impact
        indie_games_copy = indie_games.copy()
        indie_games_copy['dlc_bracket'] = pd.cut(indie_games_copy['DLC count'],
                                           bins=[-1, 0, 1, 2, 3, 5, 100],
                                           labels=['None', '1', '2', '3', '4-5', '5+'])
        dlc_profit = indie_games_copy.groupby('dlc_bracket').agg({
            'EstimatedProfit': 'median',
            'Name': 'count'
        }).reset_index()
        dlc_profit = dlc_profit[dlc_profit['Name'] >= 20]
        
        fig10 = go.Figure(go.Bar(
            x=dlc_profit['dlc_bracket'],
            y=dlc_profit['EstimatedProfit'],
            marker_color='#f39c12',
            text=dlc_profit['EstimatedProfit'].apply(lambda x: f'${x:,.0f}'),
            textposition='outside'
        ))
        fig10.update_layout(title='🎁 Profit by DLC Strategy',
                          xaxis_title='Number of DLCs',
                          yaxis_title='Median Profit (USD)',
                          height=400)
        st.plotly_chart(fig10, use_container_width=True)
    
    # Achievements
    st.markdown("### 🏆 Achievement Impact")
    col1, col2 = st.columns(2)
    
    with col1:
        indie_games_copy2 = indie_games.copy()
        indie_games_copy2['achievement_bracket'] = pd.cut(indie_games_copy2['Achievements'],
                                                   bins=[0, 1, 10, 20, 50, 100, 1000],
                                                   labels=['None', '1-10', '10-20', '20-50', '50-100', '100+'])
        achievement_success = indie_games_copy2.groupby('achievement_bracket').agg({
            'Steam Score': lambda x: ((x >= 0.7).sum() / len(x) * 100),
            'Name': 'count'
        }).reset_index()
        achievement_success = achievement_success[achievement_success['Name'] >= 50]
        
        fig11 = px.bar(achievement_success, x='achievement_bracket', y='Steam Score',
                      title='Achievement Count vs Success Rate',
                      labels={'Steam Score': 'Success Rate (%)', 'achievement_bracket': 'Achievements'},
                      color='Steam Score',
                      color_continuous_scale='viridis')
        fig11.update_layout(height=400)
        st.plotly_chart(fig11, use_container_width=True)
    
    with col2:
        # Release month timing
        monthly_success = filtered_df.groupby('release_month').agg({
            'Steam Score': lambda x: ((x >= 0.7).sum() / len(x) * 100),
            'Name': 'count'
        }).reset_index()
        monthly_success = monthly_success[monthly_success['Name'] >= 100]
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_success['month_name'] = monthly_success['release_month'].apply(lambda x: month_names[x-1])
        
        colors = ['#2ecc71' if m not in [6,7,8,11,12] else '#e74c3c' 
                 for m in monthly_success['release_month']]
        fig12 = go.Figure(go.Bar(x=monthly_success['month_name'], y=monthly_success['Steam Score'],
                                marker_color=colors))
        fig12.update_layout(title='📅 Best Release Months',
                           xaxis_title='Month', yaxis_title='Success Rate (%)',
                           height=400)
        st.plotly_chart(fig12, use_container_width=True)
    
    # Reviews impact
    st.markdown("### 💬 Review Volume Impact")
    indie_games_copy3 = indie_games.copy()
    indie_games_copy3['review_bracket'] = pd.cut(indie_games_copy3['total_reviews'],
                                       bins=[0, 10, 50, 100, 500, 1000, 100000],
                                       labels=['0-10', '10-50', '50-100', '100-500', '500-1k', '1k+'])
    review_impact = indie_games_copy3.groupby('review_bracket').agg({
        'Steam Score': lambda x: ((x >= 0.7).sum() / len(x) * 100),
        'Name': 'count'
    }).reset_index()
    review_impact = review_impact[review_impact['Name'] >= 100]
    
    fig13 = go.Figure()
    fig13.add_trace(go.Scatter(x=review_impact['review_bracket'], y=review_impact['Steam Score'],
                              mode='lines+markers', line=dict(width=3, color='#e74c3c'),
                              marker=dict(size=12), fill='tozeroy'))
    fig13.update_layout(title='Success Rate by Review Volume',
                       xaxis_title='Review Count', yaxis_title='Success Rate (%)',
                       height=400)
    st.plotly_chart(fig13, use_container_width=True)

# TAB 4: Genre Analysis
with tab4:
    st.markdown('<p class="sub-header">Genre Performance Analysis</p>', unsafe_allow_html=True)
    
    indie_games = filtered_df[filtered_df['indie'] == True]
    
    # Genre success
    genre_analysis = indie_games.groupby('primary_genre').agg({
        'Steam Score': lambda x: ((x >= 0.7).sum() / len(x) * 100),
        'EstimatedProfit': 'median',
        'Name': 'count'
    }).reset_index()
    genre_analysis = genre_analysis[genre_analysis['Name'] >= 100]
    genre_analysis = genre_analysis.sort_values('Steam Score', ascending=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig14 = go.Figure(go.Bar(
            y=genre_analysis['primary_genre'],
            x=genre_analysis['Steam Score'],
            orientation='h',
            marker_color='#3498db',
            text=genre_analysis['Steam Score'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside'
        ))
        fig14.update_layout(title='🎨 Genre Success Rates',
                           xaxis_title='Success Rate (%)',
                           yaxis_title='Genre',
                           height=500)
        st.plotly_chart(fig14, use_container_width=True)
    
    with col2:
        genre_profit = genre_analysis.sort_values('EstimatedProfit', ascending=True)
        fig15 = go.Figure(go.Bar(
            y=genre_profit['primary_genre'],
            x=genre_profit['EstimatedProfit'],
            orientation='h',
            marker_color='#e74c3c',
            text=genre_profit['EstimatedProfit'].apply(lambda x: f'${x:,.0f}'),
            textposition='outside'
        ))
        fig15.update_layout(title='💰 Genre Profitability',
                           xaxis_title='Median Profit (USD)',
                           yaxis_title='Genre',
                           height=500)
        st.plotly_chart(fig15, use_container_width=True)
    
    # Genre-Platform matrix
    st.markdown("### 🎮 Genre-Platform Success Matrix")
    genre_platform = indie_games.groupby(['primary_genre', 'platform_count'])['Steam Score'].mean().reset_index()
    genre_platform_pivot = genre_platform.pivot(index='primary_genre', 
                                                columns='platform_count', 
                                                values='Steam Score')
    
    fig16 = go.Figure(data=go.Heatmap(
        z=genre_platform_pivot.values,
        x=genre_platform_pivot.columns,
        y=genre_platform_pivot.index,
        colorscale='RdYlGn',
        text=np.round(genre_platform_pivot.values, 3),
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Steam Score")
    ))
    fig16.update_layout(title='Success Score by Genre and Platform Count',
                       xaxis_title='Platform Count',
                       yaxis_title='Genre',
                       height=500)
    st.plotly_chart(fig16, use_container_width=True)
    
    # Genre trends over time
    st.markdown("### 📈 Genre Trends Over Time")
    genre_time = filtered_df.groupby(['ReleaseYear', 'primary_genre']).size().reset_index(name='count')
    top_genres = filtered_df['primary_genre'].value_counts().head(5).index
    genre_time_filtered = genre_time[genre_time['primary_genre'].isin(top_genres)]
    
    fig17 = px.line(genre_time_filtered, x='ReleaseYear', y='count', color='primary_genre',
                   title='Top 5 Genres: Release Trends',
                   labels={'count': 'Number of Games', 'ReleaseYear': 'Year'})
    fig17.update_traces(mode='lines+markers', line=dict(width=3))
    fig17.update_layout(height=400, hovermode='x unified')
    st.plotly_chart(fig17, use_container_width=True)

# TAB 5: Deep Dive
with tab5:
    st.markdown('<p class="sub-header">Interactive Deep Dive Analysis</p>', unsafe_allow_html=True)
    
    # Scatter plot with customization
    st.markdown("### 🔍 Custom Scatter Analysis")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        x_axis = st.selectbox("X-Axis", ['Price', 'Estimated owners', 'Achievements', 
                                         'DLC count', 'Peak CCU', 'Average playtime forever'])
    with col2:
        y_axis = st.selectbox("Y-Axis", ['Steam Score', 'EstimatedProfit', 'Positive', 
                                         'Peak CCU', 'Average playtime forever'], index=0)
    with col3:
        color_by = st.selectbox("Color By", ['indie_status', 'primary_genre', 'platform_count'])
    
    fig18 = px.scatter(filtered_df, x=x_axis, y=y_axis, color=color_by,
                      size='Estimated owners', hover_data=['Name', 'Price', 'Steam Score'],
                      title=f'{y_axis} vs {x_axis}',
                      opacity=0.6, height=600)
    fig18.update_traces(marker=dict(line=dict(width=0.5, color='DarkSlateGrey')))
    st.plotly_chart(fig18, use_container_width=True)
    
    # Correlation heatmap
    st.markdown("### 📊 Feature Correlation Matrix")
    numeric_cols = ['Price', 'Estimated owners', 'Steam Score', 'Achievements', 
                   'DLC count', 'platform_count', 'Peak CCU', 'EstimatedProfit']
    corr_matrix = filtered_df[numeric_cols].corr()
    
    fig19 = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
    textfont={"size": 10},
    colorbar=dict(title="Correlation")
))
fig19.update_layout(title='Correlation Matrix (Pearson)',
                    height=600,
                    autosize=True,
                    margin=dict(l=40, r=40, t=80, b=40))
st.plotly_chart(fig19, use_container_width=True)

# Allow user to pick a game to inspect details
st.markdown("### 🔎 Inspect a Game")
game_list = filtered_df['Name'].sort_values().unique()
selected_game = st.selectbox("Select a game to inspect", options=np.insert(game_list, 0, "None"))
if selected_game and selected_game != "None":
    game_row = filtered_df[filtered_df['Name'] == selected_game].iloc[0]
    st.write("**Basic details**")
    st.write({
        "Name": game_row['Name'],
        "Price": f"${game_row['Price']:.2f}",
        "Indie": bool(game_row['indie']),
        "Primary Genre": game_row['primary_genre'],
        "Release Year": int(game_row['ReleaseYear']),
        "Estimated Owners": int(game_row['Estimated owners']),
        "Steam Score": f"{game_row['Steam Score']:.3f}",
        "Estimated Profit": f"${int(game_row['EstimatedProfit']):,}"
    })

    # small sparkline for reviews (simulated because no timeseries)
    st.markdown("**Review breakdown**")
    st.progress(min(1.0, game_row['Positive'] / max(1, game_row['total_reviews'])))
    st.write(f"Positive: {int(game_row['Positive'])} — Negative: {int(game_row['Negative'])}")

# TAB 6: Insights Summary
with tab6:
    st.markdown('<p class="sub-header">Key Insights & Takeaways</p>', unsafe_allow_html=True)
    insight_col1, insight_col2 = st.columns(2)
    with insight_col1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("### 🔥 Top Quick Insights")
        # Insight 1: Best release years for success
        success_by_year = filtered_df.groupby('ReleaseYear').agg({
            'Steam Score': lambda x: ((x >= 0.7).sum() / len(x) * 100),
            'Name': 'count'
        }).reset_index()
        best_year = success_by_year.sort_values('Steam Score', ascending=False).iloc[0]
        st.write(f"- Highest success rate year: **{int(best_year['ReleaseYear'])}** ({best_year['Steam Score']:.1f}% success)")

        # Insight 2: Best genre
        if 'primary_genre' in filtered_df.columns and len(filtered_df) > 0:
            genre_perf = filtered_df.groupby('primary_genre').agg({
                'Steam Score': lambda x: ((x >= 0.7).sum() / len(x) * 100),
                'EstimatedProfit': 'median',
                'Name': 'count'
            }).reset_index()
            top_genre = genre_perf.sort_values('Steam Score', ascending=False).iloc[0]
            st.write(f"- Top performing genre (success rate): **{top_genre['primary_genre']}** ({top_genre['Steam Score']:.1f}% success)")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("### 💡 Pricing takeaways")
        # Price bracket success
        paid = filtered_df[filtered_df['is_free'] == False].copy()
        if len(paid) > 0:
            paid['price_bracket'] = pd.cut(paid['Price'],
                                           bins=[0, 5, 10, 15, 20, 30, 50, 100],
                                           labels=['$0-5', '$5-10', '$10-15', '$15-20', '$20-30', '$30-50', '$50+'])
            pb = paid.groupby('price_bracket').agg({'Steam Score': 'mean', 'Name': 'count'}).reset_index()
            best_bracket = pb.sort_values('Steam Score', ascending=False).iloc[0]
            st.write(f"- Best avg Steam Score by price bracket: **{best_bracket['price_bracket']}** (avg score {best_bracket['Steam Score']:.3f})")
        else:
            st.write("- No paid games in the current filter.")
        st.markdown("</div>", unsafe_allow_html=True)

    with insight_col2:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("### 📈 Growth & Platform insights")
        platform_counts = filtered_df.groupby('platform_count').agg({
            'Name': 'count',
            'EstimatedProfit': 'median'
        }).reset_index()
        most_platforms = platform_counts.sort_values('Name', ascending=False).iloc[0]
        st.write(f"- Most common platform count: **{int(most_platforms['platform_count'])}** platforms (count {int(most_platforms['Name'])})")
        st.write(f"- Median profit for that group: **${int(most_platforms['EstimatedProfit']):,}**")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("### 🧭 Actionable Recommendations")
        st.write("- Consider launching seasonal marketing around months with higher success rates.")
        st.write("- For Indies: prioritize 1–2 platforms initially to reduce overhead while maximizing quality.")
        st.write("- Use price brackets with historically higher average scores as a guide; pairing good post-launch support (DLCs/patches) increases longevity.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Top Games (by Estimated Profit)")
    top_games = filtered_df.sort_values('EstimatedProfit', ascending=False).head(25)[[
        'AppID', 'Name', 'Price', 'primary_genre', 'ReleaseYear', 'Estimated owners', 'EstimatedProfit', 'Steam Score'
    ]].reset_index(drop=True)
    # present a table
    st.dataframe(top_games.style.format({
        'Price': '${:.2f}',
        'EstimatedProfit': '${:,.0f}',
        'Steam Score': '{:.3f}'
    }), height=400)

    # Download filtered dataset
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download filtered data (CSV)",
        data=csv,
        file_name='filtered_steam_games.csv',
        mime='text/csv'
    )

# Footer
st.markdown("---")
st.markdown("""
<p style="text-align:center; color: #888;">Built with ❤️ — Data simulated for demo purposes. Replace `load_data()` with your real dataset (e.g., `pd.read_csv('cleaned_data.csv')`) for production use.</p>
""", unsafe_allow_html=True)
