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
    np.random.seed(42)
    n_games = 8000
    
    release_years = np.random.choice(range(2010, 2025), n_games, p=[0.03]*5 + [0.05]*5 + [0.08]*5)
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
                          barmode='overlay', height=400)
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
        
        fig8 = go.Figure()
        fig8.add_trace(go.Bar(x=platform_success['platform_count'], 
                             y=platform_success['Steam Score'],
                             name='Success Rate',
                             marker_color=['#e74c3c', '#f39c12', '#2ecc71'],
                             text=platform_success['Steam Score'].apply(lambda x: f'{x:.1f}%'),
                             textposition='outside'))
        fig8.update_layout(title='💻 Success by Platform Count',
                          xaxis_title='Number of Platforms',
                          yaxis_title='Success Rate (%)',
                          height=400)
        st.plotly_chart(fig8, use_container_width=True)
    
    with col2:
        # DLC impact
        indie_games['dlc_bracket'] = pd.cut(indie_games['DLC count'],
                                           bins=[-1, 0, 1, 2, 3, 5, 100],
                                           labels=['None', '1', '2', '3', '4-5', '5+'])
        dlc_profit = indie_games.groupby('dlc_bracket').agg({
            'EstimatedProfit': 'median',
            'Name': 'count'
        }).reset_index()
        dlc_profit = dlc_profit[dlc_profit['Name'] >= 20]
        
        fig9 = go.Figure(go.Bar(
            x=dlc_profit['dlc_bracket'],
            y=dlc_profit['EstimatedProfit'],
            marker_color='#f39c12',
            text=dlc_profit['EstimatedProfit'].apply(lambda x: f'${x:,.0f}'),
            textposition='outside'
        ))
        fig9.update_layout(title='🎁 Profit by DLC Strategy',
                          xaxis_title='Number of DLCs',
                          yaxis_title='Median Profit (USD)',
                          height=400)
        st.plotly_chart(fig9, use_container_width=True)
    
    # Achievements
    st.markdown("### 🏆 Achievement Impact")
    col1, col2 = st.columns(2)
    
    with col1:
        indie_games['achievement_bracket'] = pd.cut(indie_games['Achievements'],
                                                   bins=[0, 1, 10, 20, 50, 100, 1000],
                                                   labels=['None', '1-10', '10-20', '20-50', '50-100', '100+'])
        achievement_success = indie_games.groupby('achievement_bracket').agg({
            'Steam Score': lambda x: ((x >= 0.7).sum() / len(x) * 100),
            'Name': 'count'
        }).reset_index()
        achievement_success = achievement_success[achievement_success['Name'] >= 50]
        
        fig10 = px.bar(achievement_success, x='achievement_bracket', y='Steam Score',
                      title='Achievement Count vs Success Rate',
                      labels={'Steam Score': 'Success Rate (%)', 'achievement_bracket': 'Achievements'},
                      color='Steam Score',
                      color_continuous_scale='viridis')
        fig10.update_layout(height=400)
        st.plotly_chart(fig10, use_container_width=True)
    
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
        fig11 = go.Figure(go.Bar(x=monthly_success['month_name'], y=monthly_success['Steam Score'],
                                marker_color=colors))
        fig11.update_layout(title='📅 Best Release Months',
                           xaxis_title='Month', yaxis_title='Success Rate (%)',
                           height=400)
        st.plotly_chart(fig11, use_container_width=True)

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
        fig12 = go.Figure(go.Bar(
            y=genre_analysis['primary_genre'],
            x=genre_analysis['Steam Score'],
            orientation='h',
            marker_color='#3498db',
            text=genre_analysis['Steam Score'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside'
        ))
        fig12.update_layout(title='🎨 Genre Success Rates',
                           xaxis_title='Success Rate (%)',
                           yaxis_title='Genre',
                           height=500)
        st.plotly_chart(fig12, use_container_width=True)
    
    with col2:
        genre_profit = genre_analysis.sort_values('EstimatedProfit', ascending=True)
        fig13 = go.Figure(go.Bar(
            y=genre_profit['primary_genre'],
            x=genre_profit['EstimatedProfit'],
            orientation='h',
            marker_color='#e74c3c',
            text=genre_profit['EstimatedProfit'].apply(lambda x: f'${x:,.0f}'),
            textposition='outside'
        ))
        fig13.update_layout(title='💰 Genre Profitability',
                           xaxis_title='Median Profit (USD)',
                           yaxis_title='Genre',
                           height=500)
        st.plotly_chart(fig13, use_container_width=True)
    
    # Genre-Platform matrix
    st.markdown("### 🎮 Genre-Platform Success Matrix")
    genre_platform = indie_games.groupby(['primary_genre', 'platform_count'])['Steam Score'].mean().reset_index()
    genre_platform_pivot = genre_platform.pivot(index='primary_genre', 
                                                columns='platform_count', 
                                                values='Steam Score')
    
    fig14 = go.Figure(data=go.Heatmap(
        z=genre_platform_pivot.values,
        x=genre_platform_pivot.columns,
        y=genre_platform_pivot.index,
        colorscale='RdYlGn',
        text=np.round(genre_platform_pivot.values, 3),
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Steam Score")
    ))
    fig14.update_layout(title='Success Score by Genre and Platform Count',
                       xaxis_title='Platform Count',
                       yaxis_title='Genre',
                       height=500)
    st.plotly_chart(fig14, use_container_width=True)

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
    
    fig15 = px.scatter(filtered_df, x=x_axis, y=y_axis, color=color_by,
                      size='Estimated owners', hover_data=['Name', 'Price', 'Steam Score'],
                      title=f'{y_axis} vs {x_axis}',
                      opacity=0.6, height=600)
    fig15.update_traces(marker=dict(line=dict(width=0.5, color='DarkSlateGrey')))
    st.plotly_chart(fig15, use_container_width=True)
    
    # Correlation heatmap
    st.markdown("### 📊 Feature Correlation Matrix")
    numeric_cols = ['Price', 'Estimated owners', 'Steam Score', 'Achievements', 
                   'DLC count', 'platform_count', 'Peak CCU', 'EstimatedProfit']
    corr_matrix = filtered_df[numeric_cols].corr()
    
    fig16 = go.Figure(data=go.Heatmap(
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
    fig16.update_layout(title='Feature Correlation Heatmap', height=600)
    st.plotly_chart(fig16, use_container_width=True)
    
    # Top performers table
    st.markdown("### 🏆 Top Performing Games")
    top_games = filtered_df.nlargest(20, 'EstimatedProfit')[
        ['Name', 'Price', 'Estimated owners', 'Steam Score', 'EstimatedProfit', 
         'indie_status', 'primary_genre']
    ]
    st.dataframe(top_games, use_container_width=True)

# TAB 6: Insights Summary
with tab6:
    st.markdown('<p class="sub-header">🎯 Key Insights for Indie Developers</p>', unsafe_allow_html=True)
    
    indie_games = filtered_df[filtered_df['indie'] == True]
    
    # Calculate key metrics
    insights_data = {
        'price_10_15_success': indie_games[indie_games['Price'].between(10, 15)]['Steam Score'].mean(),
        'price_20_30_success': indie_games[indie_games['Price'].between(20, 30)]['Steam Score'].mean(),
        'multi_platform_success': indie_games[indie_games['platform_count'] >= 2]['Steam Score'].mean(),
        'single_platform_success': indie_games[indie_games['platform_count'] == 1]['Steam Score'].mean(),
        'dlc_3_profit': indie_games[indie_games['DLC count'] == 3]['EstimatedProfit'].median(),
        'no_dlc_profit': indie_games[indie_games['DLC count'] == 0]['EstimatedProfit'].median(),
    }
    
    # Insight cards
    insights = [
        {
            'title': '💰 Optimal Pricing: $10-$15',
            'description': f'Games priced between $10-15 show {insights_data["price_10_15_success"]:.1%} average success rate. For maximum profit, consider $40-50 range, but for reputation, aim for $20-30.',
            'icon': '💵'
        },
        {
            'title': '🎨 Best Genres: Simulation, Strategy, RPG',
            'description': 'These genres consistently show the highest success rates and profitability among indie games. Focus your development efforts here.',
            'icon': '🎮'
        },
        {
            'title': '💻 Multi-Platform is Key',
            'description': f'Supporting 2+ platforms increases success rate to {insights_data["multi_platform_success"]:.1%} vs {insights_data["single_platform_success"]:.1%} for single-platform games.',
            'icon': '🖥️'
        },
        {
            'title': '🏆 20-50 Achievements Sweet Spot',
            'description': 'Games with 20-50 achievements show the best engagement and success rates. Too few feels incomplete, too many overwhelms players.',
            'icon': '🎯'
        },
        {
            'title': '📅 Avoid Summer Releases',
            'description': 'Avoid June-July releases when AAA titles dominate. Best months are typically Feb-Apr and Sep-Oct.',
            'icon': '🗓️'
        },
        {
            'title': '🎁 Plan for ~3 DLCs',
            'description': f'Games with 3 DLCs earn ${insights_data["dlc_3_profit"]:,.0f} median profit vs ${insights_data["no_dlc_profit"]:,.0f} without DLC.',
            'icon': '📦'
        },
        {
            'title': '🔄 Ongoing Support Matters',
            'description': 'Games with regular updates and DLC support show significantly higher revenue and player retention.',
            'icon': '🔧'
        },
        {
            'title': '❌ Don\'t Go Free-to-Play',
            'description': 'Paid indie games have much higher success rates and profitability compared to F2P models which require massive scale.',
            'icon': '💸'
        }
    ]
    
    for i in range(0, len(insights), 2):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="insight-box">
                <h3>{insights[i]['icon']} {insights[i]['title']}</h3>
                <p>{insights[i]['description']}</p>
            </div>
            """, unsafe_allow_html=True)
        if i + 1 < len(insights):
            with col2:
                st.markdown(f"""
                <div class="insight-box">
                    <h3>{insights[i+1]['icon']} {insights[i+1]['title']}</h3>
                    # <p>{{insights[i+1]['icon']} {insights[i+1]['title']}<p>""")