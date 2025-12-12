import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Studio Strategy Engine", layout="wide")

st.title("🎯 Game Strategy Engine: The Path to Success")
st.markdown("""
**Objective:** Customize the launch strategy based on your studio's capital.
*Data Source: Analysis of 50,000+ Steam Games & Global VG Sales (2008-2024)*
""")

# --- 2. DATA LOADING (Hardcoded for Stability) ---

# A. LAUNCH DATA (VG Sales)
df_launch = pd.DataFrame({
    'Platform': ['PC (Steam)', 'Gen 7 Console', 'Gen 8 Console', 'Handheld'],
    'Median Revenue': [1.68, 1.25, 1.45, 0.85], 
    'Strategy': ['Scale-Up (Target)', 'Legacy', 'High Cost', 'Niche']
})

# B. SEASONALITY (Line Chart Data)
df_season = pd.DataFrame({
    'Quarter': ['Q1', 'Q2', 'Q3', 'Q4'],
    'Traffic Score': [120, 90, 95, 150],
    'Competition Risk': [20, 40, 40, 100], # Inverse score for visualization
    'Label': ['Opportunity', 'Neutral', 'Neutral', 'Danger Zone']
})

# C. INDIE DATA (Scatter Data)
df_indie_matrix = pd.DataFrame({
    'Genre': ['Simulation', 'Puzzle', 'Strategy', 'RPG', 'Action', 'Adventure'],
    'Success Rate': [45, 40, 35, 30, 15, 10], 
    'Competition': [20, 20, 50, 80, 95, 95], # Numeric for Scatter X-Axis
    'Market Type': ['Safe Haven', 'Safe Haven', 'Viable', 'Crowded', 'Trap', 'Trap']
})

# D. PRO DATA (Scatter Data)
df_pro_matrix = pd.DataFrame({
    'Genre': ['RPG', 'Strategy', 'Simulation', 'Action', 'Adventure', 'Sports'],
    'Safety (Median)': [14, 9, 6, 10, 5, 8.5],
    'Potential (Total)': [2.3, 1.5, 1.2, 6.1, 3.3, 0.6], 
    'Market Type': ['Gold Mine', 'Gold Mine', 'Safe Bet', 'Gladiator Arena', 'Trap', 'Niche']
})

# E. MULTIPLIERS
df_pro_execution = pd.DataFrame({
    'Feature': ['Single-player', 'Multiplayer', 'English Only', 'Global (10+ Langs)'],
    'Traffic Lift': [1.0, 5.5, 1.0, 3.2],
    'Category': ['Social', 'Social', 'Reach', 'Reach']
})

# --- DASHBOARD LAYOUT ---

# --- SECTION 1: THE COMMON GROUND ---
with st.expander("Step 1: The Launchpad (Where & When?)", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Platform Strategy")
        # ENHANCED BAR CHART
        fig_launch = px.bar(df_launch, x='Platform', y='Median Revenue', color='Strategy',
                            title="Median Revenue: PC vs Consoles",
                            text_auto='$.2fM',
                            color_discrete_map={'Scale-Up (Target)': '#00cc96', 'Legacy': 'gray', 'High Cost': 'gray', 'Niche': 'gray'})
        
        # Add a reference line for average console revenue
        fig_launch.add_hline(y=1.35, line_dash="dash", line_color="red", annotation_text="Console Avg ($1.35M)")
        fig_launch.update_layout(showlegend=False)
        st.plotly_chart(fig_launch, use_container_width=True)

    with col2:
        st.subheader("Launch Window Analysis")
        # ENHANCED LINE CHART (Double Axis)
        fig_season = go.Figure()
        
        # Line 1: Traffic Potential
        fig_season.add_trace(go.Scatter(x=df_season['Quarter'], y=df_season['Traffic Score'],
                                        mode='lines+markers+text',
                                        name='Player Traffic',
                                        line=dict(color='green', width=3),
                                        text=df_season['Label'], textposition="top center"))
        
        # Line 2: Competition Risk (Dashed)
        fig_season.add_trace(go.Scatter(x=df_season['Quarter'], y=df_season['Competition Risk'],
                                        mode='lines',
                                        name='Competition Risk',
                                        line=dict(color='red', width=2, dash='dash')))
        
        fig_season.update_layout(title="Seasonality: Traffic vs. Competition Risk",
                                 xaxis_title="Launch Quarter", yaxis_title="Index Score")
        
        # Highlight Q4 Danger Zone
        fig_season.add_vrect(x0="Q4", x1="Q4", fillcolor="red", opacity=0.1, annotation_text="AVOID")
        
        st.plotly_chart(fig_season, use_container_width=True)

st.divider()

# --- SECTION 2: THE STRATEGIC PIVOT ---
st.header("Step 2: Select Your Budget Strategy")
budget_mode = st.radio("Select Studio Profile:", ["Bootstrapped Indie (Low Budget)", "Venture-Backed Pro (High Budget)"], horizontal=True)

st.divider()

# --- PATH A: INDIE STRATEGY ---
if budget_mode == "Bootstrapped Indie (Low Budget)":
    st.subheader("🛠️ The Indie 'Survival' Matrix")
    
    c1, c2 = st.columns([2, 1])
    with c1:
        # ENHANCED SCATTER PLOT
        fig_indie = px.scatter(df_indie_matrix, x="Competition", y="Success Rate", 
                               color="Market Type", size="Success Rate", text="Genre",
                               title="Indie Opportunity Matrix (Competition vs Success)",
                               labels={"Competition": "Market Saturation Index", "Success Rate": "Prob. of Profit (%)"},
                               color_discrete_map={'Safe Haven': '#00cc96', 'Viable': '#ab63fa', 'Crowded': '#ffa15a', 'Trap': '#ef553b'})
        
        fig_indie.update_traces(textposition='top center')
        
        # Add Reference Lines (The "Crosshairs")
        fig_indie.add_vline(x=50, line_dash="dot", line_color="gray", annotation_text="High Competition")
        fig_indie.add_hline(y=25, line_dash="dot", line_color="gray", annotation_text="Survival Threshold")
        
        st.plotly_chart(fig_indie, use_container_width=True)

    with c2:
        st.info("""
        **Analysis:**
        * **Simulation/Puzzle:** Located in the top-left (Low Competition, High Success).
        * **Action:** Located in bottom-right (High Competition, Low Success).
        
        **Verdict:** Target **Simulation**.
        """)

# --- PATH B: PRO STRATEGY ---
else:
    st.subheader("🚀 The Pro 'Dominance' Matrix")
    
    c1, c2 = st.columns([2, 1])
    with c1:
        # ENHANCED MARKET MATRIX
        fig_pro = px.scatter(df_pro_matrix, x="Safety (Median)", y="Potential (Total)", 
                             color="Market Type", size="Potential (Total)", text="Genre",
                             title="Pro Market Matrix: Safety vs. Potential",
                             labels={"Safety (Median)": "Median Traffic (Safety Floor)", "Potential (Total)": "Total Market Size (Millions)"},
                             color_discrete_map={'Gold Mine': '#00cc96', 'Safe Bet': '#ab63fa', 'Gladiator Arena': '#ef553b', 'Trap': '#d62728'})
        
        fig_pro.update_traces(textposition='top center')
        
        # Add Background Zones (Quadrants)
        fig_pro.add_shape(type="rect", x0=8, y0=1.5, x1=15, y1=7, line=dict(width=0), fillcolor="green", opacity=0.1, layer="below")
        fig_pro.add_annotation(x=13, y=6, text="GOLD MINE", showarrow=False, font=dict(color="green", size=14))
        
        fig_pro.add_shape(type="rect", x0=0, y0=0, x1=8, y1=1.5, line=dict(width=0), fillcolor="red", opacity=0.1, layer="below")
        fig_pro.add_annotation(x=4, y=0.5, text="TRAP ZONE", showarrow=False, font=dict(color="red", size=14))

        st.plotly_chart(fig_pro, use_container_width=True)

    with c2:
        st.success("""
        **Analysis:**
        * **RPG/Strategy:** The only genres in the "Gold Mine" (High Safety + High Potential).
        * **Adventure:** High Potential but low Safety (Trap).
        
        **Verdict:** Target **Strategy-RPG**.
        """)

    # EXECUTION CHART (Bar with Line overlay)
    st.subheader("Execution Multipliers")
    fig_exec = px.bar(df_pro_execution, x='Feature', y='Traffic Lift', color='Category',
                      title="Traffic Lift Factors (Baseline = 1.0)",
                      text_auto='.1f x')
    # Add a baseline line
    fig_exec.add_hline(y=1.0, line_dash="solid", line_color="black", annotation_text="Baseline Performance")
    st.plotly_chart(fig_exec, use_container_width=True)

# --- SIDEBAR ---
st.sidebar.title("🏆 Strategy Summary")
if budget_mode == "Bootstrapped Indie (Low Budget)":
    st.sidebar.warning("Strategy: **Niche Survival**")
    st.sidebar.markdown("* Genre: **Simulation**\n* Price: **$14.99**\n* Launch: **Q1**")
else:
    st.sidebar.success("Strategy: **Blue Ocean**")
    st.sidebar.markdown("* Genre: **Strategy-RPG**\n* Price: **$49.99**\n* Feature: **Multiplayer**")