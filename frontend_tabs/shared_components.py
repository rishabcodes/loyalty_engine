"""
Shared UI components for clean, professional demo interface
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def display_metrics_row(metrics):
    """Display a row of key metrics with consistent styling"""
    cols = st.columns(len(metrics))
    for col, (label, value, delta) in zip(cols, metrics):
        with col:
            if delta:
                st.metric(label, value, delta)
            else:
                st.metric(label, value)

def display_roi_gauge(roi_value, title="ROI"):
    """Create a professional ROI gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=roi_value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        delta={'reference': 0, 'suffix': '%'},
        gauge={
            'axis': {'range': [-50, 100]},
            'bar': {'color': "darkgreen" if roi_value > 0 else "darkred"},
            'steps': [
                {'range': [-50, 0], 'color': "lightgray"},
                {'range': [0, 25], 'color': "lightyellow"},
                {'range': [25, 50], 'color': "lightgreen"},
                {'range': [50, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0
            }
        }
    ))
    fig.update_layout(height=250)
    return fig

def display_campaign_table(campaigns_df):
    """Display campaigns in a clean, formatted table"""
    # Format numeric columns
    if 'roi' in campaigns_df.columns:
        campaigns_df['ROI'] = campaigns_df['roi'].apply(lambda x: f"{x*100:.1f}%")
    if 'cost' in campaigns_df.columns:
        campaigns_df['Cost'] = campaigns_df['cost'].apply(lambda x: f"${x:,.0f}")
    if 'expected_revenue' in campaigns_df.columns:
        campaigns_df['Revenue'] = campaigns_df['expected_revenue'].apply(lambda x: f"${x:,.0f}")
    
    # Select display columns
    display_cols = ['segment_name', 'promotion', 'Cost', 'Revenue', 'ROI']
    display_df = campaigns_df[[c for c in display_cols if c in campaigns_df.columns or c.lower() in campaigns_df.columns]]
    
    # Style the dataframe
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "segment_name": st.column_config.TextColumn("Segment", width="medium"),
            "promotion": st.column_config.TextColumn("Promotion", width="medium"),
            "Cost": st.column_config.TextColumn("Cost", width="small"),
            "Revenue": st.column_config.TextColumn("Revenue", width="small"),
            "ROI": st.column_config.TextColumn("ROI", width="small"),
        }
    )

def create_performance_chart(data, x_col, y_col, title):
    """Create a clean performance line chart"""
    fig = px.line(
        data, 
        x=x_col, 
        y=y_col, 
        title=title,
        markers=True
    )
    fig.update_layout(
        showlegend=False,
        height=300,
        xaxis_title="",
        yaxis_title="",
        title_x=0.5,
        title_font_size=16
    )
    return fig

def display_system_status(status_items):
    """Display system status indicators"""
    status_col = st.columns(len(status_items))
    for col, (label, ready) in zip(status_col, status_items):
        with col:
            if ready:
                st.success(f"✅ {label}")
            else:
                st.info(f"⏳ {label}")

def create_comparison_chart(classic_roi, adaptive_roi):
    """Create side-by-side ROI comparison"""
    fig = go.Figure(data=[
        go.Bar(name='Classic ML', x=['ROI'], y=[classic_roi * 100]),
        go.Bar(name='Adaptive AI', x=['ROI'], y=[adaptive_roi * 100])
    ])
    fig.update_layout(
        title="ROI Comparison",
        yaxis_title="ROI (%)",
        showlegend=True,
        height=400,
        title_x=0.5
    )
    return fig

def display_demo_instructions():
    """Display quick demo instructions"""
    with st.expander("📖 Quick Demo Guide", expanded=False):
        st.markdown("""
        ### Classic ML Tab (30 seconds)
        1. Click **"Run Quick Demo"** for instant results
        2. Review ROI and campaign recommendations
        3. Export results or generate new campaigns
        
        ### Adaptive AI Tab (60 seconds)
        1. Click **"Demo Mode"** to see learning progression
        2. Watch the system learn and discover patterns
        3. Generate campaigns using learned patterns
        
        ### Comparison Tab (30 seconds)
        1. Click **"Run Head-to-Head"** for direct comparison
        2. Review winner analysis
        3. Export results for presentation
        """)

def apply_professional_theme():
    """Apply consistent professional styling"""
    st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-left: 20px;
            padding-right: 20px;
            background-color: #f0f2f6;
            border-radius: 8px;
            color: #333333;
            font-weight: 500;
            border: 1px solid #ddd;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #e1e5eb;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1f77b4;
            color: white !important;
            border: 1px solid #1f77b4;
        }
        div[data-testid="metric-container"] {
            background-color: #f0f2f6;
            border: 1px solid #cccccc;
            padding: 10px;
            border-radius: 8px;
            margin: 5px;
        }
        .stButton > button {
            background-color: #1f77b4;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            border: none;
            padding: 0.5rem 1rem;
        }
        .stButton > button:hover {
            background-color: #1557a0;
        }
        </style>
    """, unsafe_allow_html=True)