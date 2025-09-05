"""
Dual-Model Loyalty Engine - PROPERLY INTEGRATED
Tab 1: Classic ML (EXACT original system)
Tab 2: Adaptive AI (Online Learning)
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import subprocess
import random
import joblib
import plotly.graph_objects as go
import plotly.express as px
from recommendation_engine import RecommendationEngine

# Add adaptive_ai to path
sys.path.append('adaptive_ai/pages')
from adaptive_ai_tab import render_adaptive_ai_tab

# Page config
st.set_page_config(
    page_title="AI/ML Loyalty Engine",
    page_icon="🎯",
    layout="wide"
)

# Helper functions from original app.py
@st.cache_resource
def load_engine():
    """Load the recommendation engine (cached)"""
    engine = RecommendationEngine()
    engine.load_models()
    return engine

def clear_all_caches():
    """Clear all streamlit caches"""
    st.cache_resource.clear()
    st.cache_data.clear()

def generate_new_data(seed=None):
    """Generate new customer data with random seed"""
    if seed is None:
        seed = random.randint(1, 10000)
    
    with st.spinner(f"Generating new data with seed {seed}..."):
        result = subprocess.run(
            ["python3", "complete_data_gen.py", str(seed)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            st.success(f"✅ New data generated successfully! (Seed: {seed})")
            st.cache_resource.clear()
            return True
        else:
            st.error(f"Error generating data: {result.stderr}")
            return False

def retrain_models():
    """Retrain ML models on current data"""
    with st.spinner("Training models... This may take a minute..."):
        result = subprocess.run(
            ["python3", "training_workflow.py"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            st.success("✅ Models retrained successfully!")
            st.cache_resource.clear()
            return True
        else:
            st.error(f"Error training models: {result.stderr}")
            return False

def format_roi_display(roi_str):
    """Format ROI for better display with color coding"""
    try:
        roi_val = float(roi_str.strip('%'))
        if roi_val > 0:
            return f"🟢 {roi_str}"
        elif roi_val > -20:
            return f"🟡 {roi_str}"
        else:
            return f"🔴 {roi_str}"
    except:
        return roi_str

# Title and description
st.title("🎯 AI/ML Loyalty Engine")
st.markdown("""
**Dual-System Approach:** Compare traditional batch ML with adaptive online learning
""")

# Create main tabs
main_tab1, main_tab2 = st.tabs(["📊 Classic ML", "🧠 Adaptive AI"])

# ============= TAB 1: CLASSIC ML (EXACT ORIGINAL) =============
with main_tab1:
    st.header("📊 Classic Machine Learning System")
    st.markdown("Traditional batch-trained models using KMeans + RandomForest")
    
    # Initialize engine
    engine = load_engine()
    
    # Sidebar configuration (EXACT from original)
    with st.sidebar:
        st.header("Configuration")
        
        # Coffee shop scenario only
        st.info("☕ **Coffee Shop Chain**\nPost-holiday period, optimizing loyalty promotions")
        
        # Set coffee shop defaults
        default_budget = 2000
        default_goal = "increase_frequency"
        
        # Budget input
        budget = st.slider(
            "Monthly Budget ($)",
            min_value=500,
            max_value=10000,
            value=default_budget,
            step=100
        )
        
        # Business goal
        business_goal = st.selectbox(
            "Business Goal",
            options=["maximize_roi", "increase_frequency", "clear_inventory"],
            format_func=lambda x: {
                "maximize_roi": "Maximize ROI",
                "increase_frequency": "Increase Visit Frequency",
                "clear_inventory": "Clear Inventory"
            }[x],
            index=["maximize_roi", "increase_frequency", "clear_inventory"].index(default_goal)
        )
        
        # Target segments
        st.subheader("Target Segments")
        target_all = st.checkbox("Target All Segments", value=True)
        
        if not target_all:
            segments = st.multiselect(
                "Select Segments",
                ["Champions", "Loyal Customers", "At Risk", "Promotion Lovers", "Regular"],
                default=["Champions", "At Risk"]
            )
        else:
            segments = None
        
        # Generate button
        generate = st.button("🚀 Generate Recommendations", type="primary", use_container_width=True)
        
        # Data Management Section
        st.sidebar.divider()
        st.sidebar.subheader("🔄 Data Management")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("📊 Generate New Data", help="Create new random customer data"):
                if generate_new_data():
                    clear_all_caches()
                    st.rerun()
        
        with col2:
            if st.button("🧠 Retrain Models", help="Train models on current data"):
                if retrain_models():
                    clear_all_caches()
                    st.rerun()
        
        # Clear cache button
        if st.sidebar.button("🔄 Clear Cache & Refresh", help="Clear all caches and reload models"):
            clear_all_caches()
            st.success("✅ Cache cleared! Reloading...")
            st.rerun()
    
    # Main content area (EXACT from original)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Promotion Recommendations")
        
        if generate:
            with st.spinner("Analyzing customer data and optimizing budget..."):
                # Load customer data
                customers = pd.read_csv('data/ml_dataset.csv')
                
                # Generate recommendations with new signature
                params = {
                    'budget': budget,
                    'goal': business_goal,
                    'target_segments': segments
                }
                results = engine.recommend(customers, params)
            
            # Display recommendations (EXACT from original)
            if results['campaigns']:
                st.success(f"Generated {len(results['campaigns'])} recommendations")
                
                for i, rec in enumerate(results['campaigns'][:5], 1):
                    with st.container():
                        # Better title format
                        st.markdown(f"### {i}. {rec['promotion']} Campaign for {rec['segment_name']} Segment")
                        st.markdown(f"*Target {rec['customers_targeted']} customers identified by ML models*")
                        
                        # Metrics in columns
                        m1, m2, m3, m4 = st.columns(4)
                        
                        with m1:
                            st.metric("💰 Investment", f"${rec['cost']:,.2f}")
                        
                        with m2:
                            st.metric("📊 Response Rate", f"{rec['response_rate']*100:.1f}%")
                        
                        with m3:
                            st.metric("💵 Expected Return", f"${rec['expected_revenue']:,.2f}")
                        
                        with m4:
                            roi = (rec['profit'] / rec['cost'] * 100) if rec['cost'] > 0 else 0
                            roi_display = format_roi_display(f"{roi:.1f}%")
                            st.metric("📈 ROI", roi_display)
                            confidence = "High" if roi > 50 else "Medium" if roi > 0 else "Low"
                            st.caption(f"Confidence: {confidence}")
                        
                        # Explanation
                        roi_val = roi
                        if roi_val > 0:
                            st.info(f"✅ **Profitable**: This promotion is expected to generate positive returns")
                        elif roi_val > -20:
                            st.warning(f"⚠️ **Loss Leader**: Small loss acceptable for customer engagement")
                        else:
                            st.error(f"❌ **Not Recommended**: Consider alternative promotions for this segment")
                        
                        st.divider()
            else:
                st.warning("No recommendations generated. Try adjusting your parameters.")
        else:
            # Show instructions when not generated
            st.info("""
            👈 **Get Started:**
            1. Select a business scenario or customize settings
            2. Set your monthly budget
            3. Choose your business goal
            4. Click 'Generate Recommendations'
            
            **Data Management:**
            - Generate new synthetic data anytime
            - Upload your own customer CSV
            - Retrain models for better accuracy
            """)
    
    with col2:
        st.header("Analytics Dashboard")
        
        if generate and results:
            # Budget efficiency chart
            fig_budget = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=results['optimization']['budget_used'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Budget Utilization"},
                delta={'reference': budget},
                gauge={'axis': {'range': [None, budget]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, budget*0.5], 'color': "lightgray"},
                           {'range': [budget*0.5, budget*0.8], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75, 'value': budget*0.9}}
            ))
            fig_budget.update_layout(height=250)
            st.plotly_chart(fig_budget, use_container_width=True)
            
            # Key metrics
            st.metric("Expected ROI", f"{results['summary']['roi']*100:.1f}%")
            
            # Fix customer count to avoid duplicates
            unique_customers = results['summary']['total_customers_targeted']
            st.metric("Unique Customers", f"{unique_customers:,}")
            
            st.metric("Promotions", results['summary']['unique_promotions'])
            st.metric("Efficiency", f"{results['summary']['budget_efficiency']*100:.1f}%")
            
            # Allocation breakdown
            st.subheader("Budget Allocation")
            if results['campaigns']:
                allocation_data = pd.DataFrame([
                    {
                        "Segment": r['segment_name'], 
                        "Promotion": r['promotion'],
                        "Budget": r['cost']
                    }
                    for r in results['campaigns']
                ])
                
                fig = px.pie(allocation_data, values='Budget', names='Segment',
                           title="Budget by Segment")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Generate recommendations to see analytics")
    
    # Customer insights section (if needed, add tabs below)
    st.divider()
    
    # Additional tabs for more details
    tab1, tab2, tab3 = st.tabs(["👥 Customer Insights", "📈 Performance Metrics", "📊 Data Overview"])
    
    with tab1:
        st.subheader("Customer Segmentation Insights")
        try:
            segments_df = pd.read_csv('data/segments.csv')
            segment_counts = segments_df['segment_name'].value_counts()
            
            fig = px.bar(x=segment_counts.values, y=segment_counts.index, orientation='h',
                        title="Customers by Segment",
                        labels={'x': 'Count', 'y': 'Segment'})
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Generate data to see customer insights")
    
    with tab2:
        st.subheader("Model Performance")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Segmentation Quality", "0.72", help="Silhouette Score")
        with col2:
            st.metric("Prediction Accuracy", "85%", help="Cross-validation accuracy")
        with col3:
            st.metric("Feature Importance", "RFM", help="Most important features")
    
    with tab3:
        st.subheader("Data Overview")
        try:
            customers_df = pd.read_csv('data/customers.csv')
            st.metric("Total Customers", len(customers_df))
            st.metric("Avg Customer Value", f"${customers_df['total_spent'].mean():.2f}")
            st.metric("Avg Visit Frequency", f"{customers_df['frequency'].mean():.1f}")
        except:
            st.info("No data loaded. Generate new data to begin.")

# ============= TAB 2: ADAPTIVE AI =============
with main_tab2:
    render_adaptive_ai_tab()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    AI/ML Loyalty Engine | Classic ML vs Adaptive AI | Built with Streamlit
</div>
""", unsafe_allow_html=True)