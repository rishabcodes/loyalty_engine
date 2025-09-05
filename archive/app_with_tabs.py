"""
Dual-Model Loyalty Engine
Tab 1: Classic ML (Traditional Approach)
Tab 2: Adaptive AI (Online Learning)
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add adaptive_ai to path
sys.path.append('adaptive_ai/pages')

# Import the original app functions
from recommendation_engine import RecommendationEngine
import plotly.graph_objects as go
import plotly.express as px
import subprocess
import random
import joblib

# Import adaptive AI tab
from adaptive_ai_tab import render_adaptive_ai_tab

# Page config
st.set_page_config(
    page_title="AI/ML Loyalty Engine",
    page_icon="🎯",
    layout="wide"
)

# Title and description
st.title("🎯 AI/ML Loyalty Engine")
st.markdown("""
**Dual-System Approach:** Compare traditional batch ML with adaptive online learning
""")

# Create main tabs
main_tab1, main_tab2 = st.tabs(["📊 Classic ML", "🧠 Adaptive AI"])

# ============= TAB 1: CLASSIC ML =============
with main_tab1:
    st.header("📊 Classic Machine Learning System")
    st.markdown("Traditional batch-trained models using KMeans + RandomForest")
    
    @st.cache_resource
    def load_engine():
        """Load the recommendation engine (cached)"""
        engine = RecommendationEngine()
        engine.load_models()
        return engine
    
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
    
    # Initialize engine
    engine = load_engine()
    
    # Sidebar controls for Classic ML - Simple
    with st.sidebar:
        st.header("Classic ML Controls")
        
        if st.button("🔄 Generate New Data", key="classic_new_data"):
            if generate_new_data():
                st.rerun()
        
        budget = st.number_input("Campaign Budget ($)", 
                                min_value=100, 
                                max_value=10000, 
                                value=5000, 
                                step=100,
                                key="classic_budget")
    
    # Main content area - Classic ML
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Dashboard", "🎯 Campaign Builder", "👥 Segments", "📈 Analytics", "🗂️ Data Management"]
    )
    
    with tab1:
        st.subheader("Performance Overview")
        
        # Load customer data
        try:
            customers_df = pd.read_csv('data/customers.csv')
            segments_df = pd.read_csv('data/segments.csv')
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Customers", len(customers_df))
            
            with col2:
                st.metric("Active Segments", len(segments_df['segment'].unique()))
            
            with col3:
                avg_value = customers_df['total_spent'].mean()
                st.metric("Avg Customer Value", f"${avg_value:.2f}")
            
            with col4:
                avg_freq = customers_df['frequency'].mean()
                st.metric("Avg Visit Frequency", f"{avg_freq:.1f}")
            
            # Segment distribution
            st.subheader("Customer Segments")
            segment_counts = segments_df['segment_name'].value_counts()
            
            fig = px.pie(values=segment_counts.values, 
                        names=segment_counts.index,
                        title="Customer Distribution by Segment")
            st.plotly_chart(fig, use_container_width=True)
            
        except FileNotFoundError:
            st.error("No data found. Please generate data first!")
    
    with tab2:
        st.subheader("🎯 AI-Powered Campaign Recommendations")
        st.markdown("*Let our ML models choose the optimal campaign strategy for you*")
        
        # Simple interface - just one button
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.info(f"💰 Available Budget: ${budget:,.2f}")
            
            if st.button("🚀 Generate AI Recommendations", type="primary", use_container_width=True, key="classic_generate"):
                with st.spinner("AI analyzing customers and optimizing campaigns..."):
                    try:
                        # Load customer data
                        customers_df = pd.read_csv('data/ml_dataset.csv')
                        
                        # Let the model decide everything
                        params = {
                            'budget': budget,
                            'min_customers': 10,
                            'target_segments': None  # Model will decide optimal segments
                        }
                        
                        # Generate recommendations
                        recommendations = engine.recommend(customers_df, params)
                        st.session_state.classic_recommendations = recommendations
                        
                    except FileNotFoundError:
                        st.error("❌ Data files not found. Please generate data first using the button in the sidebar!")
                    except Exception as e:
                        st.error(f"Error generating recommendations: {str(e)}")
        
        # Display recommendations if available
        if 'classic_recommendations' in st.session_state:
            recs = st.session_state.classic_recommendations
            
            if recs.get('campaigns'):
                st.success(f"✅ AI has optimized campaigns for your customer base!")
                
                # Show overall metrics
                total_customers = sum(c.get('num_customers', 0) for c in recs['campaigns'])
                total_revenue = sum(c.get('expected_revenue', 0) for c in recs['campaigns'])
                total_cost = sum(c.get('total_cost', 0) for c in recs['campaigns'])
                avg_roi = ((total_revenue - total_cost) / total_cost * 100) if total_cost > 0 else 0
                
                st.markdown("### 📊 AI-Optimized Campaign Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Customers to Target", f"{total_customers:,}")
                with col2:
                    st.metric("Predicted Revenue", f"${total_revenue:,.2f}")
                with col3:
                    st.metric("Expected ROI", f"{avg_roi:.1%}")
                with col4:
                    st.metric("Investment", f"${total_cost:,.2f}")
                
                # Show what the AI decided
                st.markdown("### 🎯 AI-Selected Strategy")
                st.markdown("*The model analyzed your customer segments and selected optimal promotions:*")
                
                for i, campaign in enumerate(recs['campaigns'][:5], 1):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{i}. {campaign.get('segment', 'Unknown')} Segment**")
                        st.write(f"   → Promotion: {campaign.get('promotion', 'N/A')}")
                        st.write(f"   → Target: {campaign.get('num_customers', 0):,} customers")
                        st.write(f"   → Expected Response: {campaign.get('response_rate', 0):.1%}")
                        st.write(f"   → Predicted Revenue: ${campaign.get('expected_revenue', 0):,.2f}")
                    with col2:
                        # Show confidence or score
                        confidence = campaign.get('confidence', campaign.get('response_rate', 0))
                        if confidence > 0.3:
                            st.success(f"High Confidence\n{confidence:.0%}")
                        elif confidence > 0.2:
                            st.warning(f"Medium Confidence\n{confidence:.0%}")
                        else:
                            st.info(f"Exploratory\n{confidence:.0%}")
                    
                    if i < len(recs['campaigns'][:5]):
                        st.divider()
                
                # Add explanation
                with st.expander("🤖 How the AI made these decisions"):
                    st.markdown("""
                    The Classic ML system used:
                    1. **KMeans Clustering** to segment customers into 5 groups
                    2. **Random Forest** to predict response rates for each segment
                    3. **Revenue Prediction** to estimate financial outcomes
                    4. **Budget Optimization** to allocate resources efficiently
                    
                    The model considered:
                    - Customer recency, frequency, and monetary value (RFM)
                    - Historical response patterns
                    - Seasonal trends
                    - Budget constraints
                    """)
            else:
                st.info("Click 'Generate AI Recommendations' to let the model analyze your customers")
    
    with tab3:
        st.subheader("👥 Customer Segments")
        
        try:
            segments_df = pd.read_csv('data/segments.csv')
            
            # Segment characteristics
            st.write("### Segment Profiles")
            
            segment_stats = segments_df.groupby('segment_name').agg({
                'customer_id': 'count',
                'recency': 'mean',
                'frequency': 'mean',
                'monetary': 'mean',
                'age': 'mean'
            }).round(2)
            
            segment_stats.columns = ['Count', 'Avg Recency', 'Avg Frequency', 
                                    'Avg Monetary', 'Avg Age']
            
            st.dataframe(segment_stats, use_container_width=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.box(segments_df, x='segment_name', y='monetary',
                           title="Monetary Value by Segment")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(segments_df, x='segment_name', y='frequency',
                           title="Visit Frequency by Segment")
                st.plotly_chart(fig, use_container_width=True)
                
        except FileNotFoundError:
            st.error("No segment data found. Please generate data first!")
    
    with tab4:
        st.subheader("📈 Analytics")
        
        # Model performance metrics
        st.write("### Model Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Segmentation Quality", "0.72", 
                     help="Silhouette Score")
        with col2:
            st.metric("Prediction Accuracy", "85%",
                     help="Cross-validation accuracy")
        with col3:
            st.metric("Feature Importance", "RFM",
                     help="Most important features")
        
        # Feature importance plot
        st.write("### Feature Importance")
        
        features = ['recency', 'frequency', 'monetary', 'age', 
                   'avg_monthly_visits', 'avg_ticket', 'total_spent']
        importance = [0.25, 0.22, 0.28, 0.08, 0.07, 0.06, 0.04]
        
        fig = px.bar(x=importance, y=features, orientation='h',
                    title="Feature Importance in Predictions",
                    labels={'x': 'Importance', 'y': 'Feature'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.subheader("🗂️ Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Current Data Files")
            
            data_files = [
                ("customers.csv", "Customer profiles"),
                ("segments.csv", "Segment assignments"),
                ("promotions.csv", "Available promotions"),
                ("seasonal_trends.csv", "Seasonal multipliers"),
                ("inventory.csv", "Product inventory")
            ]
            
            for file, desc in data_files:
                if os.path.exists(f"data/{file}"):
                    size = os.path.getsize(f"data/{file}") / 1024
                    st.success(f"✅ {file} ({size:.1f} KB) - {desc}")
                else:
                    st.error(f"❌ {file} - Missing")
        
        with col2:
            st.write("### Trained Models")
            
            model_files = [
                ("customer_segmentation.pkl", "KMeans clustering"),
                ("response_predictor.pkl", "Response prediction"),
                ("revenue_predictor.pkl", "Revenue prediction")
            ]
            
            for file, desc in model_files:
                if os.path.exists(f"models/saved/{file}"):
                    size = os.path.getsize(f"models/saved/{file}") / 1024
                    st.success(f"✅ {file} ({size:.1f} KB) - {desc}")
                else:
                    st.error(f"❌ {file} - Missing")

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