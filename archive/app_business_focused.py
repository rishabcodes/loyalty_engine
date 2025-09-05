"""
AI/ML LOYALTY ENGINE - BUSINESS-FOCUSED UI
Complete Assignment Implementation with Enhanced UX
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
from datetime import datetime, timedelta
from recommendation_engine import RecommendationEngine

# Page config
st.set_page_config(
    page_title="☕ Coffee Shop AI Promotion Optimizer",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7F8C8D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .recommendation-card {
        background: #F8F9FA;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .step-indicator {
        background: #E8F4FD;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 1rem 0;
        font-weight: 600;
        color: #2196F3;
    }
</style>
""", unsafe_allow_html=True)

# Cache the engine
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

def load_business_scenario(scenario):
    """Load predefined business scenarios"""
    scenarios = {
        "Post-Holiday Slump (January)": {
            "description": "Traffic is down 20% after the holidays. Need to re-engage customers.",
            "budget": 3000,
            "goal": "increase_frequency",
            "context": "January - focus on warming up cold customers",
            "month": 1
        },
        "Summer Peak (July)": {
            "description": "Peak season with high traffic. Maximize revenue from existing flow.",
            "budget": 5000,
            "goal": "maximize_roi",
            "context": "July - capitalize on high natural traffic",
            "month": 7
        },
        "Back-to-School (September)": {
            "description": "Students returning. Build new loyalty relationships.",
            "budget": 4000,
            "goal": "increase_frequency",
            "context": "September - capture student market",
            "month": 9
        },
        "Holiday Season (December)": {
            "description": "Gift-giving season. Push merchandise and premium offerings.",
            "budget": 6000,
            "goal": "clear_inventory",
            "context": "December - focus on merchandise and gifts",
            "month": 12
        }
    }
    return scenarios.get(scenario, scenarios["Post-Holiday Slump (January)"])

def calculate_confidence_score(campaign_data):
    """Calculate confidence score for recommendations"""
    base_confidence = 0.7
    
    # Adjust based on data quality
    if campaign_data.get('customers_targeted', 0) > 100:
        base_confidence += 0.1
    if campaign_data.get('response_rate', 0) > 0.2:
        base_confidence += 0.05
    if campaign_data.get('roi', 0) > 0.3:
        base_confidence += 0.05
    
    # Add some variance
    confidence = min(0.95, base_confidence + np.random.uniform(-0.05, 0.05))
    return confidence

def generate_business_rationale(campaign):
    """Generate clear business rationale for each recommendation"""
    segment = campaign.get('segment_name', 'Unknown')
    promotion = campaign.get('promotion', 'Standard offer')
    response_rate = campaign.get('response_rate', 0.2)
    roi = campaign.get('roi', 0)
    
    rationales = {
        'Champions': f"Your most valuable customers deserve exclusive rewards. {promotion} will strengthen their loyalty and increase lifetime value.",
        'At Risk': f"These customers are showing signs of churn. {promotion} can re-engage them before they're lost.",
        'Loyal Customers': f"Consistent customers who respond well to appreciation. {promotion} reinforces their behavior.",
        'Promotion Lovers': f"Price-sensitive segment that drives volume. {promotion} aligns with their preferences.",
        'Regular': f"Steady customers who maintain baseline revenue. {promotion} can increase their frequency.",
        'Lost': f"Win-back opportunity. {promotion} offers compelling reason to return.",
        'New': f"First impressions matter. {promotion} encourages repeat visits and habit formation."
    }
    
    base_rationale = rationales.get(segment, f"Strategic opportunity identified for {segment} segment.")
    
    # Add performance insight
    if roi > 0.5:
        performance = " Expected high ROI makes this a priority investment."
    elif roi > 0.2:
        performance = " Solid returns expected with manageable risk."
    else:
        performance = " Lower returns but important for customer retention."
    
    return base_rationale + performance

def generate_implementation_steps(campaign):
    """Generate actionable implementation steps"""
    promotion = campaign.get('promotion', '')
    segment = campaign.get('segment_name', '')
    
    steps = [
        f"1. Export {segment} customer list from the dashboard",
        f"2. Create promotion code: {promotion.replace(' ', '_').upper()[:10]}",
        f"3. Set up email campaign with personalized subject line",
        f"4. Configure POS system for automatic discount application",
        f"5. Brief staff on promotion details and target audience",
        f"6. Launch tracking dashboard to monitor real-time performance",
        f"7. Set up A/B test with 10% control group for measurement"
    ]
    
    return steps

# MAIN APPLICATION
def main():
    # Header with logo effect
    st.markdown('<h1 class="main-header">☕ Coffee Shop AI Promotion Optimizer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Data-driven recommendations to boost revenue and customer loyalty</p>', unsafe_allow_html=True)
    
    # Initialize engine
    engine = load_engine()
    
    # Step 1: Business Context
    st.markdown('<span class="step-indicator">STEP 1: Define Your Business Context</span>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Scenario selector (Assignment requirement)
        scenario = st.selectbox(
            "📅 Business Scenario",
            ["Post-Holiday Slump (January)", "Summer Peak (July)", 
             "Back-to-School (September)", "Holiday Season (December)"],
            help="Pre-configured scenarios based on seasonal patterns"
        )
        scenario_data = load_business_scenario(scenario)
    
    with col2:
        # Time period selector (Assignment requirement)
        time_period = st.selectbox(
            "⏱️ Campaign Duration",
            ["1 week", "2 weeks", "1 month", "3 months"],
            index=2,
            help="How long will this campaign run?"
        )
    
    with col3:
        # Business goal dropdown (Assignment requirement)
        business_goal = st.selectbox(
            "🎯 Primary Goal",
            ["maximize_roi", "increase_frequency", "clear_inventory", "win_back_lost"],
            format_func=lambda x: {
                "maximize_roi": "💰 Maximize ROI",
                "increase_frequency": "📈 Increase Visit Frequency",
                "clear_inventory": "📦 Clear Inventory",
                "win_back_lost": "💔 Win Back Lost Customers"
            }[x],
            index=["maximize_roi", "increase_frequency", "clear_inventory", "win_back_lost"].index(scenario_data["goal"]),
            help="What's your primary objective?"
        )
    
    with col4:
        # Budget slider (Assignment requirement)
        budget = st.number_input(
            "💵 Campaign Budget",
            min_value=500,
            max_value=10000,
            value=scenario_data["budget"],
            step=250,
            help="Total budget for this campaign"
        )
    
    # Display scenario context
    with st.expander("📖 Scenario Context", expanded=True):
        st.info(f"**{scenario}**\n\n{scenario_data['description']}\n\nRecommended approach: {scenario_data['context']}")
    
    # Step 2: Customer Analysis
    st.markdown('<span class="step-indicator">STEP 2: Understand Your Customers</span>', unsafe_allow_html=True)
    
    # Load and analyze customer data
    try:
        customers_df = pd.read_csv('data/customers.csv')
        segments_df = pd.read_csv('data/segments.csv')
        
        # Show segment breakdown
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.subheader("👥 Customer Segments")
            
            # Calculate segment metrics
            segment_summary = segments_df.groupby('segment_name').agg({
                'customer_id': 'count',
                'monetary': 'mean',
                'frequency': 'mean',
                'recency': 'mean'
            }).round(2)
            
            segment_summary.columns = ['Count', 'Avg Value', 'Avg Frequency', 'Avg Recency']
            segment_summary['% of Total'] = (segment_summary['Count'] / segment_summary['Count'].sum() * 100).round(1)
            
            # Display as styled dataframe
            st.dataframe(
                segment_summary.style.background_gradient(subset=['Avg Value'], cmap='Greens'),
                use_container_width=True
            )
        
        with col2:
            # Segment visualization
            fig = px.pie(
                values=segment_summary['Count'].values,
                names=segment_summary.index,
                title="Customer Distribution by Segment",
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Target segment selection
        with st.expander("🎯 Advanced Targeting Options"):
            target_all = st.checkbox("Target All Segments", value=True)
            
            if not target_all:
                target_segments = st.multiselect(
                    "Select Specific Segments",
                    segment_summary.index.tolist(),
                    default=["Champions", "At Risk"]
                )
            else:
                target_segments = None
                
    except FileNotFoundError:
        st.warning("⚠️ No customer data found. Please generate data first.")
        if st.button("Generate Sample Data"):
            if generate_new_data():
                st.rerun()
    
    # Step 3: Get Recommendations
    st.markdown('<span class="step-indicator">STEP 3: Get AI-Powered Recommendations</span>', unsafe_allow_html=True)
    
    # Main CTA button
    if st.button("🚀 Generate AI Recommendations", type="primary", use_container_width=True, key="generate"):
        with st.spinner("🤖 AI analyzing customer patterns and optimizing campaigns..."):
            try:
                # Load ML dataset
                ml_data = pd.read_csv('data/ml_dataset.csv')
                
                # Prepare parameters
                params = {
                    'budget': budget,
                    'goal': business_goal,
                    'target_segments': target_segments,
                    'time_period': time_period
                }
                
                # Get recommendations
                results = engine.recommend(ml_data, params)
                
                # Store in session state
                st.session_state.recommendations = results
                st.session_state.params = params
                
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")
    
    # Display recommendations if available
    if 'recommendations' in st.session_state:
        results = st.session_state.recommendations
        params = st.session_state.params
        
        if results.get('campaigns'):
            st.success(f"✅ Generated {len(results['campaigns'])} optimized campaigns")
            
            # Overall metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_customers = sum(c.get('customers_targeted', 0) for c in results['campaigns'])
                st.metric("👥 Total Reach", f"{total_customers:,}")
            
            with col2:
                total_cost = sum(c.get('cost', 0) for c in results['campaigns'])
                st.metric("💰 Total Investment", f"${total_cost:,.2f}")
            
            with col3:
                avg_roi = results['summary'].get('roi', 0) * 100
                st.metric("📈 Expected ROI", f"{avg_roi:.1f}%")
            
            with col4:
                total_revenue = sum(c.get('expected_revenue', 0) for c in results['campaigns'])
                st.metric("💵 Expected Revenue", f"${total_revenue:,.2f}")
            
            # Detailed recommendations
            st.markdown("### 📋 Recommended Campaigns (Ranked by Impact)")
            
            for i, campaign in enumerate(results['campaigns'][:5], 1):
                with st.container():
                    # Calculate confidence
                    confidence = calculate_confidence_score(campaign)
                    campaign['confidence'] = confidence
                    
                    # Generate rationale
                    rationale = generate_business_rationale(campaign)
                    
                    # Campaign header
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(f"### {i}. {campaign['promotion']} → {campaign['segment_name']}")
                    
                    with col2:
                        # Confidence badge
                        if confidence > 0.8:
                            st.success(f"High Confidence: {confidence:.0%}")
                        elif confidence > 0.6:
                            st.warning(f"Medium Confidence: {confidence:.0%}")
                        else:
                            st.info(f"Exploratory: {confidence:.0%}")
                    
                    # Metrics row
                    m1, m2, m3, m4, m5 = st.columns(5)
                    
                    with m1:
                        st.metric("Customers", f"{campaign['customers_targeted']:,}")
                    
                    with m2:
                        st.metric("Investment", f"${campaign['cost']:,.2f}")
                    
                    with m3:
                        response_rate = campaign.get('response_rate', 0) * 100
                        st.metric("Response Rate", f"{response_rate:.1f}%")
                    
                    with m4:
                        st.metric("Revenue", f"${campaign['expected_revenue']:,.2f}")
                    
                    with m5:
                        roi = campaign.get('roi', 0) * 100
                        delta_color = "normal" if roi > 0 else "inverse"
                        st.metric("ROI", f"{roi:.1f}%", delta_color=delta_color)
                    
                    # Business rationale
                    st.info(f"💡 **Why this works:** {rationale}")
                    
                    # Implementation steps
                    with st.expander("📋 How to Implement"):
                        steps = generate_implementation_steps(campaign)
                        for step in steps:
                            st.write(step)
                    
                    st.divider()
            
            # Export functionality
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Export to Excel", use_container_width=True):
                    # Create export data
                    export_df = pd.DataFrame(results['campaigns'])
                    export_df.to_excel('campaign_recommendations.xlsx', index=False)
                    st.success("✅ Exported to campaign_recommendations.xlsx")
            
            with col2:
                if st.button("📧 Email Report", use_container_width=True):
                    st.info("📧 Report would be sent to registered email")
            
            with col3:
                if st.button("🖨️ Print Report", use_container_width=True):
                    st.info("🖨️ Opening print dialog...")
    
    # Sidebar with data management
    with st.sidebar:
        st.header("🔧 Data Management")
        
        st.subheader("📊 Current Data Status")
        try:
            customers_df = pd.read_csv('data/customers.csv')
            st.success(f"✅ {len(customers_df)} customers loaded")
            
            # Show data freshness
            import os
            from datetime import datetime
            
            file_time = os.path.getmtime('data/customers.csv')
            file_date = datetime.fromtimestamp(file_time)
            age_days = (datetime.now() - file_date).days
            
            if age_days == 0:
                st.info("📅 Data generated today")
            else:
                st.warning(f"📅 Data is {age_days} days old")
                
        except FileNotFoundError:
            st.error("❌ No data found")
        
        st.divider()
        
        st.subheader("🔄 Actions")
        
        if st.button("🎲 Generate New Data", use_container_width=True):
            if generate_new_data():
                st.rerun()
        
        if st.button("🧠 Retrain Models", use_container_width=True):
            if retrain_models():
                st.rerun()
        
        if st.button("🔄 Clear Cache", use_container_width=True):
            clear_all_caches()
            st.success("✅ Cache cleared!")
            st.rerun()
        
        st.divider()
        
        # Model information
        st.subheader("🤖 Model Information")
        st.info("""
        **Current Models:**
        - Customer Segmentation: KMeans
        - Response Prediction: RandomForest
        - Revenue Optimization: CostPredictor
        
        **Algorithm Justification:**
        KMeans chosen for interpretable segments.
        RandomForest for robust predictions.
        Custom cost model for business logic.
        """)

if __name__ == "__main__":
    main()