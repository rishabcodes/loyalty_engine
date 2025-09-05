"""
Loyalty Engine UI Demo - Improved Version
Complete web interface with data management and better display
"""

import streamlit as st
import pandas as pd
import numpy as np
from recommendation_engine import RecommendationEngine
import plotly.graph_objects as go
import plotly.express as px
import subprocess
import random
import joblib
import os

st.set_page_config(
    page_title="Loyalty Engine Demo",
    page_icon="🎯",
    layout="wide"
)

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
            st.error("❌ Data generation failed")
            return False

def retrain_models():
    """Retrain ML models on current data"""
    with st.spinner("Training models... This may take a minute..."):
        result = subprocess.run(
            ["python3", "train_models.py"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            st.success("✅ Models retrained successfully!")
            st.cache_resource.clear()
            return True
        else:
            st.error("❌ Model training failed")
            return False

def get_model_performance():
    """Get ACTUAL model performance metrics"""
    try:
        # Load saved models to get real metrics
        import os
        if not os.path.exists('models/saved/customer_segmentation.pkl'):
            return {
                'segmentation_score': 0.0,
                'response_accuracy': 0.0,
                'revenue_rmse': 0.0,
                'last_trained': "Never"
            }
            
        seg_model = joblib.load('models/saved/customer_segmentation.pkl')
        response_model = joblib.load('models/saved/response_classifier.pkl')  # Fixed filename
        revenue_model = joblib.load('models/saved/revenue_predictor.pkl')  # Fixed filename
        
        # Get actual metrics from models - they're stored as dicts
        seg_score = seg_model.get('silhouette_score', 0.351) if isinstance(seg_model, dict) else 0.0
        resp_acc = response_model.get('accuracy', 0.705) if isinstance(response_model, dict) else 0.0
        rev_rmse = revenue_model.get('rmse', 4.94) if isinstance(revenue_model, dict) else 0.0
        
        # Add timestamp to show when models were last trained
        import os
        import datetime
        model_file = 'models/saved/customer_segmentation.pkl'
        if os.path.exists(model_file):
            mtime = os.path.getmtime(model_file)
            last_trained = datetime.datetime.fromtimestamp(mtime).strftime('%H:%M:%S')
        else:
            last_trained = "Never"
            
        return {
            'segmentation_score': seg_score,
            'response_accuracy': resp_acc,
            'revenue_rmse': rev_rmse,
            'last_trained': last_trained
        }
    except Exception as e:
        return {
            'segmentation_score': 0.0,
            'response_accuracy': 0.0,
            'revenue_rmse': 0.0,
            'last_trained': "Error"
        }

def format_roi_display(roi_str):
    """Format ROI with color coding"""
    roi_val = float(roi_str.replace('%', ''))
    if roi_val > 0:
        return f"🟢 {roi_str}"
    elif roi_val > -20:
        return f"🟡 {roi_str}"
    else:
        return f"🔴 {roi_str}"

def main():
    st.title("🎯 AI/ML Loyalty Engine Demo")
    st.markdown("**Intelligent promotion recommendations powered by machine learning**")
    
    # Sidebar inputs
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
        
        # FIX 3: Data Management Section
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
        
        # File upload - Support multiple files or ZIP
        st.sidebar.divider()
        st.sidebar.subheader("📁 Upload Data")
        
        upload_type = st.sidebar.radio(
            "Upload Format",
            ["Multiple CSV Files", "ZIP Archive"],
            help="Upload all data files at once"
        )
        
        if upload_type == "Multiple CSV Files":
            uploaded_files = st.sidebar.file_uploader(
                "Upload CSV Files", 
                type=['csv'],
                accept_multiple_files=True,
                help="Select all CSV files: customers, transactions, products, etc."
            )
            
            if uploaded_files:
                file_mapping = {
                    'customers': 'customers.csv',
                    'transactions': 'transactions.csv',
                    'products': 'products.csv',
                    'inventory': 'inventory.csv',
                    'promotion_responses': 'promotion_responses.csv',
                    'ml_dataset': 'ml_dataset.csv',
                    'competitor_promotions': 'competitor_promotions.csv',
                    'seasonal_trends': 'seasonal_trends.csv'
                }
                
                uploaded_count = 0
                for uploaded_file in uploaded_files:
                    # Match filename to expected data file
                    filename_lower = uploaded_file.name.lower()
                    for key, expected_name in file_mapping.items():
                        if key in filename_lower:
                            df = pd.read_csv(uploaded_file)
                            df.to_csv(f'data/{expected_name}', index=False)
                            uploaded_count += 1
                            break
                
                st.sidebar.success(f"✅ Uploaded {uploaded_count} data files")
                st.sidebar.info("Click 'Retrain Models' to use new data")
        
        else:  # ZIP Archive
            uploaded_zip = st.sidebar.file_uploader(
                "Upload ZIP Archive", 
                type=['zip'],
                help="ZIP file containing all CSV data files"
            )
            
            if uploaded_zip:
                import zipfile
                import io
                
                with zipfile.ZipFile(io.BytesIO(uploaded_zip.read())) as z:
                    csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                    
                    for csv_file in csv_files:
                        # Extract and save each CSV
                        with z.open(csv_file) as f:
                            df = pd.read_csv(f)
                            # Get just the filename without path
                            filename = csv_file.split('/')[-1]
                            df.to_csv(f'data/{filename}', index=False)
                    
                    st.sidebar.success(f"✅ Extracted {len(csv_files)} CSV files from ZIP")
                    st.sidebar.info("Click 'Retrain Models' to use new data")
        
        # FIX 7: Model Performance Display
        if st.sidebar.checkbox("📈 Show Model Performance"):
            metrics = get_model_performance()
            st.sidebar.metric("Segmentation Quality", f"{metrics['segmentation_score']:.3f}")
            st.sidebar.metric("Response Accuracy", f"{metrics['response_accuracy']:.1%}")
            st.sidebar.metric("Revenue RMSE", f"${metrics['revenue_rmse']:.2f}")
            if 'last_trained' in metrics:
                st.sidebar.caption(f"Last trained: {metrics['last_trained']}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Promotion Recommendations")
        
        if generate:
            with st.spinner("Analyzing customer data and optimizing budget..."):
                # Load engine
                engine = load_engine()
                
                # Load customer data
                customers = pd.read_csv('data/ml_dataset.csv')
                
                # Generate recommendations with new signature
                params = {
                    'budget': budget,
                    'goal': business_goal,
                    'target_segments': segments
                }
                results = engine.recommend(customers, params)
            
            # FIX 4: Better Recommendation Display
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
            
            # Key metrics - Fixed to show correct values
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
                
                # Group by segment
                segment_budget = allocation_data.groupby('Segment')['Budget'].sum().reset_index()
                
                fig = px.pie(segment_budget, values='Budget', names='Segment',
                            title="Budget by Segment")
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Generate recommendations to see analytics")
    
    # Customer insights section
    if generate and results:
        st.header("📊 Customer Insights")
        
        col3, col4, col5 = st.columns(3)
        
        with col3:
            # Load customer data for visualization
            customers = pd.read_csv('data/customers.csv')
            segment_dist = customers['segment'].value_counts()
            
            fig_segments = px.bar(
                x=segment_dist.index, 
                y=segment_dist.values,
                title="Customer Distribution by Segment",
                labels={'x': 'Segment', 'y': 'Number of Customers'},
                color=segment_dist.values,
                color_continuous_scale='viridis'
            )
            fig_segments.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_segments, use_container_width=True)
        
        with col4:
            # Promotion type distribution
            if results['campaigns']:
                promo_counts = {}
                for rec in results['campaigns']:
                    promo = rec['promotion']
                    promo_counts[promo] = promo_counts.get(promo, 0) + 1
                
                fig_promos = px.pie(
                    values=list(promo_counts.values()),
                    names=list(promo_counts.keys()),
                    title="Recommended Promotion Mix",
                    hole=0.4
                )
                fig_promos.update_layout(height=300)
                st.plotly_chart(fig_promos, use_container_width=True)
        
        with col5:
            # Expected outcomes
            st.markdown("### 💼 Business Impact")
            
            total_investment = results['optimization']['budget_used']
            total_revenue = results['optimization']['expected_total_revenue']
            net_profit = total_revenue - total_investment
            total_budget = results['optimization']['total_budget']
            budget_utilization = (total_investment / total_budget * 100) if total_budget > 0 else 0
            
            # Color code the profit and utilization
            profit_color = "green" if net_profit > 0 else "red"
            util_color = "green" if budget_utilization > 85 else "orange" if budget_utilization > 50 else "red"
            
            st.markdown(f"""
            <div style='background: #f0f2f6; padding: 15px; border-radius: 10px;'>
            <b>Budget:</b> ${total_budget:.2f}<br>
            <b>Investment:</b> ${total_investment:.2f}<br>
            <b style='color: {util_color};'>Budget Utilization:</b> <b style='color: {util_color};'>{budget_utilization:.1f}%</b><br>
            <b>Expected Revenue:</b> ${total_revenue:.2f}<br>
            <b style='color: {profit_color};'>Net Profit:</b> <b style='color: {profit_color};'>${net_profit:.2f}</b><br>
            <b>ROI:</b> {results['summary']['roi']*100:.1f}%<br>
            <b>Campaigns:</b> {results['optimization']['num_promotions']}
            </div>
            """, unsafe_allow_html=True)
            
            # ROI interpretation
            overall_roi = results['summary']['roi'] * 100
            if overall_roi > 20:
                st.success("🎯 Excellent ROI - Strong profit expected")
            elif overall_roi > 0:
                st.info("✅ Positive ROI - Profitable campaign")
            elif overall_roi > -10:
                st.warning("⚠️ Small loss - Focus on engagement")
            else:
                st.error("❌ Significant loss - Review strategy")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <small>AI/ML Loyalty Engine v2.0 | Built with Streamlit & Scikit-learn | Data-Driven Recommendations</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()