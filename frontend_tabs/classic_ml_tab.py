"""
Classic ML Tab - Full-featured implementation with real functionality
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import subprocess
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommendation_engine import RecommendationEngine

def run_classic_ml_tab():
    """Main function for Classic ML tab with full functionality"""
    
    # Initialize session state
    if 'ml_recommendations' not in st.session_state:
        st.session_state.ml_recommendations = None
    if 'classic_campaigns' not in st.session_state:
        st.session_state.classic_campaigns = None
    if 'data_generated' not in st.session_state:
        st.session_state.data_generated = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    
    # Main header
    st.header("🎯 Classic Machine Learning System")
    st.markdown("Traditional ML approach using segmentation, prediction models, and optimization")
    
    # Create main layout with sidebar-style configuration
    st.markdown("---")
    
    # Configuration Section
    st.subheader("⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Budget slider
        budget = st.slider(
            "💰 Campaign Budget",
            min_value=1000,
            max_value=50000,
            value=10000,
            step=1000,
            format="$%d",
            help="Total budget for all campaigns"
        )
    
    with col2:
        # Business goal selector
        business_goal = st.selectbox(
            "🎯 Business Goal",
            ["maximize_roi", "maximize_revenue", "maximize_reach", "balanced"],
            format_func=lambda x: {
                "maximize_roi": "Maximize ROI",
                "maximize_revenue": "Maximize Revenue", 
                "maximize_reach": "Maximize Customer Reach",
                "balanced": "Balanced Approach"
            }.get(x, x)
        )
        
        # Target segments
        all_segments = ["Champions", "Loyal", "At Risk", "Regular", "New", "Lost"]
        target_segments = st.multiselect(
            "👥 Target Segments",
            all_segments,
            default=["Champions", "Loyal", "At Risk"],
            help="Select customer segments to target"
        )
    
    st.markdown("---")
    
    # Data Management Section
    st.subheader("📊 Data Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Data generation button
        gen_data = st.button(
            "🔄 Generate Data", 
            key="gen_data",
            help="Generate fresh customer data"
        )
        if gen_data:
            with st.spinner("Generating customer data..."):
                try:
                    # Check if data exists, if not show generation instructions
                    if not os.path.exists('data/customers.csv'):
                        st.warning("Data files not found. For local development, run: `python complete_data_gen.py`")
                        st.info("Demo data is pre-generated for cloud deployment.")
                        return
                    
                    result = subprocess.run(
                        [sys.executable, "complete_data_gen.py", "42"],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if result.returncode == 0:
                        st.success("✅ Data generated!")
                        st.session_state.data_generated = True
                    else:
                        st.error("Data generation failed")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col2:
        # Train models button
        train = st.button(
            "🤖 Train Models", 
            key="train_models",
            help="Train ML models on current data"
        )
        if train:
            with st.spinner("Training models..."):
                try:
                    # Check if models exist, if not show training instructions
                    if not os.path.exists('models/segmentation_model.pkl'):
                        st.warning("Model files not found. For local development, run: `python train_models.py`")
                        st.info("Pre-trained models are included for cloud deployment.")
                        return
                    
                    result = subprocess.run(
                        [sys.executable, "train_models.py"],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    if result.returncode == 0:
                        st.success("✅ Models trained!")
                        st.session_state.models_trained = True
                    else:
                        st.error("Training failed")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col3:
        # System status
        data_ready = os.path.exists('data/customers.csv')
        models_ready = (os.path.exists('models/segmentation_model.pkl') or 
                       os.path.exists('models/promotion_model.pkl') or
                       os.path.exists('models/optimizer_model.pkl'))
        
        if data_ready and models_ready:
            st.success("✅ System Ready")
        elif data_ready:
            st.warning("⚠️ Models not trained")
        elif models_ready:
            st.warning("⚠️ No data")
        else:
            st.error("❌ Not ready")
    
    # File upload option
    with st.expander("📁 Upload Custom Data"):
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your own customer data (must contain customer_id, age, income, etc.)"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(df)} customers")
                df.to_csv('data/uploaded_customers.csv', index=False)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    
    # Main action buttons
    st.subheader("🚀 Generate Recommendations")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        generate_button = st.button(
            "🎯 Generate",
            type="primary",
            disabled=not (data_ready and models_ready),
            help="Generate recommendations with current settings"
        )
    
    with col2:
        quick_demo = st.button(
            "⚡ Quick Demo",
            help="Run with default settings"
        )
    
    with col3:
        refresh_button = st.button(
            "🔄 Refresh",
            help="Clear current results"
        )
    
    with col4:
        if st.session_state.ml_recommendations:
            export_button = st.button(
                "📥 Export",
                help="Download results as CSV"
            )
        else:
            export_button = False
    
    # Handle button actions
    if generate_button or quick_demo:
        with st.spinner("Running ML optimization..."):
            try:
                # Use real RecommendationEngine
                engine = RecommendationEngine()
                engine.load_models()
                
                # Load customer data
                if os.path.exists('data/uploaded_customers.csv'):
                    customers = pd.read_csv('data/uploaded_customers.csv')
                elif os.path.exists('data/customers.csv'):
                    customers = pd.read_csv('data/customers.csv')
                else:
                    st.error("No customer data found! Please generate data first.")
                    st.stop()
                
                # Set parameters (removed min_roi)
                params = {
                    'budget': budget if not quick_demo else 10000,
                    'goal': business_goal if not quick_demo else 'maximize_roi',
                    'target_segments': target_segments if not quick_demo else None
                }
                
                # Generate recommendations
                results = engine.recommend(customers, params)
                
                if results and 'campaigns' in results and len(results['campaigns']) > 0:
                    st.session_state.ml_recommendations = results
                    st.session_state.classic_campaigns = results['campaigns']  # Store for comparison tab
                    st.success(f"✅ Generated {len(results['campaigns'])} campaigns!")
                    st.balloons()
                else:
                    st.error("No campaigns generated. Try adjusting parameters.")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    elif refresh_button:
        st.session_state.ml_recommendations = None
        st.rerun()
    
    elif export_button:
        export_results(st.session_state.ml_recommendations)
    
    # Display results if available
    if st.session_state.ml_recommendations:
        display_recommendations(st.session_state.ml_recommendations, budget)

def display_recommendations(results, budget):
    """Display detailed recommendation results"""
    
    st.markdown("---")
    st.subheader("📈 Results")
    
    # Debug: Show what's in results
    # st.write("Debug - Results structure:", results.keys())
    # if 'campaigns' in results:
    #     st.write("Debug - First campaign:", results['campaigns'][0] if results['campaigns'] else "No campaigns")
    
    summary = results.get('summary', {})
    campaigns = results.get('campaigns', [])
    
    # Calculate totals from campaign data if not in summary or if summary values are wrong
    if campaigns:
        df_campaigns = pd.DataFrame(campaigns)
        
        # Calculate actual totals from campaigns
        actual_cost = 0
        actual_revenue = 0
        
        # Sum up costs
        if 'cost' in df_campaigns.columns:
            actual_cost = df_campaigns['cost'].sum()
        elif 'campaign_cost' in df_campaigns.columns:
            actual_cost = df_campaigns['campaign_cost'].sum()
            
        # Sum up revenues
        if 'expected_revenue' in df_campaigns.columns:
            actual_revenue = df_campaigns['expected_revenue'].sum()
        elif 'revenue' in df_campaigns.columns:
            actual_revenue = df_campaigns['revenue'].sum()
        
        # Use actual totals if summary is missing or wrong
        revenue = summary.get('total_revenue', actual_revenue)
        cost = summary.get('total_cost', actual_cost)
        
        # If summary revenue is 0 but we have actual revenue, use actual
        if revenue == 0 and actual_revenue > 0:
            revenue = actual_revenue
        if cost == 0 and actual_cost > 0:
            cost = actual_cost
            
        # Calculate ROI from actual values
        if cost > 0:
            roi = (revenue - cost) / cost
        else:
            roi_raw = summary.get('total_roi', summary.get('roi', 0))
            # Check if ROI is percentage or decimal
            if abs(roi_raw) > 2:
                roi = roi_raw / 100  # Convert percentage to decimal for calculations
            else:
                roi = roi_raw
    else:
        roi_raw = summary.get('total_roi', summary.get('roi', 0))
        # Check if ROI is percentage or decimal
        if abs(roi_raw) > 2:
            roi = roi_raw / 100  # Convert percentage to decimal
        else:
            roi = roi_raw
        revenue = summary.get('total_revenue', 0)
        cost = summary.get('total_cost', 0)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    num_campaigns = len(campaigns) if campaigns else summary.get('campaigns_selected', 0)
    
    with col1:
        st.metric(
            "ROI",
            f"{roi*100:.1f}%",
            f"+${revenue - cost:,.0f}" if revenue > cost else f"-${cost - revenue:,.0f}"
        )
    
    with col2:
        st.metric(
            "Revenue",
            f"${revenue:,.0f}",
            f"{((revenue/cost - 1)*100):.0f}% return" if cost > 0 else "N/A"
        )
    
    with col3:
        st.metric(
            "Cost",
            f"${cost:,.0f}",
            f"{(cost/budget)*100:.0f}% of budget" if budget > 0 else "N/A"
        )
    
    with col4:
        st.metric(
            "Campaigns",
            num_campaigns,
            "optimized"
        )
    
    # ROI Visualization
    st.markdown("---")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎯 ROI Performance")
        
        # Simple ROI indicator
        roi_percent_display = roi * 100
        if roi > 0.5:
            st.success(f"Excellent ROI: {roi_percent_display:.1f}%")
        elif roi > 0.15:
            st.info(f"Good ROI: {roi_percent_display:.1f}%")
        elif roi > 0:
            st.warning(f"Low ROI: {roi_percent_display:.1f}%")
        else:
            st.error(f"Negative ROI: {roi_percent_display:.1f}%")
        
        # ROI gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = roi * 100,
            title = {'text': "Return on Investment (%)"},
            gauge = {
                'axis': {'range': [-50, 100]},
                'bar': {'color': "green" if roi > 0.15 else "red"},
                'steps': [
                    {'range': [-50, 0], 'color': "lightgray"},
                    {'range': [0, 15], 'color': "lightyellow"},
                    {'range': [15, 50], 'color': "lightgreen"},
                    {'range': [50, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 15
                }
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True, key="classic_roi_gauge")
    
    with col2:
        st.subheader("💼 Recommended Campaigns")
        
        if campaigns and len(campaigns) > 0:
            # Convert to dataframe for easier processing
            df_campaigns = pd.DataFrame(campaigns)
            
            # Sort by ROI or expected revenue if available
            if 'roi' in df_campaigns.columns:
                df_campaigns = df_campaigns.sort_values('roi', ascending=False)
            elif 'expected_roi' in df_campaigns.columns:
                df_campaigns = df_campaigns.sort_values('expected_roi', ascending=False)
            elif 'expected_revenue' in df_campaigns.columns:
                df_campaigns = df_campaigns.sort_values('expected_revenue', ascending=False)
            
            # Display top campaigns with detailed info
            num_to_show = min(5, len(df_campaigns))
            
            for i in range(num_to_show):
                campaign = df_campaigns.iloc[i]
                
                # Extract data with flexible column names
                segment = campaign.get('segment', campaign.get('segment_name', campaign.get('customer_segment', 'Unknown')))
                promotion = campaign.get('promotion', campaign.get('promotion_name', campaign.get('campaign_type', 'Unknown')))
                cost = campaign.get('cost', campaign.get('campaign_cost', 0))
                revenue = campaign.get('expected_revenue', campaign.get('revenue', 0))
                roi_raw = campaign.get('roi', campaign.get('expected_roi', 0))
                response_rate = campaign.get('response_probability', campaign.get('response_rate', campaign.get('conversion_rate', 0)))
                target_size = campaign.get('customers_targeted', campaign.get('target_size', campaign.get('segment_size', 0)))
                
                # Calculate profit
                profit = revenue - cost
                
                # Determine if ROI is already in percentage or decimal format
                # If ROI > 2, it's likely already a percentage (e.g., 73.57 means 73.57%)
                # If ROI <= 2, it's likely decimal (e.g., 0.73 means 73%)
                if abs(roi_raw) > 2:
                    roi_percentage = roi_raw
                    roi_decimal = roi_raw / 100
                else:
                    roi_decimal = roi_raw
                    roi_percentage = roi_raw * 100
                
                # If no ROI provided, calculate it
                if roi_raw == 0 and cost > 0:
                    roi_decimal = (revenue - cost) / cost
                    roi_percentage = roi_decimal * 100
                
                # Calculate overall score (0-100) based on multiple factors
                score_components = []
                
                # ROI component (40% weight)
                if roi_percentage > 0:
                    roi_score = min(100, roi_percentage)  # Already in percentage, cap at 100
                    score_components.append(roi_score * 0.4)
                
                # Response rate component (30% weight)
                if response_rate > 0:
                    response_score = min(100, response_rate * 200)  # Scale up since rates are usually < 0.5
                    score_components.append(response_score * 0.3)
                
                # Profit margin component (30% weight)
                if revenue > 0:
                    profit_margin = profit / revenue
                    margin_score = min(100, profit_margin * 150)
                    score_components.append(margin_score * 0.3)
                
                overall_score = sum(score_components) if score_components else 50
                
                # Display campaign card
                with st.container():
                    st.markdown(f"### Campaign #{i+1}: {promotion}")
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown(f"**Target Segment:** {segment}")
                        if target_size > 0:
                            st.markdown(f"**Audience Size:** {target_size:,} customers")
                        st.markdown(f"**Investment:** ${cost:,.2f}")
                        st.markdown(f"**Expected Revenue:** ${revenue:,.2f}")
                    
                    with col_b:
                        st.markdown(f"**ROI:** {roi_percentage:.1f}%")
                        st.markdown(f"**Net Profit:** ${profit:,.2f}")
                        if response_rate > 0:
                            # Check if response rate is decimal or percentage
                            if response_rate <= 1:
                                st.markdown(f"**Response Rate:** {response_rate*100:.1f}%")
                            else:
                                st.markdown(f"**Response Rate:** {response_rate:.1f}%")
                        
                        # Overall score with color coding
                        if overall_score >= 80:
                            score_color = "🟢"
                            score_label = "Excellent"
                        elif overall_score >= 60:
                            score_color = "🟡"
                            score_label = "Good"
                        elif overall_score >= 40:
                            score_color = "🟠"
                            score_label = "Fair"
                        else:
                            score_color = "🔴"
                            score_label = "Poor"
                        
                        st.markdown(f"**Overall Score:** {score_color} {overall_score:.0f}/100 ({score_label})")
                    
                    # Add a divider between campaigns
                    if i < num_to_show - 1:
                        st.markdown("---")
            
            if len(df_campaigns) > num_to_show:
                st.info(f"Showing top {num_to_show} of {len(df_campaigns)} recommended campaigns")
        else:
            st.warning("No campaign recommendations available")
    
    # Charts - only if we have campaigns
    if campaigns and len(campaigns) > 0:
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        df_campaigns = pd.DataFrame(campaigns)
        
        with col1:
            st.subheader("📊 Segments Targeted")
            
            # Find segment column
            segment_col = None
            for col in ['segment', 'segment_name', 'customer_segment']:
                if col in df_campaigns.columns:
                    segment_col = col
                    break
            
            if segment_col:
                segment_counts = df_campaigns[segment_col].value_counts()
                
                fig = px.pie(
                    values=segment_counts.values,
                    names=segment_counts.index,
                    title="Campaign Distribution"
                )
                st.plotly_chart(fig, use_container_width=True, key="classic_segment_pie")
            else:
                st.info("Segment data not available")
        
        with col2:
            st.subheader("💰 Revenue by Promotion")
            
            # Find promotion and revenue columns
            promo_col = 'promotion' if 'promotion' in df_campaigns.columns else 'promotion_name'
            revenue_col = 'expected_revenue' if 'expected_revenue' in df_campaigns.columns else 'revenue'
            
            if promo_col in df_campaigns.columns and revenue_col in df_campaigns.columns:
                promo_revenue = df_campaigns.groupby(promo_col)[revenue_col].sum()
                
                fig = px.bar(
                    x=promo_revenue.index,
                    y=promo_revenue.values,
                    title="Expected Revenue",
                    labels={'x': 'Promotion', 'y': 'Revenue ($)'}
                )
                st.plotly_chart(fig, use_container_width=True, key="classic_revenue_bar")
            else:
                st.info("Revenue data not available")
    
    # Insights - only if we have campaigns
    if campaigns and len(campaigns) > 0:
        st.markdown("---")
        st.subheader("💡 Key Insights")
        
        df_campaigns = pd.DataFrame(campaigns)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Find the right column names
            segment_col = 'segment' if 'segment' in df_campaigns.columns else 'segment_name' if 'segment_name' in df_campaigns.columns else None
            promo_col = 'promotion' if 'promotion' in df_campaigns.columns else 'promotion_name' if 'promotion_name' in df_campaigns.columns else None
            roi_col = 'roi' if 'roi' in df_campaigns.columns else 'expected_roi' if 'expected_roi' in df_campaigns.columns else None
            response_col = 'response_probability' if 'response_probability' in df_campaigns.columns else 'response_rate' if 'response_rate' in df_campaigns.columns else None
            
            insights = "**Performance Summary:**\n"
            
            if segment_col and roi_col:
                top_segment = df_campaigns.groupby(segment_col)[roi_col].mean().idxmax()
                insights += f"- Best Segment: {top_segment}\n"
            
            if promo_col and roi_col:
                top_promo = df_campaigns.groupby(promo_col)[roi_col].mean().idxmax()
                insights += f"- Best Promotion: {top_promo}\n"
            
            if response_col:
                avg_response = df_campaigns[response_col].mean()
                insights += f"- Avg Response Rate: {avg_response*100:.1f}%\n"
            
            insights += f"- Budget Used: {(cost/budget)*100:.0f}%"
            
            st.info(insights)
        
        with col2:
            if len(df_campaigns) > 0:
                top_campaign = df_campaigns.iloc[0]
                
                campaign_info = "**Top Campaign:**\n"
                
                if segment_col:
                    campaign_info += f"- Target: {top_campaign.get(segment_col, 'Unknown')}\n"
                
                if promo_col:
                    campaign_info += f"- Promotion: {top_campaign.get(promo_col, 'Unknown')}\n"
                
                if roi_col:
                    campaign_info += f"- Expected ROI: {top_campaign.get(roi_col, 0)*100:.0f}%\n"
                
                if 'cost' in df_campaigns.columns:
                    campaign_info += f"- Investment: ${top_campaign.get('cost', 0):,.0f}"
                
                st.success(campaign_info)

def export_results(results):
    """Export results to CSV"""
    if results and 'campaigns' in results:
        campaigns_df = pd.DataFrame(results['campaigns'])
        csv = campaigns_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"campaigns_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
        st.success("✅ Ready to download!")