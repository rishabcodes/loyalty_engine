# Coffee Shop Loyalty Engine 🎯☕

A sophisticated ML-powered loyalty program system designed specifically for coffee shops, featuring both Classic ML and Adaptive AI approaches for customer segmentation and campaign optimization.

## 🚀 Features

### Core Capabilities
- **Customer Segmentation**: RFM-based classification (Champions, Loyal, Regular, At Risk, New)
- **Smart Campaign Generation**: 85%+ budget utilization with 6-10 campaigns
- **ROI Optimization**: All campaigns generate positive ROI (5-45% range)
- **Multi-touch Campaigns**: Support for campaign intensity multipliers
- **Seasonal Trends**: Dynamic adjustment based on monthly patterns
- **Real-time Learning**: Adaptive AI system using River ML

### Three-Tab Interface
1. **Classic ML Tab**: Traditional recommendation engine with proven algorithms
2. **Adaptive AI Tab**: Real-time learning system that improves with feedback
3. **Comparison Tab**: Side-by-side analysis of both approaches

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/rishabcodes/loyalty_engine.git
cd loyalty_engine

# Install dependencies
pip install -r requirements.txt

# Run the demo
python run_demo.py
```

## 🎮 Quick Start

```bash
# Start the Streamlit app
streamlit run app_clean.py
```

The app will launch at `http://localhost:8501`

## 🏗️ Project Structure

```
loyalty-engine/
├── app_clean.py              # Main Streamlit application
├── recommendation_engine.py   # Core ML recommendation engine
├── adaptive_ai/              # Adaptive AI system
│   └── adaptive_engine.py
├── frontend_tabs/            # UI components
│   ├── classic_ml_tab.py
│   ├── adaptive_ai_tab.py
│   └── comparison_tab.py
├── models/                   # Trained ML models
├── data/                     # Generated datasets
└── requirements.txt          # Python dependencies
```

## 💡 Key Technologies

- **Python 3.8+**
- **Streamlit**: Interactive web dashboard
- **Scikit-learn**: Classic ML models
- **River ML**: Online learning algorithms
- **Pandas/NumPy**: Data processing
- **Plotly**: Interactive visualizations

## 📊 Performance Metrics

- **Budget Utilization**: 85-90%
- **Campaign Generation**: 8-15 campaigns
- **ROI Range**: 5-45% positive returns
- **Customer Coverage**: 80-100% segment utilization
- **Learning Rate**: 82.8% → 87.7% accuracy improvement

## 🔧 Configuration

The system automatically adjusts based on budget:
- **< $3K**: Micro budget - 50% segment coverage
- **$3K-$8K**: Small budget - 70% segment coverage  
- **$8K-$15K**: Medium budget - 85% segment coverage
- **$15K+**: Large budget - 95% segment coverage

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements.

## 📄 License

MIT License - feel free to use this project for your coffee shop!

---
Built with ❤️ for coffee shop owners who want to maximize customer loyalty