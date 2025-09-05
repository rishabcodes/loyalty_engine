```
STREAMLIT CLOUD DEPLOYMENT CLEANUP - SPECIFIC FILE OPERATIONS

CRITICAL: Only perform the exact operations listed below. Do NOT delete any files not explicitly mentioned.

=== OPERATION 1: REMOVE REDUNDANT APP FILES ===

DELETE these exact files (keep app_clean.py):
- app.py
- app_business_focused.py
- app_dual_system.py
- app_unified.py
- app_unified_adaptive_section.py
- app_with_tabs.py

KEEP these files:
- app_clean.py (main entry point)
- recommendation_engine.py (current working version)

=== OPERATION 2: REMOVE BACKUP FILES ===

DELETE these exact backup files:
- recommendation_engine_backup.py
- recommendation_engine_backup2.py
- recommendation_engine_emergency_backup.py
- recommendation_engine_final_backup.py

=== OPERATION 3: REMOVE DEVELOPMENT ARTIFACTS ===

DELETE these exact files:
- emergencyplan.md
- .DS_Store (if present at root)
- vercel.json (conflicts with Streamlit deployment)
- Dockerfile (not needed for Streamlit Cloud)

KEEP these files:
- README.md
- assignment.md
- assignment_validator.py

=== OPERATION 4: CREATE STREAMLIT DEPLOYMENT CONFIG ===

CREATE new file: .streamlit/config.toml
LOCATION: Create .streamlit directory at root, then create config.toml inside it
CONTENT:
```toml
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[browser]
gatherUsageStats = false
```

=== OPERATION 5: UPDATE README.MD FOR DEPLOYMENT ===

FIND this section in README.md:
```bash
# Run the demo
python run_demo.py
```

REPLACE with:
```bash
# For local development
python run_demo.py

# For Streamlit Cloud deployment
# The app will automatically run app_clean.py
```

ADD this section after installation in README.md:
```markdown
## 🌐 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

## 🚀 Deployment

### Streamlit Cloud (Recommended)
1. Fork this repository
2. Connect to Streamlit Cloud
3. Deploy with main file: `app_clean.py`
4. The app will be available at: `https://your-app-name.streamlit.app`

### Local Development
```bash
streamlit run app_clean.py
```
```

=== OPERATION 6: DATA GENERATION STRATEGY FIX ===

FIND this line in frontend_tabs/classic_ml_tab.py (around line 85):
```python
result = subprocess.run(
    [sys.executable, "complete_data_gen.py", "42"],
```

REPLACE with:
```python
# Check if data exists, if not show generation instructions
if not os.path.exists('data/customers.csv'):
    st.warning("Data files not found. For local development, run: `python complete_data_gen.py`")
    st.info("Demo data is pre-generated for cloud deployment.")
    return
    
result = subprocess.run(
    [sys.executable, "complete_data_gen.py", "42"],
```

FIND this line in frontend_tabs/classic_ml_tab.py (around line 105):
```python
result = subprocess.run(
    [sys.executable, "train_models.py"],
```

REPLACE with:
```python
# Check if models exist, if not show training instructions  
if not os.path.exists('models/segmentation_model.pkl'):
    st.warning("Model files not found. For local development, run: `python train_models.py`")
    st.info("Pre-trained models are included for cloud deployment.")
    return
    
result = subprocess.run(
    [sys.executable, "train_models.py"],
```

=== OPERATION 7: ENTRY POINT DISAMBIGUATION ===

CREATE new file: main.py
LOCATION: Root directory
CONTENT:
```python
"""
Main entry point for Streamlit Cloud deployment
Redirects to the clean app interface
"""

import streamlit as st
import subprocess
import sys
import os

# Redirect to the main app
exec(open('app_clean.py').read())
```

=== OPERATION 8: UPDATE .gitignore FOR DEPLOYMENT ===

FIND this section in .gitignore:
```
# Data files (can be regenerated)
# Uncomment if you don't want to track generated data
# data/*.csv
# models/*.pkl
# models/*.joblib
```

REPLACE with:
```
# Data files (keep for deployment)
# data/*.csv  - Keep for Streamlit Cloud
# models/*.pkl - Keep for Streamlit Cloud
# models/*.joblib - Keep for Streamlit Cloud

# But ignore dynamic generation logs
data/*.log
models/*.log
```

=== OPERATION 9: FIX REQUIREMENTS.TXT VERSIONS ===

REPLACE entire requirements.txt content with:
```
# Core Data Processing
pandas==1.5.3
numpy==1.24.3

# Machine Learning  
scikit-learn==1.3.0
river==0.19.0
joblib==1.3.2

# Visualization
plotly==5.15.0
matplotlib==3.7.1
seaborn==0.12.2

# Web Framework
streamlit==1.25.0

# Utilities
faker==19.3.0
tabulate==0.9.0
```

=== VERIFICATION CHECKLIST ===

After completion, verify these files exist:
✅ app_clean.py (main entry point)
✅ .streamlit/config.toml (new)
✅ main.py (new redirect file)  
✅ README.md (updated)
✅ requirements.txt (version-pinned)
✅ data/ directory (with CSV files)
✅ models/ directory (with .pkl files)

And these files are GONE:
❌ app.py, app_business_focused.py, app_dual_system.py, app_unified.py, app_unified_adaptive_section.py, app_with_tabs.py
❌ recommendation_engine_backup*.py files
❌ emergencyplan.md
❌ vercel.json
❌ Dockerfile
❌ .DS_Store

RESULT: Clean, deployment-ready repository optimized for Streamlit Cloud with clear entry point and no conflicting configurations.
```