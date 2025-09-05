"""
Single-command demo runner for the Loyalty Engine
Usage: python run_demo.py
"""

import subprocess
import sys
import time
import os

def run_complete_demo():
    """Run the complete demo experience"""
    print("🚀 STARTING LOYALTY ENGINE DEMO")
    print("=" * 50)
    
    # 1. Generate fresh data
    print("\n1. Generating fresh demo data...")
    try:
        subprocess.run([sys.executable, "complete_data_gen.py", "42"], check=True)
        print("   ✓ Data generation complete")
    except subprocess.CalledProcessError as e:
        print(f"   ⚠️  Warning: Data generation had issues: {e}")
        print("   Continuing with existing data...")
    
    # 2. Train models  
    print("\n2. Training ML models...")
    try:
        subprocess.run([sys.executable, "train_models.py"], check=True)
        print("   ✓ Models trained successfully")
    except subprocess.CalledProcessError as e:
        print(f"   ⚠️  Warning: Model training had issues: {e}")
        print("   Continuing with existing models...")
    
    # 3. Validate system
    print("\n3. Validating system...")
    try:
        subprocess.run([sys.executable, "assignment_validator.py"], check=True)
        print("   ✓ System validation complete")
    except subprocess.CalledProcessError as e:
        print(f"   ⚠️  Warning: Validation had issues: {e}")
    
    # 4. Launch demo
    print("\n4. Launching interactive demo...")
    print("\n🎯 DEMO READY!")
    print("📱 Opening browser at: http://localhost:8501")
    print("💡 Click 'Quick Demo' buttons for guided experience")
    print("\n" + "=" * 50)
    print("\nPress Ctrl+C to stop the demo")
    print("=" * 50 + "\n")
    
    try:
        # Activate virtual environment if it exists
        if os.path.exists("venv"):
            if sys.platform == "win32":
                activate_cmd = "venv\\Scripts\\activate && "
            else:
                activate_cmd = "source venv/bin/activate && "
        else:
            activate_cmd = ""
        
        # Launch the clean app
        subprocess.run(
            f"{activate_cmd}{sys.executable} -m streamlit run app_clean.py",
            shell=True,
            check=True
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Demo stopped by user")
        print("Thank you for trying the Loyalty Engine!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error launching demo: {e}")
        print("Please ensure Streamlit is installed: pip install streamlit")

if __name__ == "__main__":
    run_complete_demo()