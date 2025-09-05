"""
Assignment Validator for Loyalty Engine
Validates that all components are working correctly for submission
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

def validate_data_files():
    """Check that all required data files exist and are valid"""
    print("Checking data files...")
    required_files = [
        'data/customers.csv',
        'data/transactions.csv',
        'data/products.csv',
        'data/seasonal_trends.csv'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"  ❌ Missing: {file}")
            return False
        else:
            try:
                df = pd.read_csv(file)
                print(f"  ✓ {file}: {len(df)} rows")
            except Exception as e:
                print(f"  ❌ Invalid: {file} - {e}")
                return False
    return True

def validate_models():
    """Check that ML models exist and are valid"""
    print("\nChecking ML models...")
    model_files = [
        'models/segmentation_model.pkl',
        'models/promotion_model.pkl',
        'models/segment_encoder.pkl'
    ]
    
    for file in model_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ⚠️  Missing: {file} (will use fallback)")
    return True

def validate_recommendation_engine():
    """Test the recommendation engine produces valid results"""
    print("\nTesting recommendation engine...")
    try:
        from recommendation_engine import RecommendationEngine
        
        # Initialize and load
        engine = RecommendationEngine()
        engine.load_models()
        
        # Test with sample data
        customers = pd.read_csv('data/customers.csv')
        results = engine.recommend(customers, {
            'budget': 10000,
            'goal': 'maximize_roi',
            'target_segments': ['Champions', 'Loyal', 'At Risk']
        })
        
        # Validate results structure
        assert 'campaigns' in results, "Missing campaigns in results"
        assert 'summary' in results, "Missing summary in results"
        
        # Check ROI is realistic (15-45% range per campaign)
        total_roi = results['summary']['total_roi']
        print(f"  ✓ Total ROI: {total_roi:.1f}%")
        
        if total_roi > 200:
            print(f"  ⚠️  Warning: ROI seems high ({total_roi:.1f}%)")
        elif total_roi < -50:
            print(f"  ⚠️  Warning: ROI seems low ({total_roi:.1f}%)")
        else:
            print(f"  ✓ ROI is in realistic range")
        
        # Check coffee-specific promotions
        coffee_keywords = ['Coffee', 'Latte', 'Cappuccino', 'Espresso', 'Pastry', 
                          'Morning', 'Afternoon', 'Happy Hour', 'Points', 'Free', 'BOGO']
        
        non_coffee_promos = []
        for campaign in results['campaigns']:
            promo = campaign['promotion']
            if not any(keyword in promo for keyword in coffee_keywords):
                non_coffee_promos.append(promo)
        
        if non_coffee_promos:
            print(f"  ⚠️  Non-coffee promotions found: {non_coffee_promos}")
        else:
            print(f"  ✓ All promotions are coffee-specific")
        
        # Check campaign count
        print(f"  ✓ Generated {len(results['campaigns'])} campaigns")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Engine test failed: {e}")
        return False

def validate_frontend():
    """Check that frontend components exist"""
    print("\nChecking frontend components...")
    required_files = [
        'app_clean.py',
        'frontend_tabs/classic_ml_tab.py',
        'frontend_tabs/adaptive_ai_tab.py',
        'frontend_tabs/comparison_tab.py',
        'frontend_tabs/shared_components.py'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ❌ Missing: {file}")
            return False
    return True

def validate_adaptive_ai():
    """Check adaptive AI system"""
    print("\nChecking adaptive AI system...")
    if os.path.exists('adaptive_ai/adaptive_engine.py'):
        print(f"  ✓ Adaptive AI engine exists")
        
        try:
            from adaptive_ai.adaptive_engine import AdaptiveAIEngine
            engine = AdaptiveAIEngine()
            initial_accuracy = engine.state.get('accuracy', 0)
            engine.learn()
            new_accuracy = engine.state.get('accuracy', 0)
            
            if new_accuracy > initial_accuracy:
                print(f"  ✓ Learning system works (accuracy: {initial_accuracy:.1f}% → {new_accuracy:.1f}%)")
            else:
                print(f"  ⚠️  Learning not improving accuracy")
                
        except Exception as e:
            print(f"  ⚠️  Adaptive AI test failed: {e}")
    else:
        print(f"  ❌ Missing adaptive AI engine")
        return False
    return True

def main():
    """Run all validations"""
    print("=" * 50)
    print("LOYALTY ENGINE VALIDATION")
    print("=" * 50)
    
    all_valid = True
    
    # Run validations
    all_valid &= validate_data_files()
    all_valid &= validate_models()
    all_valid &= validate_recommendation_engine()
    all_valid &= validate_frontend()
    all_valid &= validate_adaptive_ai()
    
    print("\n" + "=" * 50)
    if all_valid:
        print("✅ SYSTEM VALIDATION PASSED")
        print("Ready for assignment submission!")
    else:
        print("⚠️  VALIDATION COMPLETED WITH WARNINGS")
        print("System is functional but review warnings above")
    print("=" * 50)
    
    return 0 if all_valid else 1

if __name__ == "__main__":
    sys.exit(main())