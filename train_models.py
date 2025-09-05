"""
Model Training Pipeline for Coffee Shop Loyalty System
Trains and saves ML models for segmentation and campaign optimization
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import joblib
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Trains ML models for loyalty system"""
    
    def __init__(self, data_path='data/', model_path='models/'):
        self.data_path = data_path
        self.model_path = model_path
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
        # Create model directory if it doesn't exist
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
    
    def load_data(self):
        """Load training data from CSV files"""
        try:
            customers = pd.read_csv(f'{self.data_path}customers.csv')
            transactions = pd.read_csv(f'{self.data_path}transactions.csv')
            
            # Convert date columns
            if 'join_date' in customers.columns:
                customers['join_date'] = pd.to_datetime(customers['join_date'])
            if 'timestamp' in transactions.columns:
                transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
                
            return customers, transactions
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None
    
    def prepare_features(self, customers, transactions):
        """Prepare features for model training"""
        
        # Aggregate transaction features
        agg_dict = {
            'amount': ['sum', 'mean', 'std', 'count']
        }
        
        # Add optional columns if they exist
        if 'quantity' in transactions.columns:
            agg_dict['quantity'] = ['sum', 'mean']
        if 'product' in transactions.columns:
            agg_dict['product'] = lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'
        if 'category' in transactions.columns:
            agg_dict['category'] = lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'
        if 'promotion_used' in transactions.columns:
            agg_dict['promotion_used'] = lambda x: x.notna().sum() / len(x)
        
        transaction_features = transactions.groupby('customer_id').agg(agg_dict).reset_index()
        
        # Flatten column names dynamically
        new_cols = ['customer_id']
        for col in transaction_features.columns[1:]:
            if isinstance(col, tuple):
                if col[0] == 'amount':
                    if col[1] == 'sum': new_cols.append('total_amount')
                    elif col[1] == 'mean': new_cols.append('avg_amount')
                    elif col[1] == 'std': new_cols.append('std_amount')
                    elif col[1] == 'count': new_cols.append('transaction_count')
                elif col[0] == 'quantity':
                    if col[1] == 'sum': new_cols.append('total_quantity')
                    elif col[1] == 'mean': new_cols.append('avg_quantity')
                elif col[0] == 'product': new_cols.append('favorite_product')
                elif col[0] == 'category': new_cols.append('favorite_category')
                elif col[0] == 'promotion_used': new_cols.append('promotion_usage_rate')
            else:
                new_cols.append(col)
        transaction_features.columns = new_cols
        
        # Merge with customer data
        features = customers.merge(transaction_features, on='customer_id', how='left')
        
        # Fill missing values
        features = features.fillna(0)
        
        # Create additional features
        if 'join_date' in features.columns:
            features['days_since_join'] = (datetime.now() - features['join_date']).dt.days
        
        # Calculate customer lifetime value if not present
        if 'lifetime_value' not in features.columns and 'total_amount' in features.columns:
            features['lifetime_value'] = features['total_amount']
        
        return features
    
    def train_segmentation_model(self, features):
        """Train customer segmentation model"""
        print("Training segmentation model...")
        
        # Prepare features
        feature_cols = ['age', 'income', 'location_distance', 'total_amount', 
                       'avg_amount', 'transaction_count', 'promotion_usage_rate',
                       'days_since_join', 'lifetime_value']
        
        # Filter to existing columns
        feature_cols = [col for col in feature_cols if col in features.columns]
        
        X = features[feature_cols].fillna(0)
        
        # Create target variable if not exists
        if 'segment' not in features.columns:
            # Create synthetic segments based on spending and frequency
            features['segment'] = self._create_synthetic_segments(features)
        
        y = features['segment']
        
        # Encode target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.encoders['segment'] = le
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['segmentation'] = scaler
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        print(f"Segmentation model accuracy: {accuracy:.2%}")
        
        # Store model
        self.models['segmentation'] = model
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop features for segmentation:")
        print(importance.head())
        
        return model
    
    def train_promotion_model(self, features):
        """Train promotion effectiveness prediction model"""
        print("\nTraining promotion effectiveness model...")
        
        # Create synthetic training data for promotion effectiveness
        promotion_data = self._create_promotion_training_data(features)
        
        if promotion_data is None:
            print("Skipping promotion model training - insufficient data")
            return None
        
        X = promotion_data[['segment_encoded', 'promotion_type', 'discount_amount',
                          'customer_value', 'days_since_last_purchase']]
        y = promotion_data['conversion_rate']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['promotion'] = scaler
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Promotion model MSE: {mse:.4f}")
        print(f"Promotion model R2: {r2:.4f}")
        
        self.models['promotion'] = model
        return model
    
    def train_campaign_optimizer(self, features):
        """Train campaign optimization model"""
        print("\nTraining campaign optimization model...")
        
        # Create synthetic campaign performance data
        campaign_data = self._create_campaign_training_data(features)
        
        if campaign_data is None:
            print("Skipping campaign optimizer training - insufficient data")
            return None
        
        feature_cols = ['budget', 'segment_size', 'avg_customer_value',
                       'promotion_intensity', 'seasonality_factor']
        
        X = campaign_data[feature_cols]
        y = campaign_data['roi']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['campaign'] = scaler
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Campaign optimizer MSE: {mse:.4f}")
        print(f"Campaign optimizer R2: {r2:.4f}")
        
        self.models['optimizer'] = model
        return model
    
    def _create_synthetic_segments(self, features):
        """Create synthetic customer segments based on RFM-like logic"""
        segments = []
        
        for _, row in features.iterrows():
            if 'total_amount' in features.columns and 'transaction_count' in features.columns:
                spend = row.get('total_amount', 0)
                freq = row.get('transaction_count', 0)
                
                # Simple segmentation logic
                if spend > features['total_amount'].quantile(0.75) and freq > features['transaction_count'].quantile(0.75):
                    segment = 'Champions'
                elif spend > features['total_amount'].quantile(0.5) and freq > features['transaction_count'].quantile(0.5):
                    segment = 'Loyal'
                elif spend > features['total_amount'].quantile(0.25):
                    segment = 'Regular'
                elif freq < features['transaction_count'].quantile(0.25):
                    segment = 'At Risk'
                elif spend < features['total_amount'].quantile(0.25):
                    segment = 'Lost'
                else:
                    segment = 'New'
            else:
                # Random assignment if features not available
                segment = np.random.choice(['Champions', 'Loyal', 'Regular', 'At Risk', 'Lost', 'New'])
            
            segments.append(segment)
        
        return segments
    
    def _create_promotion_training_data(self, features):
        """Create synthetic promotion effectiveness training data"""
        if len(features) == 0:
            return None
            
        # Create synthetic promotion scenarios
        n_scenarios = len(features) * 5  # 5 promotions per customer
        
        data = []
        segment_map = {'Champions': 5, 'Loyal': 4, 'Regular': 3, 'At Risk': 2, 'Lost': 1, 'New': 0}
        
        for _ in range(n_scenarios):
            segment = np.random.choice(list(segment_map.keys()))
            
            data.append({
                'segment_encoded': segment_map[segment],
                'promotion_type': np.random.randint(0, 5),
                'discount_amount': np.random.uniform(0.1, 0.4),
                'customer_value': np.random.uniform(50, 500),
                'days_since_last_purchase': np.random.randint(1, 90),
                'conversion_rate': np.random.uniform(0.1, 0.6)
            })
        
        return pd.DataFrame(data)
    
    def _create_campaign_training_data(self, features):
        """Create synthetic campaign performance training data"""
        if len(features) == 0:
            return None
            
        # Create synthetic campaign scenarios
        n_campaigns = 500
        
        data = []
        for _ in range(n_campaigns):
            budget = np.random.uniform(1000, 50000)
            segment_size = np.random.randint(10, 1000)
            
            # Calculate expected ROI with some logic
            base_roi = np.random.uniform(0.5, 2.5)
            
            # Adjust based on factors
            if budget > 20000:
                base_roi *= 0.9  # Diminishing returns
            if segment_size < 50:
                base_roi *= 0.8  # Small audience penalty
                
            data.append({
                'budget': budget,
                'segment_size': segment_size,
                'avg_customer_value': np.random.uniform(20, 200),
                'promotion_intensity': np.random.uniform(0.1, 0.5),
                'seasonality_factor': np.random.uniform(0.8, 1.2),
                'roi': base_roi + np.random.uniform(-0.2, 0.2)
            })
        
        return pd.DataFrame(data)
    
    def save_models(self):
        """Save all trained models"""
        print("\nSaving models...")
        
        # Save main models
        for name, model in self.models.items():
            with open(f'{self.model_path}{name}_model.pkl', 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {name}_model.pkl")
        
        # Save scalers
        for name, scaler in self.scalers.items():
            with open(f'{self.model_path}{name}_scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            print(f"Saved {name}_scaler.pkl")
        
        # Save encoders
        for name, encoder in self.encoders.items():
            with open(f'{self.model_path}{name}_encoder.pkl', 'wb') as f:
                pickle.dump(encoder, f)
            print(f"Saved {name}_encoder.pkl")
        
        # Save training metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'models': list(self.models.keys()),
            'scalers': list(self.scalers.keys()),
            'encoders': list(self.encoders.keys())
        }
        
        with open(f'{self.model_path}training_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print("\nAll models saved successfully!")
        return True
    
    def train_all(self):
        """Train all models in the pipeline"""
        print("="*50)
        print("Starting Model Training Pipeline")
        print("="*50)
        
        # Load data
        customers, transactions = self.load_data()
        
        if customers is None or transactions is None:
            print("Failed to load data. Please run complete_data_gen.py first.")
            return False
        
        # Prepare features
        features = self.prepare_features(customers, transactions)
        
        # Train models
        self.train_segmentation_model(features)
        self.train_promotion_model(features)
        self.train_campaign_optimizer(features)
        
        # Save all models
        self.save_models()
        
        print("\n" + "="*50)
        print("Model Training Complete!")
        print("="*50)
        
        return True


def main():
    """Main training pipeline"""
    trainer = ModelTrainer()
    success = trainer.train_all()
    
    if success:
        print("\nModels are ready for use in the recommendation engine!")
    else:
        print("\nTraining failed. Please check data files and try again.")


if __name__ == "__main__":
    main()