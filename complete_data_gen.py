"""
Complete Data Generation Pipeline for Coffee Shop Loyalty System
Generates realistic customer, transaction, and business data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_customers(n=500):
    """Generate customer data with realistic attributes"""
    
    personas = ['coffee_enthusiast', 'convenience_driven', 'social_butterfly', 
                'health_conscious', 'budget_minded', 'sporadic']
    
    customers = []
    for i in range(n):
        customers.append({
            'customer_id': i + 1,
            'age': np.random.randint(18, 75),
            'income': np.random.randint(25000, 150000),
            'location_distance': np.random.uniform(0.1, 10),
            'persona': np.random.choice(personas),
            'preferred_time': np.random.choice(['morning', 'afternoon', 'evening']),
            'join_date': datetime.now() - timedelta(days=np.random.randint(1, 730)),
            'lifetime_value': np.random.uniform(50, 5000),
            'churn_risk': np.random.uniform(0, 1),
            'email_opt_in': np.random.choice([True, False], p=[0.7, 0.3]),
            'app_user': np.random.choice([True, False], p=[0.6, 0.4])
        })
    
    return pd.DataFrame(customers)

def generate_transactions(customers_df, n=80000):
    """Generate transaction history"""
    
    products = {
        'coffee': {'price': 4.5, 'category': 'beverage'},
        'latte': {'price': 5.5, 'category': 'beverage'},
        'cappuccino': {'price': 5.0, 'category': 'beverage'},
        'espresso': {'price': 3.5, 'category': 'beverage'},
        'pastry': {'price': 4.0, 'category': 'food'},
        'sandwich': {'price': 8.5, 'category': 'food'},
        'salad': {'price': 9.0, 'category': 'food'},
        'merchandise': {'price': 15.0, 'category': 'retail'}
    }
    
    transactions = []
    customer_ids = customers_df['customer_id'].values
    
    for _ in range(n):
        product = random.choice(list(products.keys()))
        base_price = products[product]['price']
        
        transactions.append({
            'transaction_id': len(transactions) + 1,
            'customer_id': np.random.choice(customer_ids),
            'product': product,
            'category': products[product]['category'],
            'amount': base_price * np.random.uniform(0.9, 1.1),
            'quantity': np.random.randint(1, 3),
            'timestamp': datetime.now() - timedelta(
                days=np.random.randint(0, 365),
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60)
            ),
            'promotion_used': np.random.choice([None, '10% off', '20% off', 'BOGO'], 
                                             p=[0.7, 0.1, 0.1, 0.1])
        })
    
    return pd.DataFrame(transactions)

def calculate_rfm_segments(customers_df, transactions_df):
    """Calculate RFM segments for customers"""
    
    # Calculate RFM metrics
    current_date = datetime.now()
    
    rfm = transactions_df.groupby('customer_id').agg({
        'timestamp': lambda x: (current_date - x.max()).days,  # Recency
        'transaction_id': 'count',  # Frequency
        'amount': 'sum'  # Monetary
    }).reset_index()
    
    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    
    # Score RFM with error handling for small datasets
    try:
        rfm['r_score'] = pd.qcut(rfm['recency'], 5, labels=False, duplicates='drop') + 1
        rfm['r_score'] = 6 - rfm['r_score']  # Reverse for recency (lower is better)
    except:
        rfm['r_score'] = pd.cut(rfm['recency'], bins=5, labels=False) + 1
        rfm['r_score'] = 6 - rfm['r_score']
    
    try:
        rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=False, duplicates='drop') + 1
    except:
        rfm['f_score'] = pd.cut(rfm['frequency'], bins=5, labels=False) + 1
    
    try:
        rfm['m_score'] = pd.qcut(rfm['monetary'], 5, labels=False, duplicates='drop') + 1
    except:
        rfm['m_score'] = pd.cut(rfm['monetary'], bins=5, labels=False) + 1
    
    # Combine scores
    rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)
    
    # Map to segments
    def segment_customers(row):
        if row['rfm_score'] in ['555', '554', '544', '545', '454', '455', '445']:
            return 'Champions'
        elif row['rfm_score'] in ['543', '444', '435', '355', '354', '345', '344', '335']:
            return 'Loyal'
        elif row['rfm_score'] in ['532', '535', '534', '443', '434', '343', '334', '325']:
            return 'Regular'
        elif row['rfm_score'] in ['331', '321', '312', '221', '213', '231', '241', '251']:
            return 'At Risk'
        elif row['rfm_score'] in ['155', '154', '144', '214', '215', '115', '114', '113']:
            return 'Lost'
        else:
            return 'New'
    
    rfm['segment'] = rfm.apply(segment_customers, axis=1)
    
    # Merge back to customers
    customers_df = customers_df.merge(
        rfm[['customer_id', 'recency', 'frequency', 'monetary', 'segment']], 
        on='customer_id', 
        how='left'
    )
    
    # Fill NaN segments with 'New'
    customers_df['segment'] = customers_df['segment'].fillna('New')
    
    # Rename monetary to total_spend for consistency
    customers_df = customers_df.rename(columns={'monetary': 'total_spend'})
    
    return customers_df

def generate_products():
    """Generate product catalog"""
    products = pd.DataFrame([
        {'product_id': 1, 'name': 'Coffee', 'category': 'beverage', 'price': 4.5, 'cost': 1.5, 'popularity': 0.9},
        {'product_id': 2, 'name': 'Latte', 'category': 'beverage', 'price': 5.5, 'cost': 2.0, 'popularity': 0.85},
        {'product_id': 3, 'name': 'Cappuccino', 'category': 'beverage', 'price': 5.0, 'cost': 1.8, 'popularity': 0.8},
        {'product_id': 4, 'name': 'Espresso', 'category': 'beverage', 'price': 3.5, 'cost': 1.2, 'popularity': 0.7},
        {'product_id': 5, 'name': 'Pastry', 'category': 'food', 'price': 4.0, 'cost': 1.5, 'popularity': 0.75},
        {'product_id': 6, 'name': 'Sandwich', 'category': 'food', 'price': 8.5, 'cost': 3.5, 'popularity': 0.7},
        {'product_id': 7, 'name': 'Salad', 'category': 'food', 'price': 9.0, 'cost': 3.8, 'popularity': 0.6},
        {'product_id': 8, 'name': 'Merchandise', 'category': 'retail', 'price': 15.0, 'cost': 7.0, 'popularity': 0.4}
    ])
    return products

def generate_seasonal_trends():
    """Generate seasonal trend multipliers"""
    months = range(1, 13)
    trends = []
    
    for month in months:
        # Summer peak for cold drinks, winter peak for hot drinks
        if month in [6, 7, 8]:  # Summer
            coffee_mult = 0.8
            cold_mult = 1.3
        elif month in [12, 1, 2]:  # Winter
            coffee_mult = 1.2
            cold_mult = 0.7
        else:
            coffee_mult = 1.0
            cold_mult = 1.0
            
        trends.append({
            'month': month,
            'coffee_multiplier': coffee_mult,
            'food_multiplier': 1.0 + np.random.uniform(-0.1, 0.1),
            'retail_multiplier': 1.5 if month == 12 else 1.0  # Holiday boost
        })
    
    return pd.DataFrame(trends)

def generate_competitor_promotions():
    """Generate competitor promotion data"""
    competitors = ['StarCafe', 'BeanBros', 'LocalRoast']
    promotions = []
    
    for _ in range(20):
        promotions.append({
            'competitor': np.random.choice(competitors),
            'promotion_type': np.random.choice(['discount', 'bogo', 'loyalty_points']),
            'discount_pct': np.random.randint(10, 30),
            'start_date': datetime.now() - timedelta(days=np.random.randint(0, 30)),
            'end_date': datetime.now() + timedelta(days=np.random.randint(1, 30)),
            'impact_estimate': np.random.uniform(-0.1, -0.3)  # Negative impact on our sales
        })
    
    return pd.DataFrame(promotions)

def save_all_data():
    """Generate and save all data files"""
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Generate all data
    print("Generating customers...")
    customers = generate_customers(500)
    
    print("Generating transactions...")
    transactions = generate_transactions(customers, 80000)
    
    print("Calculating RFM segments...")
    customers = calculate_rfm_segments(customers, transactions)
    
    print("Generating products...")
    products = generate_products()
    
    print("Generating seasonal trends...")
    seasonal = generate_seasonal_trends()
    
    print("Generating competitor data...")
    competitors = generate_competitor_promotions()
    
    # Save to CSV
    customers.to_csv('data/customers.csv', index=False)
    transactions.to_csv('data/transactions.csv', index=False)
    products.to_csv('data/products.csv', index=False)
    seasonal.to_csv('data/seasonal_trends.csv', index=False)
    competitors.to_csv('data/competitor_promotions.csv', index=False)
    
    print("All data files generated successfully!")
    
    # Print summary
    print(f"\nData Summary:")
    print(f"- Customers: {len(customers)}")
    print(f"- Transactions: {len(transactions)}")
    print(f"- Products: {len(products)}")
    print(f"- Customer Segments:")
    print(customers['segment'].value_counts())
    
    return {
        'customers': customers,
        'transactions': transactions,
        'products': products,
        'seasonal': seasonal,
        'competitors': competitors
    }

if __name__ == "__main__":
    save_all_data()