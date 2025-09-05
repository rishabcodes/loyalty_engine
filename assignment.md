# AI/ML Loyalty Engine - Hands-on Prototype Assignment

## Assignment Overview

**Time Limit:** 8 hours (1 working day)  
**Tools Allowed:** Any programming language, frameworks, AI tools, online resources  
**Primary Deliverable:** Working prototype with demo  
**Data:** Use any dummy/synthetic data generation approach

## Objective

Build a working prototype of an AI/ML loyalty engine that takes merchant parameters and customer data to recommend optimal promotions. The prototype should demonstrate core functionality with a simple interface and basic ML capabilities.

## Core Requirements

### Must-Have Features
1. **Data Input Interface:** Accept merchant parameters (budget, goals, time period)
2. **Customer Segmentation:** Basic ML-based customer clustering
3. **Promotion Recommendation:** Algorithm that suggests specific promotions
4. **Budget Optimization:** Allocate budget across recommended promotions
5. **Results Dashboard:** Display recommendations with expected outcomes

### Technical Stack
Use any programming language, frameworks, AI tools, or online resources of your choice.

## Assignment Tasks

### Task 1: Data Generation & Preparation

**Deliverable:** Synthetic dataset + data preprocessing pipeline

**Requirements:**
1. **Generate dummy customer data** (500-1000 customers):
   - Customer ID, age, location
   - Purchase history (transactions over 12 months)
   - Product categories purchased
   - Spending amounts and frequency
   - Last purchase date

2. **Create merchant context data:**
   - Product catalog with categories and margins
   - Seasonal trends (monthly multipliers)
   - Competitor promotion calendar
   - Inventory levels by category

3. **Data preprocessing pipeline:**
   - Calculate relevant customer behavioral metrics
   - Create appropriate features for your ML models
   - Handle missing data and outliers using your chosen approach

### Task 2: ML Model Implementation

**Deliverable:** Working ML models with basic training pipeline

**Requirements:**
1. **Customer Segmentation Model:**
   - Implement an approach to segment customers into meaningful groups
   - Justify your choice of algorithm and number of segments
   - Assign descriptive segment names based on characteristics

2. **Promotion Response Prediction:**
   - Build a model to predict customer response to different promotions
   - Choose appropriate features and target variables
   - Consider factors like customer behavior, promotion type, and timing

3. **Model Training Pipeline:**
   - Train models on your synthetic data
   - Implement appropriate validation approach
   - Save trained models for reuse in the recommendation engine

### Task 3: Recommendation Engine

**Deliverable:** Core algorithm that generates promotion recommendations

**Requirements:**
1. **Promotion Strategy Logic:**
   - Define your chosen promotion types and their mechanics
   - Create logic for matching promotions to customer segments
   - Develop a method to estimate expected ROI for each promotion

2. **Budget Optimization Algorithm:**
   - Design an approach to distribute budget across segments and promotion types
   - Optimize for your chosen objective function (ROI, revenue, participation, etc.)
   - Handle constraints like minimum/maximum spend limits per promotion

3. **Recommendation Generator:**
   - Take inputs: budget, target segments, business goals
   - Output: ranked list of recommended promotions with rationale
   - Include expected metrics and confidence levels

### Task 4: User Interface & Demo

**Deliverable:** Simple UI for inputting parameters and viewing results

**Requirements:**
1. **Input Form:**
   - Budget slider/input field
   - Business goal selection (dropdown)
   - Time period selector
   - Optional: target customer segments

2. **Results Display:**
   - Recommended promotions in ranked order
   - Expected outcomes (participation %, revenue lift, cost)
   - Budget allocation breakdown
   - Customer segment analysis

3. **Demo Functionality:**
   - Load sample merchant scenario
   - Generate recommendations on button click
   - Display results in user-friendly format

## Sample Business Scenarios (Choose One to Implement)

### Scenario A: Fashion Retailer
- **Budget:** $5,000/month
- **Goal:** Clear slow-moving inventory
- **Context:** Winter season, competitor running 25% off
- **Customer base:** 1,000 customers, mixed demographics

### Scenario B: Coffee Shop Chain
- **Budget:** $2,000/month  
- **Goal:** Increase customer visit frequency
- **Context:** Post-holiday period, low foot traffic
- **Customer base:** 500 regular customers, loyalty program members

### Scenario C: Online Electronics Store
- **Budget:** $10,000/month
- **Goal:** Increase average order value
- **Context:** Back-to-school season approaching
- **Customer base:** 2,000 customers, tech enthusiasts

## Submission Instructions

1. **Code Repository:** Push to GitHub/GitLab or submit as ZIP file
2. **Demo Video:** 5-minute screen recording showing:
   - Quick code walkthrough
   - Live demo of functionality
   - Sample recommendations explanation
   - Brief technical approach overview
4. **Deadline:** Submit within 24 hours of assignment start

**Good luck!** We're looking forward to seeing your hands-on implementation and technical problem-solving approach.