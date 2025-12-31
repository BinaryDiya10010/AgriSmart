import pandas as pd
import pickle
import os
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class LoanMLService:
    
    def __init__(self):
        self.model = None
        self.crop_encoder = None
        self.state_encoder = None
        self._load_or_train()
    
    def _load_or_train(self):
        model_path = 'models/loan_estimator/xgboost_loan_model.pkl'
        
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.crop_encoder = data['crop_encoder']
                    self.state_encoder = data['state_encoder']
            except Exception as e:
                self._train_model()
        else:
            self._train_model()
    
    def _train_model(self):
        
        try:
            # Load datasets
            production_df = pd.read_csv('datasets/crop_production.csv')
            price_df = pd.read_csv('datasets/Crop_Data.csv')
            
            # Merge production with prices
            production_df['Crop'] = production_df['Crop'].str.strip().str.title()
            price_df['Commodity'] = price_df['Commodity'].str.strip().str.title()
            
            if 'Modal_x0020_Price' in price_df.columns:
                price_df = price_df.rename(columns={'Modal_x0020_Price': 'Modal_Price'})
            
            # Sample data for faster training
            production_sample = production_df.sample(n=min(50000, len(production_df)), random_state=42)
            
            # Merge
            merged_df = production_sample.merge(
                price_df[['Commodity', 'Modal_Price']].drop_duplicates(),
                left_on='Crop',
                right_on='Commodity',
                how='inner'
            )
            
           
            merged_df = merged_df.dropna(subset=['Production', 'Area', 'Modal_Price'])
            merged_df = merged_df[merged_df['Production'] > 0]
            merged_df = merged_df[merged_df['Area'] > 0]
            
            # Calculate yield (production per area)
            merged_df['Yield_Per_Hectare'] = merged_df['Production'] / merged_df['Area']
            
            # Calculate revenue (yield × price)
            merged_df['Revenue_Per_Hectare'] = merged_df['Yield_Per_Hectare'] * merged_df['Modal_Price']
            
            # Calculate investment (estimated as 60% of revenue - industry standard)
            merged_df['Investment_Per_Hectare'] = merged_df['Revenue_Per_Hectare'] * 0.6
            
            # Calculate optimal loan (75% of investment - standard lending rate)
            merged_df['Optimal_Loan_Per_Hectare'] = merged_df['Investment_Per_Hectare'] * 0.75
            
            # Encode categorical variables
            self.crop_encoder = LabelEncoder()
            self.state_encoder = LabelEncoder()
            
            merged_df['Crop_Encoded'] = self.crop_encoder.fit_transform(merged_df['Crop'])
            merged_df['State_Encoded'] = self.state_encoder.fit_transform(merged_df['State_Name'])
            
            # Select features
            feature_cols = ['Crop_Encoded', 'State_Encoded', 'Area', 'Modal_Price', 'Yield_Per_Hectare']
            X = merged_df[feature_cols]
            y = merged_df['Optimal_Loan_Per_Hectare']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train XGBoost
            self.model = XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_train, y_train, verbose=False)
            
            # Evaluate
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)
            
            # Metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            test_mse = mean_squared_error(y_test, y_pred_test)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_rmse = np.sqrt(test_mse)
            
            # Save model
            os.makedirs('models/loan_estimator', exist_ok=True)
            model_path = 'models/loan_estimator/xgboost_loan_model.pkl'
            
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'crop_encoder': self.crop_encoder,
                    'state_encoder': self.state_encoder,
                    'metrics': {
                        'r2': test_r2,
                        'mse': test_mse,
                        'mae': test_mae,
                        'rmse': test_rmse
                    }
                }, f)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
    
    def estimate_loan(self, crop, state, farm_size, market_price=None):
        if self.model is None:
            # Fallback to formula
            return self._formula_based_estimate(crop, farm_size, market_price)
        
        try:
            # Encode crop and state
            crop_clean = crop.strip().title()
            state_clean = state.strip().title()
            
            # Check if crop/state in training data
            if crop_clean not in self.crop_encoder.classes_:
                print(f"Crop '{crop}' not in training data, using formula")
                return self._formula_based_estimate(crop, farm_size, market_price)
            
            if state_clean not in self.state_encoder.classes_:
                # Use most common state as fallback
                state_clean = self.state_encoder.classes_[0]
            
            crop_encoded = self.crop_encoder.transform([crop_clean])[0]
            state_encoded = self.state_encoder.transform([state_clean])[0]
            
            # Get market price
            if market_price is None:
                price_df = pd.read_csv('datasets/Crop_Data.csv')
                # Fix column name
                if 'Modal_x0020_Price' in price_df.columns:
                    price_df = price_df.rename(columns={'Modal_x0020_Price': 'Modal_Price'})
                price_row = price_df[price_df['Commodity'].str.title() == crop_clean]
                market_price = price_row['Modal_Price'].mean() if not price_row.empty else 2500
            
            # Estimate yield (use average from production data)
            production_df = pd.read_csv('datasets/crop_production.csv')
            crop_data = production_df[production_df['Crop'].str.title() == crop_clean]
            avg_yield = (crop_data['Production'] / crop_data['Area']).mean() if not crop_data.empty else 20
            
            # Prepare features
            features = [[crop_encoded, state_encoded, farm_size, market_price, avg_yield]]
            
            # Predict loan per hectare
            loan_per_hectare = self.model.predict(features)[0]
            
            # Calculate totals
            total_loan = loan_per_hectare * farm_size
            total_investment = total_loan / 0.75
            gross_revenue = avg_yield * market_price * farm_size
            
            own_investment = total_investment - total_loan
            net_profit = gross_revenue - own_investment - (total_loan * 0.065)
            
            if net_profit < 0:
                net_profit = max(0, gross_revenue * 0.15)  # Minimum 15% profit margin
            
            roi = (net_profit / total_investment) * 100 if total_investment > 0 else 0
            
            return {
                'crop': crop,
                'state': state,
                'farm_size': farm_size,
                'recommended_loan': max(0, total_loan),
                'total_investment': max(0, total_investment),
                'expected_yield': abs(avg_yield),
                'market_price': abs(market_price),
                'gross_revenue': max(0, gross_revenue),
                'net_profit': int(net_profit),
                'roi': roi,
                'interest_rate': 6.5,
                'data_source': 'ML Model (XGBoost)',
                'model_used': True
            }
            
        except Exception as e:
            print(f"ML prediction error: {e}")
            return self._formula_based_estimate(crop, farm_size, market_price)
    
    def _formula_based_estimate(self, crop, farm_size, market_price=None):
        """Fallback formula-based calculation"""
        if market_price is None:
            market_price = 2500
        
        production_cost_per_hectare = 50000
        total_investment = production_cost_per_hectare * farm_size
        recommended_loan = total_investment * 0.75
        
        gross_revenue = 25 * market_price * farm_size
        own_investment = total_investment - recommended_loan
        net_profit = gross_revenue - own_investment - (recommended_loan * 0.065)
        
        if net_profit < 0:
            net_profit = max(0, gross_revenue * 0.15)  # Minimum 15% profit margin
        
        roi = (net_profit / total_investment * 100) if total_investment > 0 else 0
        
        return {
            'crop': crop,
            'farm_size': farm_size,
            'recommended_loan': recommended_loan,
            'total_investment': total_investment,
            'expected_yield': abs(25),
            'market_price': abs(market_price),
            'gross_revenue': gross_revenue,
            'net_profit': int(net_profit),
            'roi': roi,
            'interest_rate': 6.5,
            'data_source': 'Formula-based (Fallback)',
            'model_used': False
        }


print("\nInitializing Loan ML Service...")
loan_ml_service = LoanMLService()
print("Loan ML Service ready!\n")


if __name__ == '__main__':
    print("\nTesting loan estimation...")
    result = loan_ml_service.estimate_loan('Rice', 'Punjab', 2.5)
    print(f"\nCrop: {result['crop']}")
    print(f"Farm Size: {result['farm_size']} hectares")
    print(f"Recommended Loan: ₹{result['recommended_loan']:,.2f}")
    print(f"Expected Revenue: ₹{result['gross_revenue']:,.2f}")
    print(f"ROI: {result['roi']:.2f}%")
    print(f"Data Source: {result['data_source']}")
