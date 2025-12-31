import pandas as pd
import pickle
import os
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

class CropMLService:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.crop_classes = None
        self.feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        self._load_or_train()
    
    def _load_or_train(self):
        model_path = 'models/crop_recommendation/xgboost_model.pkl'
        
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.scaler = data['scaler']
                    self.label_encoder = data['label_encoder']
                    self.crop_classes = data['classes']
            except Exception as e:
                self._train_model()
        else:
            self._train_model()
    
    def _train_model(self):
        
        try:
            df = pd.read_csv('datasets/Crop_recommendation.csv')
            
            X = df[self.feature_names]
            y = df['label']
            
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            self.crop_classes = self.label_encoder.classes_
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.model = XGBClassifier(
                n_estimators=100,  
                max_depth=5,       
                learning_rate=0.05,  
                subsample=0.7,     
                colsample_bytree=0.7,  
                min_child_weight=3,  
                gamma=0.1,  
                reg_alpha=0.1,  
                reg_lambda=1.0,  
                objective='multi:softprob',
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
            
            self.model.fit(X_train_scaled, y_train, verbose=False)
            
            train_accuracy = self.model.score(X_train_scaled, y_train)
            test_accuracy = self.model.score(X_test_scaled, y_test)
            gap = (train_accuracy - test_accuracy) * 100
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
            
            y_pred = self.model.predict(X_test_scaled)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            os.makedirs('models/crop_recommendation', exist_ok=True)
            model_path = 'models/crop_recommendation/xgboost_model.pkl'
            
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'label_encoder': self.label_encoder,
                    'classes': self.crop_classes,
                    'feature_names': self.feature_names,
                    'test_accuracy': test_accuracy,
                    'train_accuracy': train_accuracy,
                    'overfitting_gap': gap,
                    'cv_mean': cv_scores.mean(),
                    'f1_score': f1
                }, f)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
    
    def recommend_crops(self, soil_data, climate_data, farm_data=None):
        if self.model is None:
            return {'error': 'Model not loaded', 'crops': []}
        
        try:
            features = [[
                float(soil_data.get('nitrogen', 50)),
                float(soil_data.get('phosphorus', 50)),
                float(soil_data.get('potassium', 50)),
                float(climate_data.get('temperature', 25)),
                float(climate_data.get('humidity', 70)),
                float(soil_data.get('ph', 6.5)),
                float(climate_data.get('rainfall', 100))
            ]]
            
            features_scaled = self.scaler.transform(features)
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            top_indices = np.argsort(probabilities)[::-1]  
            
            recommendations = []
            for i, idx in enumerate(top_indices):
                crop = self.crop_classes[idx]
                confidence = probabilities[idx]
                
                if confidence < 0.45:
                    continue
                
                recommendations.append({
                    'rank': i + 1,
                    'name': crop.capitalize(),
                    'compatibility': int(confidence * 100),
                    'confidence': 'High' if confidence > 0.7 else 'Medium' if confidence > 0.5 else 'Low',
                    'probability': float(confidence)
                })
            
            return {
                'crops': recommendations,
                'location': f"{climate_data.get('district', 'Unknown')}, {climate_data.get('state', 'India')}",
                'model_version': 'XGBoost v2.0 (Regularized)'
            }
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return {'error': str(e), 'crops': []}


#print("\nInitializing Crop ML Service...")
crop_ml_service = CropMLService()
#print("Crop ML Service ready!\n")


if __name__ == '__main__':
    #print("\nTesting model...")
    result = crop_ml_service.recommend_crops(
        soil_data={'nitrogen': 90, 'phosphorus': 42, 'potassium': 43, 'ph': 6.5},
        climate_data={'temperature': 21, 'humidity': 82, 'rainfall': 203, 'state': 'Punjab', 'district': 'Ludhiana'}
    )
    #print(f"\nTop 5 Recommendations:")
    for crop in result['crops']:
        print(f"  {crop['rank']}. {crop['name']}: {crop['compatibility']}% ({crop['confidence']})")
