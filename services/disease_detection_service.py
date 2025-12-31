import os
import json
import numpy as np
from PIL import Image

class DiseaseDetectionService:
    
    def __init__(self):
        self.model = None
        self.class_indices = None
        self.class_names = None
        self._load_model()
    
    def _load_model(self):
        model_path = 'models/disease_detection/mobilenetv2_disease.h5'
        indices_path = 'models/disease_detection/class_indices.json'
        
        if os.path.exists(model_path) and os.path.exists(indices_path):
            try:
                import tensorflow as tf
                from tensorflow.keras.models import load_model
                
                self.model = load_model(model_path)
                
                with open(indices_path, 'r') as f:
                    self.class_indices = json.load(f)
                
                self.class_names = {v: k for k, v in self.class_indices.items()}
                
            except Exception as e:
                self.model = None
        else:
            pass
    
    def predict(self, image_path):
        if self.model is None:
            raise Exception("Model not loaded. Please train the model first.")
        return self._predict_with_model(image_path)
    
    def _predict_with_model(self, image_path):
        try:
            from tensorflow.keras.preprocessing import image as keras_image
            
            img = keras_image.load_img(image_path, target_size=(224, 224))
            img_array = keras_image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  
            
            predictions = self.model.predict(img_array, verbose=0)[0]
            top_class_idx = np.argmax(predictions)
            confidence = predictions[top_class_idx]
            disease = self.class_names[top_class_idx]
            
            all_predictions = []
            for idx, prob in enumerate(predictions):
                all_predictions.append({
                    'disease': self.class_names[idx],
                    'probability': float(prob * 100)
                })
            
            all_predictions.sort(key=lambda x: x['probability'], reverse=True)
            
            return {
                'disease': disease,
                'confidence': float(confidence * 100),
                'severity': self._calculate_severity(disease, confidence),
                'all_predictions': all_predictions,
                'treatment': self._get_treatment(disease),
                'model_used': 'CNN (MobileNetV2)'
            }
            
        except Exception as e:
            raise Exception(f"Error during prediction: {str(e)}")
    
    def _calculate_severity(self, disease, confidence):
        if disease.lower() == 'healthy':
            return 'None'
        elif confidence > 0.85:
            return 'High'
        elif confidence > 0.65:
            return 'Medium'
        else:
            return 'Low'
    
    def _get_treatment(self, disease):
        treatments = {
            'Powdery': {
                'name': 'Powdery Mildew',
                'description': 'Fungal disease causing white powdery coating on leaves',
                'immediate': [
                    'Remove severely infected leaves immediately',
                    'Apply sulfur-based fungicide (80% WP @ 2g/L water)',
                    'Spray neem oil solution (3ml/L) every 7 days',
                    'Ensure proper air circulation between plants',
                    'Apply copper oxychloride (3g/L) if severe'
                ],
                'preventive': [
                    'Avoid overhead irrigation - water at base',
                    'Maintain proper plant spacing (30-45cm)',
                    'Apply potassium bicarbonate as preventive (5g/L)',
                    'Remove and destroy plant debris regularly',
                    'Use resistant varieties in next season'
                ],
                'organic': [
                    'Milk spray (1:9 milk:water ratio)',
                    'Garlic extract spray',
                    'Baking soda solution (1 tsp/L water)'
                ]
            },
            'Rust': {
                'name': 'Rust Disease',
                'description': 'Fungal disease with orange/red pustules on leaves',
                'immediate': [
                    'Remove and burn infected leaves (do not compost)',
                    'Apply mancozeb 75% WP @ 2.5g/L water',
                    'Spray copper oxychloride (3g/L) every 10 days',
                    'Use systemic fungicide (Propiconazole @ 1ml/L)',
                    'Repeat spray after 15 days'
                ],
                'preventive': [
                    'Practice crop rotation with non-host crops',
                    'Use rust-resistant varieties',
                    'Avoid waterlogging - ensure good drainage',
                    'Maintain balanced NPK nutrition',
                    'Monitor temperature and humidity'
                ],
                'organic': [
                    'Neem oil (5ml/L) + soap solution',
                    'Garlic-chili extract spray',
                    'Compost tea application'
                ]
            },
            'Healthy': {
                'name': 'Healthy Plant',
                'description': 'No disease detected - plant appears healthy',
                'immediate': [
                    'No treatment needed',
                    'Continue regular monitoring',
                    'Maintain current care routine'
                ],
                'preventive': [
                    'Ensure adequate water management',
                    'Maintain balanced fertilization (NPK)',
                    'Practice crop rotation annually',
                    'Keep field clean from weeds',
                    'Monitor for pest infestations'
                ],
                'organic': [
                    'Apply mulch to retain moisture',
                    'Use compost for soil health',
                    'Encourage beneficial insects'
                ]
            }
        }
        
        return treatments.get(disease, treatments['Healthy'])


disease_detection_service = DiseaseDetectionService()



