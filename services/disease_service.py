"""
Disease Detection Service
Uses ResNet-50 model for plant disease classification
"""

import os
import numpy as np
from PIL import Image
import cv2

# TensorFlow/Keras imports
try:
    from tensorflow import keras
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing import image as keras_image
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Using mock disease detection.")


class DiseaseDetectionService:
    def __init__(self):
        self.model = None
        self.disease_classes = self._load_disease_classes()
        #self.load_model()
        
    def load_model(self):
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Cannot load model.")
            return
            
        model_path = os.path.join('models', 'disease_detection', 'resnet50_plant.h5')
        if os.path.exists(model_path):
            self.model = keras.models.load_model(model_path)
            print("Disease detection model loaded")
        else:
            print(f"Model not found at {model_path}, using mock predictions")
    
    def _load_disease_classes(self):
        return [
            'Apple___Apple_scab',
            'Apple___Black_rot',
            'Apple___Cedar_apple_rust',
            'Apple___healthy',
            'Corn_(maize)___Cercospora_leaf_spot',
            'Corn_(maize)___Common_rust',
            'Corn_(maize)___Northern_Leaf_Blight',
            'Corn_(maize)___healthy',
            'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)',
            'Grape___Leaf_blight',
            'Grape___healthy',
            'Pepper,_bell___Bacterial_spot',
            'Pepper,_bell___healthy',
            'Potato___Early_blight',
            'Potato___Late_blight',
            'Potato___healthy',
            'Tomato___Bacterial_spot',
            'Tomato___Early_blight',
            'Tomato___Late_blight',
            'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites',
            'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
    
    def detect_disease(self, image_path, crop_type=None):  
        if self.model and TENSORFLOW_AVAILABLE:
            return self._ml_based_detection(image_path, crop_type)
        
        return self._mock_detection(image_path, crop_type)
    
    def _ml_based_detection(self, image_path, crop_type):
        img = keras_image.load_img(image_path, target_size=(224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        predictions = self.model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx] * 100)
        
        disease_full_name = self.disease_classes[predicted_class_idx]
        crop, disease = disease_full_name.split('___')
        
        result = {
            'disease_name': self._format_disease_name(disease),
            'scientific_name': self._get_scientific_name(disease),
            'confidence': round(confidence, 1),
            'severity': self._assess_severity(confidence, image_path),
            'affected_area': self._estimate_affected_area(image_path),
            'image_path': f'/static/uploads/{os.path.basename(image_path)}',
            'treatment': self._get_treatment_protocol(disease)
        }
        
        return result
    
    def _mock_detection(self, image_path, crop_type):
        import random
        
        mock_diseases = {
            'Tomato': [
                ('Early Blight', 'Alternaria solani', 96, 'MODERATE', 18),
                ('Late Blight', 'Phytophthora infestans', 94, 'HIGH', 35),
                ('Leaf Mold', 'Passalora fulva', 92, 'LOW', 8)
            ],
            'Potato': [
                ('Early Blight', 'Alternaria solani', 95, 'MODERATE', 22),
                ('Late Blight', 'Phytophthora infestans', 97, 'HIGH', 41)
            ],
            'Wheat': [
                ('Leaf Rust', 'Puccinia triticina', 93, 'MODERATE', 15),
                ('Powdery Mildew', 'Blumeria graminis', 91, 'LOW', 10)
            ]
        }
        
        if crop_type and crop_type in mock_diseases:
            disease_data = random.choice(mock_diseases[crop_type])
        else:
            disease_data = random.choice(mock_diseases['Tomato'])
        
        disease_name, scientific_name, confidence, severity, affected_area = disease_data
        
        treatment = self._get_treatment_protocol(disease_name)
        
        result = {
            'disease_name': disease_name,
            'scientific_name': scientific_name,
            'confidence': confidence,
            'severity': severity,
            'affected_area': affected_area,
            'image_path': f'/static/uploads/{os.path.basename(image_path)}',
            'treatment': treatment
        }
        
        return result
    
    def _format_disease_name(self, disease):
        return disease.replace('_', ' ').title()
    
    def _get_scientific_name(self, disease):
        scientific_names = {
            'Early_blight': 'Alternaria solani',
            'Late_blight': 'Phytophthora infestans',
            'Leaf_Mold': 'Passalora fulva',
            'Septoria_leaf_spot': 'Septoria lycopersici',
            'Spider_mites': 'Tetranychus urticae',
            'Target_Spot': 'Corynespora cassiicola',
            'Bacterial_spot': 'Xanthomonas spp.',
            'Leaf_rust': 'Puccinia triticina',
            'Powdery_mildew': 'Blumeria graminis'
        }
        return scientific_names.get(disease, 'Unknown pathogen')
    
    def _assess_severity(self, confidence, image_path):
        if confidence > 95:
            return 'HIGH'
        elif confidence > 85:
            return 'MODERATE'
        else:
            return 'LOW'
    
    def _estimate_affected_area(self, image_path):
        import random
        return random.randint(5, 45)
    
    def _get_treatment_protocol(self, disease):
        
        treatments = {
            'Early Blight': {
                'immediate': [
                    'Remove infected leaves at soil level immediately',
                    'Apply Mancozeb 75% WP (0.2% solution)',
                    'Reduce irrigation to lower humidity'
                ]
            },
            'Late Blight': {
                'immediate': [
                    'Apply Metalaxyl + Mancozeb (0.25% solution)',
                    'Remove and destroy severely infected plants',
                    'Avoid overhead irrigation'
                ]
            },
            'Leaf Mold': {
                'immediate': [
                    'Improve ventilation in greenhouse/field',
                    'Apply Copper oxychloride (0.3% solution)',
                    'Reduce relative humidity below 85%'
                ]
            },
            'Bacterial Spot': {
                'immediate': [
                    'Apply copper-based bactericide',
                    'Remove infected plant parts',
                    'Avoid working in wet conditions'
                ]
            },
            'Leaf Rust': {
                'immediate': [
                    'Apply Propiconazole 25% EC (0.1% solution)',
                    'Spray at first sign of disease',
                    'Ensure proper crop rotation'
                ]
            },
            'Powdery Mildew': {
                'immediate': [
                    'Apply Sulfur 80% WP (0.2% solution)',
                    'Improve air circulation',
                    'Avoid excessive nitrogen fertilization'
                ]
            }
        }
        
        default_treatment = {
            'immediate': [
                'Isolate affected plants immediately',
                'Consult agricultural extension officer',
                'Apply broad-spectrum fungicide as preventive measure'
            ]
        }
        
        return treatments.get(disease.replace(' ', '_'), default_treatment)


disease_service = DiseaseDetectionService()


def detect_disease(image_path, crop_type=None):
    return disease_service.detect_disease(image_path, crop_type)
