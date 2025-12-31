import os

class Config:    
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'agrismart-dev-secret-key-2025'
    
    DATABASE_PATH = os.path.join(BASE_DIR, 'data', 'agrismart.db')
    
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    
    # ML Model paths
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    CROP_MODEL_PATH = os.path.join(MODELS_DIR, 'crop_recommendation', 'crop_model.pkl')
    DISEASE_MODEL_PATH = os.path.join(MODELS_DIR, 'disease_detection', 'resnet50_plant.h5')
    PROPHET_MODELS_DIR = os.path.join(MODELS_DIR, 'price_forecasting', 'prophet_models')
    
    # API Keys (use environment variables in production)
    AGMARKNET_API_KEY = os.environ.get('AGMARKNET_API_KEY') or 'mock_key'
    WEATHER_API_KEY = os.environ.get('WEATHER_API_KEY') or 'mock_key'
    
    # Application settings
    DEBUG = True
    PORT = 5000
    HOST = '0.0.0.0'
