"""
Configuration module for Flask Noise Reduction Service
"""
import os
from pathlib import Path

# Base directory (noise-reduction-service package directory)
BASE_DIR = Path(__file__).parent

# Model configuration (model inside package)
MODEL_PATH = os.getenv(
    'MODEL_PATH', 
    str(BASE_DIR / 'models' / 'DTLN_vivos_best.h5')
)

# Audio parameters (matching DTLN model training config)
SAMPLE_RATE = 16000  # 16kHz
BLOCK_LEN = 512
BLOCK_SHIFT = 128

# Flask configuration
class Config:
    """Base configuration"""
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size
    UPLOAD_FOLDER = Path(__file__).parent / 'uploads'
    ALLOWED_EXTENSIONS = {'wav'}
    
    # Model settings
    MODEL_PATH = MODEL_PATH
    SAMPLE_RATE = SAMPLE_RATE
    BLOCK_LEN = BLOCK_LEN
    BLOCK_SHIFT = BLOCK_SHIFT
    
    # CORS settings for ESP32
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
    
    # Server settings
    HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    PORT = int(os.getenv('FLASK_PORT', 5000))
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    # In production, set proper secret key
    SECRET_KEY = os.getenv('SECRET_KEY')

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get configuration based on environment"""
    env = os.getenv('FLASK_ENV', 'development')
    return config.get(env, config['default'])
