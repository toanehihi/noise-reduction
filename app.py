import logging
from flask import Flask
from flask_cors import CORS

from config import get_config
from services import NoiseReductionService
from api import api_bp, init_api

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app(config_name='default'):
    # Initialize Flask app
    app = Flask(__name__)
    
    # Load configuration
    config_obj = get_config() if config_name == 'default' else get_config()
    app.config.from_object(config_obj)
    
    # Enable CORS
    CORS(app, origins=app.config['CORS_ORIGINS'])
    
    # Initialize service layer
    logger.info("Initializing Noise Reduction Service...")
    try:
        noise_service = NoiseReductionService(
            model_path=app.config['MODEL_PATH'],
            sample_rate=app.config['SAMPLE_RATE']
        )
        logger.info("Service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise
    
    # Initialize API layer with service
    init_api(noise_service, app.config)
    
    # Register blueprints
    app.register_blueprint(api_bp)
    
    logger.info("Application created successfully")
    return app


if __name__ == '__main__':
    # Create app
    app = create_app()
    
    # Start server
    logger.info(f"Starting Flask server on {app.config['HOST']}:{app.config['PORT']}")
    logger.info(f"Debug mode: {app.config['DEBUG']}")
    
    app.run(
        host=app.config['HOST'],
        port=app.config['PORT'],
        debug=app.config['DEBUG']
    )
