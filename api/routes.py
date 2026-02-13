import logging
from pathlib import Path
from flask import Blueprint, request, send_file, jsonify
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

# Create blueprint
api_bp = Blueprint('api', __name__)

# Global service instance (injected from app.py)
_noise_service = None


def init_api(noise_service, config):
    """
    Initialize API with service instance and config
    
    Args:
        noise_service: Instance of NoiseReductionService
        config: Flask config object
    """
    global _noise_service, _config
    _noise_service = noise_service
    _config = config
    
    # Create upload folder if needed
    config['UPLOAD_FOLDER'].mkdir(exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed"""
    allowed_extensions = _config.get('ALLOWED_EXTENSIONS', {'wav'})
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions


@api_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    
    Returns service status and model information
    """
    try:
        service_info = _noise_service.get_info()
        
        response = {
            'status': 'healthy' if service_info['status'] == 'ready' else 'unhealthy',
            'service': service_info
        }
        
        status_code = 200 if service_info['status'] == 'ready' else 503
        return jsonify(response), status_code
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@api_bp.route('/denoise', methods=['POST'])
def denoise_audio():
    """
    Audio denoising endpoint
    
    Accepts:
        POST multipart/form-data with 'file' field containing .wav audio
    
    Returns:
        Denoised audio file (.wav) on success
        JSON error message on failure
    """
    input_path = None
    output_path = None
    
    try:
        # Validate request has file
        if 'file' not in request.files:
            logger.warning("No file part in request")
            return jsonify({'error': 'No file part in the request'}), 400
        
        file = request.files['file']
        
        # Validate filename not empty
        if file.filename == '':
            logger.warning("Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file extension
        if not allowed_file(file.filename):
            logger.warning(f"Invalid file type: {file.filename}")
            return jsonify({
                'error': 'Invalid file type. Only .wav files are allowed'
            }), 400
        
        # Secure the filename
        filename = secure_filename(file.filename)
        
        # Generate paths
        upload_folder = _config['UPLOAD_FOLDER']
        input_path = upload_folder / f'input_{filename}'
        output_path = upload_folder / f'denoised_{filename}'
        
        # Save uploaded file
        logger.info(f"Saving uploaded file: {filename}")
        file.save(str(input_path))
        
        # Process audio using service
        logger.info(f"Processing audio: {filename}")
        _noise_service.process_audio(str(input_path), str(output_path))
        
        # Send denoised file back
        logger.info(f"Sending denoised file: {filename}")
        
        response = send_file(
            str(output_path),
            mimetype='audio/wav',
            as_attachment=True,
            download_name=f'denoised_{filename}'
        )
        
        # Clean up files after sending
        @response.call_on_close
        def cleanup():
            try:
                if input_path and input_path.exists():
                    input_path.unlink()
                    logger.info(f"Cleaned up input file: {input_path}")
                if output_path and output_path.exists():
                    output_path.unlink()
                    logger.info(f"Cleaned up output file: {output_path}")
            except Exception as e:
                logger.error(f"Error cleaning up files: {e}")
        
        return response
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return jsonify({
            'error': 'File not found',
            'message': str(e)
        }), 404
        
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return jsonify({
            'error': 'Invalid audio file',
            'message': str(e)
        }), 400
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        
        # Clean up files on error
        try:
            if input_path and input_path.exists():
                input_path.unlink()
            if output_path and output_path.exists():
                output_path.unlink()
        except:
            pass
        
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@api_bp.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    logger.warning("File too large")
    max_size_mb = _config.get('MAX_CONTENT_LENGTH', 0) / (1024 * 1024)
    return jsonify({
        'error': 'File too large',
        'max_size_mb': max_size_mb
    }), 413
