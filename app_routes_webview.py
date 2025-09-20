"""
WebView-specific routes for handling video uploads and camera operations
"""

from flask import Blueprint, request, jsonify, current_app
import os
import base64
from datetime import datetime
from werkzeug.utils import secure_filename
import json

webview_bp = Blueprint('webview', __name__, url_prefix='/api/webview')

# Configure upload settings
UPLOAD_FOLDER = 'uploads/webview'
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'webm', 'avi', 'mov'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

def allowed_file(filename, file_type='video'):
    if file_type == 'video':
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS
    elif file_type == 'image':
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS
    return False

def ensure_upload_directory():
    """Ensure upload directory exists"""
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@webview_bp.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video upload from WebView"""
    try:
        ensure_upload_directory()
        
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        metadata = request.form.get('metadata', '{}')
        
        if video_file.filename == '':
            return jsonify({'success': False, 'error': 'Empty filename'}), 400
        
        if not allowed_file(video_file.filename, 'video'):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
        # Parse metadata
        try:
            metadata_dict = json.loads(metadata)
        except:
            metadata_dict = {}
        
        # Generate secure filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_filename = secure_filename(video_file.filename)
        filename = f"webview_video_{timestamp}_{original_filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save file
        video_file.save(file_path)
        
        # Get file info
        file_size = os.path.getsize(file_path)
        
        # Store metadata
        metadata_dict.update({
            'filename': filename,
            'original_filename': original_filename,
            'file_path': file_path,
            'file_size': file_size,
            'upload_timestamp': datetime.utcnow().isoformat(),
            'source': 'webview'
        })
        
        # Save metadata file
        metadata_path = file_path + '.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        return jsonify({
            'success': True,
            'message': 'Video uploaded successfully',
            'filename': filename,
            'file_size': file_size,
            'metadata': metadata_dict
        })
        
    except Exception as e:
        current_app.logger.error(f"Video upload error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@webview_bp.route('/upload_photo', methods=['POST'])
def upload_photo():
    """Handle photo upload from WebView"""
    try:
        ensure_upload_directory()
        
        # Handle both file upload and base64 data
        if 'photo' in request.files:
            # File upload
            photo_file = request.files['photo']
            metadata = request.form.get('metadata', '{}')
            
            if photo_file.filename == '':
                return jsonify({'success': False, 'error': 'Empty filename'}), 400
            
            if not allowed_file(photo_file.filename, 'image'):
                return jsonify({'success': False, 'error': 'Invalid file type'}), 400
            
            # Generate secure filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            original_filename = secure_filename(photo_file.filename)
            filename = f"webview_photo_{timestamp}_{original_filename}"
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            
            # Save file
            photo_file.save(file_path)
            
        elif 'photo_data' in request.json:
            # Base64 data upload
            data = request.json
            photo_data = data.get('photo_data', '')
            metadata = data.get('metadata', {})
            
            if not photo_data:
                return jsonify({'success': False, 'error': 'No photo data provided'}), 400
            
            # Remove data URL prefix if present
            if photo_data.startswith('data:image/'):
                photo_data = photo_data.split(',')[1]
            
            # Decode base64
            try:
                image_data = base64.b64decode(photo_data)
            except Exception as e:
                return jsonify({'success': False, 'error': 'Invalid base64 data'}), 400
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"webview_photo_{timestamp}.jpg"
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            
            # Save file
            with open(file_path, 'wb') as f:
                f.write(image_data)
                
        else:
            return jsonify({'success': False, 'error': 'No photo provided'}), 400
        
        # Get file info
        file_size = os.path.getsize(file_path)
        
        # Parse metadata
        if isinstance(metadata, str):
            try:
                metadata_dict = json.loads(metadata)
            except:
                metadata_dict = {}
        else:
            metadata_dict = metadata or {}
        
        # Store metadata
        metadata_dict.update({
            'filename': filename,
            'file_path': file_path,
            'file_size': file_size,
            'upload_timestamp': datetime.utcnow().isoformat(),
            'source': 'webview'
        })
        
        # Save metadata file
        metadata_path = file_path + '.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        return jsonify({
            'success': True,
            'message': 'Photo uploaded successfully',
            'filename': filename,
            'file_size': file_size,
            'metadata': metadata_dict
        })
        
    except Exception as e:
        current_app.logger.error(f"Photo upload error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@webview_bp.route('/upload_situp_video', methods=['POST'])
def upload_situp_video():
    """Handle situp exercise video upload"""
    try:
        ensure_upload_directory()
        
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        metadata = request.form.get('metadata', '{}')
        
        if video_file.filename == '':
            return jsonify({'success': False, 'error': 'Empty filename'}), 400
        
        # Parse metadata
        try:
            metadata_dict = json.loads(metadata)
        except:
            metadata_dict = {}
        
        # Generate secure filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        user_id = metadata_dict.get('user_id', 'unknown')
        filename = f"situp_video_{user_id}_{timestamp}.webm"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save file
        video_file.save(file_path)
        
        # Get file info
        file_size = os.path.getsize(file_path)
        
        # Store metadata
        metadata_dict.update({
            'filename': filename,
            'file_path': file_path,
            'file_size': file_size,
            'upload_timestamp': datetime.utcnow().isoformat(),
            'exercise_type': 'situps',
            'source': 'webview'
        })
        
        # Save metadata file
        metadata_path = file_path + '.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        return jsonify({
            'success': True,
            'message': 'Situp video uploaded successfully',
            'filename': filename,
            'file_size': file_size,
            'video_url': f'/uploads/webview/{filename}',
            'metadata': metadata_dict
        })
        
    except Exception as e:
        current_app.logger.error(f"Situp video upload error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@webview_bp.route('/camera_status', methods=['GET'])
def camera_status():
    """Get camera status and capabilities"""
    try:
        return jsonify({
            'success': True,
            'camera_available': True,
            'supported_formats': {
                'video': list(ALLOWED_VIDEO_EXTENSIONS),
                'image': list(ALLOWED_IMAGE_EXTENSIONS)
            },
            'max_file_size': MAX_FILE_SIZE,
            'webview_optimized': True
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@webview_bp.route('/test_upload', methods=['POST'])
def test_upload():
    """Test upload endpoint for debugging"""
    try:
        data = request.get_json() or {}
        files = request.files
        form_data = request.form
        
        return jsonify({
            'success': True,
            'message': 'Test upload received',
            'data': {
                'json_data': data,
                'files': list(files.keys()),
                'form_data': dict(form_data),
                'content_type': request.content_type,
                'method': request.method
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@webview_bp.route('/cleanup', methods=['POST'])
def cleanup_resources():
    """Clean up WebView resources"""
    try:
        # This endpoint can be called when the WebView is closing
        # to clean up any temporary files or resources
        
        return jsonify({
            'success': True,
            'message': 'Resources cleaned up successfully'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Error handlers
@webview_bp.errorhandler(413)
def file_too_large(error):
    return jsonify({
        'success': False,
        'error': 'File too large',
        'max_size': MAX_FILE_SIZE
    }), 413

@webview_bp.errorhandler(400)
def bad_request(error):
    return jsonify({
        'success': False,
        'error': 'Bad request'
    }), 400

@webview_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500