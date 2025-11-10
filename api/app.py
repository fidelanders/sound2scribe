from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import tempfile
import os
import sys
from pydub import AudioSegment  # We use pydub to validate the audio

app = Flask(__name__)
CORS(app)

# Print versions for debugging
print("Python version:", sys.version)
print("Whisper version:", whisper.__version__)

# Initialize model
model = None

def load_whisper_model():
    """Load Whisper model with compatibility handling"""
    global model
    try:
        print("Loading Whisper model...")

        # Try different model sizes
        model_sizes = ["base", "small", "tiny"]
        for model_size in model_sizes:
            try:
                print(f"Trying {model_size} model...")
                model = whisper.load_model(model_size)
                print(f"Successfully loaded {model_size} model!")
                return model
            except Exception as e:
                print(f"Failed to load {model_size} model: {e}")
                continue

        raise Exception("Could not load any Whisper model")

    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Load model at startup
try:
    load_whisper_model()
    print("Model loaded successfully during startup!")
except Exception as e:
    print(f"Warning: Could not load model at startup: {e}")
    model = None

# Function to validate audio file
def validate_audio(file_path):
    """Validate if the audio file can be loaded by pydub"""
    try:
        audio = AudioSegment.from_file(file_path)
        return audio
    except Exception as e:
        print(f"Invalid audio file: {e}")
        return None

@app.route('/upload', methods=['POST'])
def upload_file():
    if model is None:
        try:
            load_whisper_model()
        except Exception as e:
            return jsonify({'error': f'Model not available: {str(e)}'}), 500

    temp_path = None
    try:
        if 'audio_file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['audio_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Validate file size
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset seek position

        if file_size == 0:
            return jsonify({'error': 'File is empty'}), 400

        if file_size > 100 * 1024 * 1024:  # 100MB
            return jsonify({'error': 'File too large (max 100MB)'}), 400

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.audio') as tmp:
            file.save(tmp.name)
            temp_path = tmp.name

        # Validate the audio file before transcription
        audio = validate_audio(temp_path)
        if audio is None:
            return jsonify({'error': 'Invalid audio file'}), 400

        print(f"Transcribing file: {file.filename} ({file_size} bytes)")

        # Perform transcription with simple parameters
        result = model.transcribe(
            temp_path,
            fp16=False,  # Force FP32
            verbose=False
        )

        transcription = result['text'].strip()

        if not transcription:
            transcription = "No speech detected in audio"

        print("Transcription successful!")
        return jsonify({
            'text': transcription,
            'language': result.get('language', 'unknown')
        })

    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return jsonify({'error': f'Transcription failed: {str(e)}'}), 500

    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                print(f"Error deleting temp file: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    status = 'healthy' if model is not None else 'model not loaded'
    return jsonify({'status': status, 'model_loaded': model is not None})

if __name__ == '__main__':
    print("Starting server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
