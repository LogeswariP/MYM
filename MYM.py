import os
import wave
import librosa
import numpy as np
import tensorflow as tf
import asyncio
import json
import logging
import tempfile
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
import google.generativeai as gg
import warnings
from flask import Flask, render_template, request, jsonify
import base64

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Suppress warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default_secret_key")

# API Keys
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
GG_API_KEY = os.environ.get("GG_API_KEY", "")

# Configure Google Generative AI
gg.configure(api_key=GG_API_KEY)
generation_config = {
    'temperature': 0.7,
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 700
}
sys_ins = """
You are a smart, emotionally intuitive, and slightly humorous virtual therapist-best friend. People come to you to process their emotions in a world that often feels too heavy.
Your job is to:
1. Summarize their emotional state in a kind, human-like sentence.
3. Give a supportive, encouraging message based on their feelings, something that helps them feel understood and a little better.
4. Respond in a warm, friendly, and slightly playful tone, like their most caring and emotionally aware best friend.
Keep your language simple, empathetic, and relatable. Use gentle humor if it fits, but never at the cost of emotional safety.
"""
safety_settings = [
    {"category": 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_NONE'},
    {"category": 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_NONE'},
    {"category": 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'},
    {"category": 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_NONE'}
]

model = gg.GenerativeModel(
    'gemini-2.0-flash',
    generation_config=generation_config,
    safety_settings=safety_settings,
    system_instruction=sys_ins
)

# Load TensorFlow model at startup to avoid repeated loading
try:
    emotion_model = tf.keras.models.load_model("emotion_model(93).h5")
    app.logger.info("Emotion model loaded successfully")
     # 🔍 DEBUG: Print the input shape to verify compatibility
    print(" Model input shape:", emotion_model.input_shape)
except Exception as e:
    app.logger.error(f"Error loading emotion model: {e}")
    emotion_model = None

def extract_features(file_path, n_mfcc=40, max_pad_len=130):
    """Extract MFCC features and reshape to (1, 130, 40, 1) for CNN model"""
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)

        # Transpose MFCCs to shape (time, features) → (130, 40)
        mfccs = mfccs.T

        # Pad or trim to 130 time steps
        if mfccs.shape[0] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[0]
            mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mfccs = mfccs[:max_pad_len, :]

        # Final shape: (1, 130, 40, 1)
        return np.expand_dims(mfccs, axis=(0, -1))

    except Exception as e:
        app.logger.error(f"Error extracting features: {e}")
        return None


'''def extract_features(file_path, max_pad_len=174):
    """Extract MFCC features from audio file for emotion analysis"""
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return np.array(mfccs).reshape(1, 40, 174, 1)  # Reshape for CNN input
    except Exception as e:
        app.logger.error(f"Error extracting features: {e}")
        return None
'''
def predict_emotion(file_path):
    """Predict emotion from audio file"""
    try:
        if emotion_model is None:
            return "Model not loaded", 0.0
            
        # Extract features
        features = extract_features(file_path)
        if features is None:
            return "Feature extraction failed", 0.0
            
        # Predict emotion
        prediction = emotion_model.predict(features)
        predicted_label = np.argmax(prediction)
        emotions = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]
        return emotions[predicted_label], float(np.max(prediction))
    except Exception as e:
        app.logger.error(f"Error in emotion prediction: {e}")
        return "Error", 0.0

async def transcribe_audio(file_path):
    """Transcribe audio using Deepgram API"""
    try:
        # Initialize Deepgram SDK
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)

        # Open the audio file
        with open(file_path, "rb") as file:
            buffer_data = file.read()

        payload: FileSource = {"buffer": buffer_data}

        # Configure Deepgram options
        options = PrerecordedOptions(model="nova-2", smart_format=True)

        # Transcribe the file
        response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)
        response_json = json.loads(response.to_json(indent=4))
        transcript = response_json["results"]["channels"][0]["alternatives"][0]["transcript"]
        return transcript
    except Exception as e:
        app.logger.error(f"Exception during transcription: {e}")
        return None

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/process-audio', methods=['POST'])
def process_audio():
    """Process the audio data and return results"""
    try:
        # Get audio data from the request
        audio_data = request.json.get('audio')
        if not audio_data:
            return jsonify({'error': 'No audio data received'}), 400
            
        # Decode the base64 audio data
        audio_binary = base64.b64decode(audio_data.split(',')[1])
        
        # Create a temporary file to save the audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file_path = temp_file.name
            
            # Save audio to the temporary file
            with wave.open(temp_file_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit (2 bytes)
                wf.setframerate(44100)
                wf.writeframes(audio_binary)
        
        # Process the audio
        emotion, confidence = predict_emotion(temp_file_path)
        transcript = asyncio.run(transcribe_audio(temp_file_path))
        
        if not transcript:
            return jsonify({'error': 'Could not transcribe audio'}), 500
            
        # Generate AI response
        user_prompt = f"Transcript: {transcript}"
        convo = model.start_chat()
        ai_response = convo.send_message(user_prompt).text.strip().replace("**","")
        
        # Clean up the temporary file
        os.unlink(temp_file_path)
        
        # Return results
        return jsonify({
            'emotion': emotion,
            'confidence': float(confidence),
            'transcript': transcript,
            'ai_response': ai_response
        })
        
    except Exception as e:
        app.logger.error(f"Error processing audio: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
