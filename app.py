import streamlit as st
import pandas as pd
import librosa
import numpy as np
import joblib
import os
import tempfile
from utils import extract_features

# Page setup
st.set_page_config(page_title="Emotion Detector", page_icon="🎤", layout="centered")

st.title("🎤 Speech Emotion Detector")
st.write("Upload an audio file to detect the emotion")

# Load model and scaler
@st.cache(allow_output_mutation=True)
def load_model():
    try:
        model = joblib.load('models/emotion_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except:
        st.error("❌ models/emotion_model.pkl or scaler.pkl not found. Run train.py first.")
        return None, None

model, scaler = load_model()

# Emotion labels - from RAVDESS dataset
emotion_dict = {
    0: 'neutral',
    1: 'calm',
    2: 'happy',
    3: 'sad',
    4: 'angry',
    5: 'fearful',
    6: 'disgust',
    7: 'surprised'
}

# Emoji mapping for emotions
emoji_dict = {
    'neutral': '😐',
    'calm': '😌',
    'happy': '😊',
    'sad': '😢',
    'angry': '😠',
    'fearful': '😨',
    'disgust': '🤢',
    'surprised': '😲'
}

if model and scaler:
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload WAV/MP3 file",
        type=['wav', 'mp3', 'flac', 'ogg']
    )

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        # Predict button
        if st.button('🔍 Predict Emotion'):
            with st.spinner('Processing audio...'):
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                try:
                    # Extract features
                    features = extract_features(tmp_path)
                    features = np.array(features).reshape(1, -1)

                    # Scale features
                    features_scaled = scaler.transform(features)

                    # Prediction
                    prediction = model.predict(features_scaled)[0]
                    proba = model.predict_proba(features_scaled)[0]

                    emotion = emotion_dict[prediction]
                    confidence = np.max(proba) * 100

                    # Display result
                    st.success("✅ Prediction Complete!")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted Emotion", f"{emoji_dict[emotion]} {emotion.capitalize()}")
                    with col2:
                        st.metric("Confidence", f"{confidence:.2f}%")

                    # Show all probabilities as bar chart
                    st.subheader("All Emotion Probabilities:")
                    prob_data = {emotion_dict[i]: proba[i] for i in range(len(proba))}

                    # Convert to DataFrame for Streamlit 1.11.1
                    probs_df = pd.DataFrame.from_dict(prob_data, orient='index', columns=['Probability'])
                    probs_df = probs_df.sort_values('Probability', ascending=False)
                    st.bar_chart(probs_df)

                except Exception as e:
                    st.error(f"Error: {str(e)}")

                finally:
                    # Delete temp file
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
else:
    st.warning("⚠️ Run `python train.py` first to train the model")

st.markdown("---")
st.caption("Built with Librosa + Scikit-learn + Streamlit")