import librosa
import numpy as np
def extract_features(file_path):
    """
    Audio file nunchi MFCC + Chroma + Mel features extract chestham
    """
    try:
        # Audio load cheyyi - 3 seconds ki trim cheyyi
        y, sr = librosa.load(file_path, duration=3, offset=0.5)

        # 1. MFCC - 40 features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)

        # 2. Chroma - 12 features  
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)

        # 3. Mel Spectrogram - 128 features
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_mean = np.mean(mel.T, axis=0)

        # 4. Spectral Contrast - 7 features
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast.T, axis=0)

        # 5. Tonnetz - 6 features
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        tonnetz_mean = np.mean(tonnetz.T, axis=0)

        # Anni features combine cheyyi
        features = np.hstack([mfccs_mean, chroma_mean, mel_mean, contrast_mean, tonnetz_mean])

        return features

    except Exception as e:
        print(f"Error extracting features: {e}")
=======
import librosa
import numpy as np

def extract_features(file_path):
    """
    Audio file nunchi MFCC + Chroma + Mel features extract chestham
    """
    try:
        # Audio load cheyyi - 3 seconds ki trim cheyyi
        y, sr = librosa.load(file_path, duration=3, offset=0.5)

        # 1. MFCC - 40 features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)

        # 2. Chroma - 12 features  
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)

        # 3. Mel Spectrogram - 128 features
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_mean = np.mean(mel.T, axis=0)

        # 4. Spectral Contrast - 7 features
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast.T, axis=0)

        # 5. Tonnetz - 6 features
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        tonnetz_mean = np.mean(tonnetz.T, axis=0)

        # Anni features combine cheyyi
        features = np.hstack([mfccs_mean, chroma_mean, mel_mean, contrast_mean, tonnetz_mean])

        return features

    except Exception as e:
        print(f"Error extracting features: {e}")
        return None
