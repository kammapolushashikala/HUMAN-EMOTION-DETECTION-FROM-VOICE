<<<<<<< HEAD
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
from utils import extract_features

# RAVDESS dataset path
DATA_PATH = 'data/RAVDESS'

# Emotion mapping - 3rd part of RAVDESS filename is emotion code
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Convert emotion name to number
emotion_to_num = {v: k for k, v in enumerate(emotion_map.values())}

def load_data():
    """
    Load all audio files from RAVDESS dataset and extract features
    """
    X = []
    y = []

    print("📂 Loading data...")

    # Loop through all actor folders in data/RAVDESS
    for actor_folder in os.listdir(DATA_PATH):
        actor_path = os.path.join(DATA_PATH, actor_folder)

        if not os.path.isdir(actor_path):
            continue

        print(f"Processing {actor_folder}...")

        # Loop through all wav files in each actor folder
        for file in os.listdir(actor_path):
            if file.endswith('.wav'):
                file_path = os.path.join(actor_path, file)

                try:
                    # Filename format: 03-01-06-01-02-01-12.wav
                    # 3rd part is emotion code: 06 = fearful
                    emotion_code = file.split('-')[2]
                    emotion = emotion_map[emotion_code]

                    # Extract features
                    features = extract_features(file_path)

                    X.append(features)
                    y.append(emotion_to_num[emotion])

                except Exception as e:
                    print(f"Error with {file}: {e}")
                    continue

    return np.array(X), np.array(y)

def main():
    print("🚀 Starting training...")

    # Load data
    X, y = load_data()

    if len(X) == 0:
        print("❌ No data found. Check if.wav files exist in data/RAVDESS folder")
        return

    print(f"✅ Total samples: {len(X)}")
    print(f"Feature shape: {X[0].shape}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Feature scaling - very important for SVM
    print("📊 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM model
    print("🤖 Training SVM model...")
    model = SVC(kernel='rbf', C=10, gamma=0.01, probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Test accuracy
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=list(emotion_map.values())))

    # Create models folder if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Save model
    joblib.dump(model, 'models/emotion_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    print("\n💾 Model saved: models/emotion_model.pkl")
    print("💾 Scaler saved: models/scaler.pkl")
    print("\n🎉 Training complete! Now you can run the app with: streamlit run app.py")

if __name__ == '__main__':
=======
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
from utils import extract_features

# RAVDESS dataset path
DATA_PATH = 'data/RAVDESS'

# Emotion mapping - 3rd part of RAVDESS filename is emotion code
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Convert emotion name to number
emotion_to_num = {v: k for k, v in enumerate(emotion_map.values())}

def load_data():
    """
    Load all audio files from RAVDESS dataset and extract features
    """
    X = []
    y = []

    print("📂 Loading data...")

    # Loop through all actor folders in data/RAVDESS
    for actor_folder in os.listdir(DATA_PATH):
        actor_path = os.path.join(DATA_PATH, actor_folder)

        if not os.path.isdir(actor_path):
            continue

        print(f"Processing {actor_folder}...")

        # Loop through all wav files in each actor folder
        for file in os.listdir(actor_path):
            if file.endswith('.wav'):
                file_path = os.path.join(actor_path, file)

                try:
                    # Filename format: 03-01-06-01-02-01-12.wav
                    # 3rd part is emotion code: 06 = fearful
                    emotion_code = file.split('-')[2]
                    emotion = emotion_map[emotion_code]

                    # Extract features
                    features = extract_features(file_path)

                    X.append(features)
                    y.append(emotion_to_num[emotion])

                except Exception as e:
                    print(f"Error with {file}: {e}")
                    continue

    return np.array(X), np.array(y)

def main():
    print("🚀 Starting training...")

    # Load data
    X, y = load_data()

    if len(X) == 0:
        print("❌ No data found. Check if.wav files exist in data/RAVDESS folder")
        return

    print(f"✅ Total samples: {len(X)}")
    print(f"Feature shape: {X[0].shape}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Feature scaling - very important for SVM
    print("📊 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM model
    print("🤖 Training SVM model...")
    model = SVC(kernel='rbf', C=10, gamma=0.01, probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Test accuracy
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=list(emotion_map.values())))

    # Create models folder if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Save model
    joblib.dump(model, 'models/emotion_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    print("\n💾 Model saved: models/emotion_model.pkl")
    print("💾 Scaler saved: models/scaler.pkl")
    print("\n🎉 Training complete! Now you can run the app with: streamlit run app.py")

if __name__ == '__main__':
>>>>>>> f0032836680bce774295b1f086492dee22fc91a6
    main()