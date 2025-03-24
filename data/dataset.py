import os
import glob
import numpy as np
import librosa
from data.preprocessing import extract_features

class SoundDataset:
    def __init__(self, normal_dir: str, abnormal_dir: str = None, config: dict = None):
        """
        Initialize the dataset with directories for normal (and optional abnormal) sounds.
        normal_dir: path to directory containing normal sound files (e.g., .wav).
        abnormal_dir: path to directory containing anomalous sound files (optional).
        config: configuration dictionary for feature extraction parameters.
        """
        self.normal_dir = normal_dir
        self.abnormal_dir = abnormal_dir
        # Gather all file paths
        self.normal_files = sorted(glob.glob(os.path.join(normal_dir, "*.wav"))) if normal_dir else []
        self.abnormal_files = sorted(glob.glob(os.path.join(abnormal_dir, "*.wav"))) if abnormal_dir else []
        # Feature extraction parameters from config
        if config is None:
            config = {}
        model_cfg = config.get('model', {})
        self.n_mels = model_cfg.get('n_mels', 64)
        self.frames = model_cfg.get('frames', 5)
        self.n_fft = model_cfg.get('n_fft', 1024)
        self.hop_length = model_cfg.get('hop_length', 512)
    
    def load_data(self):
        """
        Load all audio files and return feature vectors and labels.
        Returns:
            X: numpy array of shape (num_samples, n_mels*frames) with extracted features.
            y: numpy array of labels (0 for normal, 1 for anomaly) or None if no anomaly data.
        """
        X_list = []
        y_list = []
        # Process normal files
        for file_path in self.normal_files:
            signal, sr = librosa.load(file_path, sr=None)  # Load with original sampling rate
            features = extract_features(signal, sr, n_mels=self.n_mels, frames=self.frames,
                                        n_fft=self.n_fft, hop_length=self.hop_length)
            if features.size == 0:
                continue  # skip if feature extraction returned empty (signal too short)
            X_list.append(features)
            # Label 0 for each feature window from a normal file
            y_list.append(np.zeros(features.shape[0], dtype=np.int32))
        # Process abnormal files (if any)
        for file_path in self.abnormal_files:
            signal, sr = librosa.load(file_path, sr=None)
            features = extract_features(signal, sr, n_mels=self.n_mels, frames=self.frames,
                                        n_fft=self.n_fft, hop_length=self.hop_length)
            if features.size == 0:
                continue
            X_list.append(features)
            # Label 1 for each feature window from an abnormal file
            y_list.append(np.ones(features.shape[0], dtype=np.int32))
        if len(X_list) == 0:
            return np.array([]), None
        # Concatenate all feature windows from all files
        X = np.vstack(X_list)
        y = np.concatenate(y_list) if y_list else None
        return X, y
