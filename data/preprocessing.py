import numpy as np
import librosa

def extract_features(signal: np.ndarray, sr: int, n_mels: int = 64, frames: int = 5,
                     n_fft: int = 1024, hop_length: int = 512) -> np.ndarray:
    """
    Extract log-mel spectrogram features from an audio signal and create sliding window feature vectors.
    signal: 1D numpy array of audio samples.
    sr: sample rate of the audio.
    Returns a numpy array of shape (num_windows, n_mels*frames).
    """
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    # Convert power spectrogram to decibel (log scale)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    # Determine number of frames we can slice out
    num_frames = log_mel_spec.shape[1]
    if num_frames < frames:
        # Not enough frames for even one sliding window
        return np.empty((0, n_mels * frames), dtype=np.float32)
    features_vector_size = num_frames - frames + 1
    dims = n_mels * frames
    # Initialize feature array
    features = np.zeros((features_vector_size, dims), dtype=np.float32)
    # Slide window across the time dimension and flatten
    for t in range(frames):
        features[:, t * n_mels:(t + 1) * n_mels] = log_mel_spec[:, t:t + features_vector_size].T
    return features
