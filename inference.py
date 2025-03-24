import yaml
import numpy as np
from data.dataset import SoundDataset
from models.ae_model import AEModel
from utils.logger import get_logger

if __name__ == "__main__":
    logger = get_logger("inference")
    logger.info("Loading configuration...")
    with open("configs/base.yaml", "r") as f:
        config = yaml.safe_load(f)
    # Load the trained model
    model = AEModel(config)
    model_path = config['paths']['model_save_path']
    model.load_model(model_path)
    logger.info(f"Loaded model from {model_path}")
    # Prepare test data
    test_normal_dir = config['data']['test_normal_dir']
    test_abnormal_dir = config['data'].get('test_abnormal_dir')
    logger.info("Loading test dataset...")
    test_dataset = SoundDataset(test_normal_dir, abnormal_dir=test_abnormal_dir, config=config)
    X_test, y_test = test_dataset.load_data()
    logger.info(f"Test data loaded: {X_test.shape[0]} samples")
    # Run the model to get reconstructions
    reconstructions = model.predict(X_test)
    # Compute reconstruction error for each sample (mean squared error per sample)
    errors = np.mean(np.square(reconstructions - X_test), axis=1)
    # Determine threshold for anomaly
    threshold = config.get('inference', {}).get('threshold')
    if threshold is None:
        # If no threshold in config, use a heuristic (e.g., 95th percentile of errors) 
        threshold = np.percentile(errors, 95)
        logger.info(f"No threshold specified. Using 95th percentile of errors as threshold: {threshold:.4f}")
    # Classify as anomaly if error exceeds threshold
    anomaly_flags = errors > threshold
    # Output results
    num_anomalies = np.sum(anomaly_flags)
    logger.info(f"Detected {num_anomalies} anomalies out of {len(errors)} samples (threshold={threshold:.4f}).")
    if y_test is not None:
        # If ground truth is available, compute accuracy or other metrics
        accuracy = np.mean((anomaly_flags.astype(int) == y_test.astype(int))) * 100
        logger.info(f"Accuracy against ground truth: {accuracy:.2f}%")
