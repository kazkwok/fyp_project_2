import yaml
import numpy as np
from data.dataset import SoundDataset
from models.ae_model import AEModel
from utils.logger import get_logger
from utils.visualization import plot_loss

if __name__ == "__main__":
    # Set up logger
    logger = get_logger("train")
    logger.info("Loading configuration...")
    # Load configuration from YAML file
    with open("configs/base.yaml", "r") as f:
        config = yaml.safe_load(f)
    # Prepare training data (assuming normal data only for training)
    train_normal_dir = config['data']['train_normal_dir']
    train_abnormal_dir = config['data'].get('train_abnormal_dir')  # might be None for unsupervised training
    logger.info("Loading training dataset...")
    train_dataset = SoundDataset(train_normal_dir, abnormal_dir=None, config=config)
    X_train, y_train = train_dataset.load_data()
    logger.info(f"Training data loaded: {X_train.shape[0]} samples, each of dimension {X_train.shape[1]}")
    # Initialize model
    model = AEModel(config)
    # If a pretrained model is specified, load it (for transfer learning or resumed training)
    pretrained_path = config['paths'].get('pretrained_model_path')
    if pretrained_path:
        model.load_model(pretrained_path)
        logger.info(f"Loaded pre-trained model weights from {pretrained_path}")
    # Train the model (for autoencoder, y_train is not used since the model predicts X itself)
    logger.info("Starting model training...")
    history = model.train(X_train, y_train if train_abnormal_dir else None)
    logger.info("Training completed.")
    # Save the trained model
    save_path = config['paths']['model_save_path']
    model.save_model(save_path)
    logger.info(f"Model saved to {save_path}")
    # (Optional) Plot the training loss and save or show it
    if history is not None:
        loss_plot_path = config['paths'].get('loss_plot_path')
        plot_loss(history, save_path=loss_plot_path)
        if loss_plot_path:
            logger.info(f"Training loss plot saved to {loss_plot_path}")
