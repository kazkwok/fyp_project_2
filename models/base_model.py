import abc
import tensorflow as tf

class BaseModel(abc.ABC):
    def __init__(self, config: dict):
        """Base model initializer that stores config and will hold the tf.keras model."""
        self.config = config
        self.model = None  # This will be a tf.keras.Model after build_model is called

    @abc.abstractmethod
    def build_model(self):
        """Subclasses should implement this method to build the model architecture."""
        pass

    def save_model(self, save_path: str):
        """Save the entire model to the given file path."""
        if self.model is None:
            raise ValueError("Model has not been built.")
        self.model.save(save_path)
    
    def load_model(self, load_path: str):
        """Load a model from the given file path."""
        self.model = tf.keras.models.load_model(load_path)
    
    def train(self, x, y=None, **kwargs):
        """
        Train the model on given data. If y is None, for autoencoder use-case, use x as y (reconstruction).
        Additional kwargs (e.g., batch_size, epochs) are passed to tf.keras.Model.fit().
        Returns the training history.
        """
        if self.model is None:
            # Build and compile the model if not already done
            self.build_model()
            # Compile the model with parameters from config
            optimizer = tf.keras.optimizers.Adam(self.config['training']['learning_rate'])
            loss_fn = self.config['model'].get('loss', 'mse')
            self.model.compile(optimizer=optimizer, loss=loss_fn)
        # If y is not provided, assume autoencoder scenario (target = input)
        if y is None:
            y = x
        # Use config for batch size and epochs if not explicitly provided
        batch_size = kwargs.pop('batch_size', self.config['training'].get('batch_size', 32))
        epochs = kwargs.pop('epochs', self.config['training'].get('epochs', 1))
        # Train the model
        history = self.model.fit(x, y, batch_size=batch_size, epochs=epochs, **kwargs)
        return history

    def predict(self, x):
        """Run inference on the model for the given input."""
        if self.model is None:
            raise ValueError("Model is not built or loaded.")
        return self.model.predict(x)
