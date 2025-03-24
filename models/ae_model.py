from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from models.base_model import BaseModel
import tensorflow as tf

class AEModel(BaseModel):
    def __init__(self, config: dict):
        super().__init__(config)
        # Determine input dimension from config (n_mels * frames)
        self.n_mels = config['model']['n_mels']
        self.frames = config['model']['frames']
        self.input_dim = self.n_mels * self.frames
        # Architecture parameters (latent dim and hidden units)
        self.latent_dim = config['model'].get('latent_dim', 8)
        self.hidden_dim = config['model'].get('hidden_dim', 64)
        # Build and compile the autoencoder model
        self.build_model()
        optimizer = tf.keras.optimizers.Adam(config['training']['learning_rate'])
        # Use mean squared error for reconstruction loss
        self.model.compile(optimizer=optimizer, loss=config['model'].get('loss', 'mse'))
    
    def build_model(self):
        """Builds the autoencoder architecture and sets self.model."""
        # Define the layers of the autoencoder
        inputs = Input(shape=(self.input_dim,), name="input_layer")
        # Encoder: two hidden layers (ReLU)
        x = Dense(self.hidden_dim, activation='relu')(inputs)
        x = Dense(self.hidden_dim, activation='relu')(x)
        # Bottleneck (latent representation)
        latent = Dense(self.latent_dim, activation='relu', name="latent_layer")(x)
        # Decoder: two hidden layers (ReLU)
        x = Dense(self.hidden_dim, activation='relu')(latent)
        x = Dense(self.hidden_dim, activation='relu')(x)
        # Output layer: reconstruct input_dim (linear activation)
        outputs = Dense(self.input_dim, activation=None, name="output_layer")(x)
        # Create the Keras Model
        self.model = Model(inputs=inputs, outputs=outputs, name="AutoencoderModel")
