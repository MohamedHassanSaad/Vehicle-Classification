import tensorflow as tf
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense

def build_encoder(input_shape):
    """Defines the CNN-based feature extractor (Encoder)."""
    inputs = tf.keras.Input(shape=input_shape)
    
    x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = GlobalAveragePooling1D()(x)
    
    return tf.keras.Model(inputs, x, name="Encoder")
