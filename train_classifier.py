from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_classifier():
    """Defines the classification head on top of the encoder."""
    model = Sequential([
        Dense(128, activation='relu'),
        Dense(3, activation='softmax')  # 3 vehicle categories
    ])
    return model
