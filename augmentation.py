import numpy as np

def time_shift(signal, shift):
    """Shifts signal by a specified number of samples."""
    return np.roll(signal, shift)

def time_reverse(signal):
    """Reverses signal in time."""
    return signal[::-1]

def downsample(signal, factor):
    """Reduces number of samples."""
    return signal[::factor]

def upsample(signal, factor):
    """Interpolates signal to increase the number of samples."""
    return np.repeat(signal, factor)

def hybrid_augmentation(signal):
    """Applies multiple augmentations to generate diverse samples."""
    return [
        time_shift(signal, shift=10),
        time_reverse(signal),
        downsample(signal, factor=2),
        upsample(signal, factor=3),
    ]
