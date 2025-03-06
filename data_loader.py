import numpy as np
import pandas as pd
import os

def load_seismic_data(data_path):
    """Loads seismic data from CSV files and returns as NumPy arrays."""
    files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    data = []
    
    for file in files:
        df = pd.read_csv(os.path.join(data_path, file))
        data.append(df.values)
    
    return np.array(data)
