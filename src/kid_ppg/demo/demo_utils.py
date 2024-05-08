from importlib import resources as impresources
from . import demo_data
import pickle

demo_data_file = impresources.files(demo_data) / 'PPGDalia_S6_stairs.pkl'
demo_data_filename = demo_data_file.resolve()

def load_demo_data():
    """Loads demo data included in the KID-PPG package. 

    The data are taken from the PPGDalia dataset, Subject 6, Stairs activity.

    Returns:
        X (numpy.ndarray): PPG and Accelerometer signals. Size is 
          [N_samples, N_channels = 4, 256]. First channel is the PPG signal followed
          by 3 acceleration signals. 256 points correspond to 8-second windows with
          32.0Hz sampling frequency. 

        y (numpy.ndarray): Ground-truth heart rate corresponding to the windows in X.
    """
    with open(demo_data_filename, 'rb') as handle:
        data = pickle.load(handle)

    X, y = data['X'], data['y']

    return X, y