from importlib import resources as impresources
from . import demo_data
import pickle

demo_data_file = impresources.files(demo_data) / 'PPGDalia_S6_stairs.pkl'
demo_data_filename = demo_data_file.resolve()

def load_demo_data():
    with open(demo_data_filename, 'rb') as handle:
        data = pickle.load(handle)

    X, y = data['X'], data['y']

    return X, y