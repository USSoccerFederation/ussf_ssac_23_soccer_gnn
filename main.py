from os.path import join
import pickle

from spektral.data import Graph
# even though this import is not used, without it the loaded pickle file will
# throw an error because it won't know what a CounterAttackDataset class is
from src.counter_attack_dataset import CounterAttackDataset


def load_from_pickle(file_path: str) -> list[Graph]:
    with open(file_path, 'rb') as handle:
        d = pickle.load(handle)
    return d


if __name__ == "__main__":
    DATA_FOLDER = 'data'
    DATA_FILE = 'counterattack_data.pkl'

    data_path = join(DATA_FOLDER, DATA_FILE)

    data = load_from_pickle(file_path=data_path)


