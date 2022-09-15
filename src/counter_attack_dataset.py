from abc import ABC
from spektral.data import Dataset


class CounterAttackDataset(Dataset, ABC):
    """
    Dataset class that inherits all functionality from spektral.data.Dataset
    This class is necessary to open the pickled dataset file
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)