'''
Abstract class for dataset processors
'''
from abc import ABC, abstractmethod


class DatasetProcessorParent(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def load_train_dataset(self):
        pass

    @abstractmethod
    def load_val_dataset(self):
        pass

    @abstractmethod
    def print_dataset_stats(self):
        pass


