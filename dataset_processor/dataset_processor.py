'''
Abstract class for dataset processors
'''
from abc import ABC, abstractmethod


class DatasetProcessor:

    def __init__(self) -> None:
        pass

    @abstractmethod
    def load_dataset(self):
        pass


