import sys
import os

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# add sibling directory to path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))# def 

from dataset_processors.cifar10_processor import CIFAR10Processor


cifar10 = CIFAR10Processor()

cifar10.print_dataset_stats()

print(cifar10.val_dataset[0])
