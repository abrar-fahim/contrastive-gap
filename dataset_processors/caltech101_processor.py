import torchvision.datasets as dset
import clip
import torch
import random
from torch.utils.data import DataLoader, Subset
from src.utils import  get_checkpoint_path
from dataset_processors.dataset_processor_parent import DatasetProcessorParent
import os
from clips.hf_clip import HFClip
import numpy as np
import wandb

from torchvision.datasets import Caltech101
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


class Caltech101Processor(DatasetProcessorParent):

    def __init__(self) -> None:
        self.root = './datasets'
        super().__init__()

        self.name = 'Caltech 101'
        self.keyname = self.name.replace(' ', '').lower()
        self.print_dataset_stats()
        


    def load_val_dataset(self):
        self.val_dataset = Caltech101(root=self.root, download=True, transform=self.preprocess)

        self.classes = self.val_dataset.categories

        # add 'photo of ' to the beginning of each class name
        self.classes = ['photo of ' + class_name for class_name in self.classes]

    def load_train_dataset(self):
        self.train_dataset = Caltech101(root=self.root, download=True, transform=self.preprocess)

    def get_accuracy(self, linear_classifier: LogisticRegression, all_val_features: torch.FloatTensor, all_val_labels: list) -> float:
        '''
        Mean per-class accuracy
        From:
        - https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py,

        - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
        '''

        # get predictions
        predictions = linear_classifier.predict(all_val_features)

        # get confusion matrix using sklearn
       
        matrix = confusion_matrix(all_val_labels, predictions, normalize='true')

        # get per-class accuracy
        per_class_accuracy = np.diag(matrix)

        assert all(per_class_accuracy >= 0) and all(per_class_accuracy <= 1), 'Per-class accuracy should be between 0 and 1'

        # get mean per-class accuracy
        mean_per_class_accuracy = np.mean(per_class_accuracy)

        return mean_per_class_accuracy
        







'''
<!DOCTYPE html><html><head><title>Google Drive - Virus scan warning</title><meta http-equiv="content-type" content="text/html; charset=utf-8"/><style nonce="kf9bqnWP8WINbOFFKbCWSA">.goog-link-button{position:relative;color:#15c;text-decoration:underline;cursor:pointer}.goog-link-button-disabled{color:#ccc;text-decoration:none;cursor:default}body{color:#222;font:normal 13px/1.4 arial,sans-serif;margin:0}.grecaptcha-badge{visibility:hidden}.uc-main{padding-top:50px;text-align:center}#uc-dl-icon{display:inline-block;margin-top:16px;padding-right:1em;vertical-align:top}#uc-text{display:inline-block;max-width:68ex;text-align:left}.uc-error-caption,.uc-warning-caption{color:#222;font-size:16px}#uc-download-link{text-decoration:none}.uc-name-size a{color:#15c;text-decoration:none}.uc-name-size a:visited{color:#61c;text-decoration:none}.uc-name-size a:active{color:#d14836;text-decoration:none}.uc-footer{color:#777;font-size:11px;padding-bottom:5ex;padding-top:5ex;text-align:center}.uc-footer a{color:#15c}.uc-footer a:visited{color:#61c}.uc-footer a:active{color:#d14836}.uc-footer-divider{color:#ccc;width:100%}.goog-inline-block{position:relative;display:-moz-inline-box;display:inline-block}* html .goog-inline-block{display:inline}*:first-child+html .goog-inline-block{display:inline}sentinel{}</style><link rel="icon" href="//ssl.gstatic.com/docs/doclist/images/drive_2022q3_32dp.png"/></head><body><div class="uc-main"><div id="uc-dl-icon" class="image-container"><div class="drive-sprite-aux-download-file"></div></div><div id="uc-text"><p class="uc-warning-caption">Google Drive can't scan this file for viruses.</p><p class="uc-warning-subcaption"><span class="uc-name-size"><a href="/open?id=137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp">101_ObjectCategories.tar.gz</a> (126M)</span> is too large for Google to scan for viruses. Would you still like to download this file?</p><form id="download-form" action="https://drive.usercontent.google.com/download" method="get"><input type="submit" id="uc-download-link" class="goog-inline-block jfk-button jfk-button-action" value="Download anyway"/><input type="hidden" name="id" value="137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp"><input type="hidden" name="export" value="download"><input type="hidden" name="confirm" value="t"><input type="hidden" name="uuid" value="29caa046-0818-4b88-aa71-aa9aca00d098"></form></div></div><div class="uc-footer"><hr class="uc-footer-divider"></div></body></html>
'''