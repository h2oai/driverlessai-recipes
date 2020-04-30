"""Prep and upload the MNIST datasset"""

# Contributors: Michelle Tanco - michelle.tanco@h2oai
# Created: March 8th, 2020


from typing import Union, List
from h2oaicore.data import CustomData
import datatable as dt
import numpy as np
import pandas as pd

_global_modules_needed_by_name = ['mnist==0.2.2']
import mnist


class MNISTData(CustomData):
    @staticmethod
    def create_data(X: dt.Frame = None):

        train_images = mnist.train_images()
        train_labels = mnist.train_labels()
        test_images = mnist.test_images()
        test_labels = mnist.test_labels()

        train_images = train_images.reshape((len(train_images), -1))
        test_images = test_images.reshape((len(test_images), -1))

        train_data = pd.DataFrame(train_images)
        test_data = pd.DataFrame(test_images)

        train_data = train_data.add_prefix('b')
        test_data = test_data.add_prefix('b')

        train_data["number"] = train_labels
        test_data["number"] = test_labels

        train_data = train_data.apply(np.int8)
        test_data = test_data.apply(np.int8)

        return {"mnist_train": train_data, "mnist_test": test_data}
