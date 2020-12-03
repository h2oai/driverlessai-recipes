""" Data Recipe to Crop the Cheque Image (or any Image) in a fixed dimension """

DATA_DIR = "/path/to/images/"

## set the dimension of cropping area
crop_dimension = (10, 240, 450, 300)

from h2oaicore.data import CustomData
from PIL import Image
import pandas as pd
import shutil, os


class CropDataset(CustomData):
    @staticmethod
    def create_data():
        ## images - folder containing original images
        ## cropped - folder to store cropped images
        ## labels.csv - filename consisting of labels
        path_to_files = os.path.join(DATA_DIR, "images/")
        path_to_labels = os.path.join(DATA_DIR, "labels.csv")
        output_path = os.path.join(path_to_files, "cropped/")
        os.makedirs(output_path, exist_ok=True)

        df = pd.read_csv(path_to_labels)

        ## Take only the image name from the .csv
        df["path"] = df["path"].map(lambda x: os.path.split(x)[-1])
        image_filenames = df["path"].values

        ## Load Image, Perform Cropping, Save Cropped Image
        for idx, image_name in enumerate(image_filenames):
            image_path = os.path.join(path_to_files, image_name)

            im = Image.open(image_path)
            im = im.crop(crop_dimension)

            cropped_path = os.path.join(output_path, image_name)
            im.save(cropped_path)

        ## save the final output in a zip archive
        df.to_csv(os.path.join(output_path, "labels.csv"), index=False)
        shutil.make_archive(os.path.join(path_to_files, "cropped"), "zip", output_path)
        zip_archive_path = os.path.join(path_to_files, "cropped.zip")
        shutil.rmtree(output_path)
        return zip_archive_path
