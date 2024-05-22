"""Data recipe to transform input audio to Mel spectrograms"""

"""
This data recipe makes the following steps:
1. Reads audio file
2. Converts audio file to the Mel spectrogram
3. Save Mel spectrogram to .png image
4. Upload image dataset to DAI

Recipe is based on the Kaggle Freesound Audio Tagging 2019 challenge:
https://www.kaggle.com/c/freesound-audio-tagging-2019

To use the recipe follow the next steps:
1. Download a subsample of the audio dataset from here:
http://h2o-public-test-data.s3.amazonaws.com/bigdata/server/Image Data/freesound_audio.zip
2. Unzip it and specify the path to the dataset in the DATA_DIR global variable
3. Upload the dataset into Driverless AI using the Add Data Recipe option

The transformed dataset is also available and could be directly uploaded to Driverless AI:
http://h2o-public-test-data.s3.amazonaws.com/bigdata/server/Image Data/freesound_images.zip
"""

DATA_DIR = "/path/to/freesound_audio/"

import os
import shutil
import numpy as np
import pandas as pd
from h2oaicore.data import CustomData


class AudioToMelSpectogram:
    """
    Transforms input audio files into Mel Spectrograms.
    Audio reading and transformation is mostly taken from here:
    https://github.com/lRomul/argus-freesound/blob/master/src/audio.py

    For Ubuntu, required to do: sudo apt-get install libsndfile1 libsndfile1-dev
    For Centos, required to do: sudo yum install libsndfile libsndfile-dev
    """
    _modules_needed_by_name = ["librosa==0.10.2.post1"]

    def __init__(
            self, min_seconds=2, sampling_rate=44100, n_mels=128, hop_length=345 * 2
    ):
        # Audio hyperparameters
        self.min_seconds = min_seconds
        self.sampling_rate = sampling_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = self.n_mels * 20
        self.fmin = 20
        self.fmax = self.sampling_rate // 2

    def read_audio(self, file_path):
        min_samples = int(self.min_seconds * self.sampling_rate)
        import librosa
        y, sr = librosa.load(file_path, sr=self.sampling_rate)
        # Trim silence
        trim_y, trim_idx = librosa.effects.trim(y)

        # Pad to min_samples
        if len(trim_y) < min_samples:
            padding = min_samples - len(trim_y)
            offset = padding // 2
            trim_y = np.pad(trim_y, (offset, padding - offset), "constant")
        return trim_y

    def audio_to_melspectrogram(self, audio):
        import librosa
        spectrogram = librosa.feature.melspectrogram(
            audio,
            sr=self.sampling_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        spectrogram = librosa.power_to_db(spectrogram)
        spectrogram = spectrogram.astype(np.float32)
        return spectrogram

    def melspectrogram_to_image(self, spectrogram):
        norm_min = np.min(spectrogram)
        norm_max = np.max(spectrogram)

        # Normalize to [0, 255] range
        image = 255 * (spectrogram - norm_min) / (norm_max - norm_min)
        image = image.astype(np.uint8)

        return image

    def read_as_image(self, file_path):
        audio = self.read_audio(file_path)
        mels = self.audio_to_melspectrogram(audio)
        image = self.melspectrogram_to_image(mels)
        return image


class AudioDataset(CustomData):
    """
    Takes input audio files and re-writes them as Mel Spectrogram images
    """

    @staticmethod
    def create_data():
        # Path to the directory with audios
        path_to_files = os.path.join(DATA_DIR, "audio/")
        if not os.path.isdir(path_to_files):
            return []
        # Path to a .csv with labels. First column is audio name, second column is label
        path_to_labels = os.path.join(DATA_DIR, "labels.csv")
        if not os.path.isfile(path_to_labels):
            return []

        # Create output directory
        output_path = os.path.join(path_to_files, "mel_spectrograms/")
        os.makedirs(output_path, exist_ok=True)

        # Read data
        df = pd.read_csv(path_to_labels)
        audio_filenames = df.iloc[:, 0]

        # Convert audio to melspectrogram and save as image
        wav2mel = AudioToMelSpectogram()
        import cv2

        for idx, audio_name in enumerate(audio_filenames):
            audio_path = os.path.join(path_to_files, audio_name)
            image_path = os.path.join(output_path, f"{audio_name}.png")
            img = wav2mel.read_as_image(audio_path)
            cv2.imwrite(image_path, img)
            df.iloc[idx, 0] = f"{audio_name}.png"

        # Save .csv file with labels
        df.to_csv(os.path.join(output_path, "labels.csv"), index=False)

        # Create .zip archive to upload to DAI
        shutil.make_archive(
            os.path.join(path_to_files, "mel_spectrograms"), "zip", output_path
        )
        zip_archive_path = os.path.join(path_to_files, "mel_spectrograms.zip")

        return zip_archive_path
