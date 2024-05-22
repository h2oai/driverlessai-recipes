"""
Speech to text using Azure Cognitive Services

Settings for this recipe:

Assing AZURE_SERVICE_KEY and AZURE_SERVICE_REGION global variable prior to usage

Assign WAV_COLNAME global variable with proper column name from your dataset.
This colums should contain absolute paths to .wav file which needs to be converted to text.

"""

import datatable as dt
import numpy as np
import pandas as pd
from h2oaicore.data import BaseData
import typing

# Please fill up before usage
AZURE_SERVICE_KEY = ''
AZURE_SERVICE_REGION = ''
WAV_COLNAME = ''


class AzureWav2Txt(BaseData):
    """Base class for a custom data creation recipe that can be specified externally to Driverless AI.
    To use as recipe, in the class replace CustomData with your class name and replace BaseData with CustomData
    Note: Experimental API, will most likely change in future versions.
    """

    """Specify the python package dependencies (will be installed via pip install mypackage==1.3.37)"""
    _modules_needed_by_name = ["azure-cognitiveservices-speech==1.37.0"]

    @staticmethod
    def create_data(X: dt.Frame = None) -> dt.Frame:
        if X is None:
            return []

        import azure.cognitiveservices.speech as speechsdk

        def _wav_to_str(filename: str, cfg) -> typing.Optional[str]:
            audio_config = speechsdk.audio.AudioConfig(filename=filename)
            speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
            result = speech_recognizer.recognize_once()
            return result.text if result.reason == speechsdk.ResultReason.RecognizedSpeech else None

        X = X.to_pandas()
        if WAV_COLNAME in X.columns:
            speech_config = speechsdk.SpeechConfig(subscription=AZURE_SERVICE_KEY, region=AZURE_SERVICE_REGION)
            X[WAV_COLNAME] = X[WAV_COLNAME].astype(str)
            X[WAV_COLNAME + "_txt"] = X[WAV_COLNAME].apply(lambda s: _wav_to_str(s, speech_config))

        return dt.Frame(X)
