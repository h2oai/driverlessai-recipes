"""An example of integration with Azure Speech Recognition Service"""
#
# Custom transformer: AzureSpeechToText
#
# Inputs:
#  - a string column with location of wav files PCM 16bit, max 15seconds
#
# Outputs:
#  - a string column with translation of the wav files or None
#    in case of a problem ()
#
# Environment:
#  - DAI_CUSTOM_AzureSpeechToText_SERVICE_KEY:
#     Contains API key for Azure Speech Service
#
#  - DAI_CUSTOM_AzureSpeechToText_SERVICE_REGION:
#     Azure region to access Speech Service.
#     Default: 'westus'
#

import datatable as dt
import numpy as np
import os
import typing

from h2oaicore.transformer_utils import CustomTransformer


class AzureSpeechToText(CustomTransformer):
    _unsupervised = True

    _numeric_output = False
    _display_name = 'AzureSpeechToTextTransformer'
    _modules_needed_by_name = ["azure-cognitiveservices-speech==1.16.0"]

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)

    @staticmethod
    def do_acceptance_test():
        return False

    @staticmethod
    def is_enabled():
        return True

    @staticmethod
    def can_use(accuracy, interpretability, **kwargs):
        return False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        service_key, service_region = self.get_service_key(), self.get_service_region()
        import azure.cognitiveservices.speech as speechsdk
        # TODO reject missing configuration or wrong configuration directly
        self.speechsdk = speechsdk
        self.speech_config = self.get_speech_config(service_key, service_region)

    #
    # Custom configuration - here derived from environment.
    # However, better option would be to provide an interface
    # to access user configuration.
    #
    def _get_config_opt(self, name: str, default: str = None) -> str:
        return os.getenv(f'DAI_CUSTOM_{self.__class__.__name__}_{name}', default)

    def get_service_key(self) -> str:
        return self._get_config_opt('SERVICE_KEY')

    def get_service_region(self) -> str:
        return self._get_config_opt('SERVICE_REGION', 'westus')

    def get_speech_config(self, service_key, service_region):
        return self.speechsdk.SpeechConfig(subscription=service_key, region=service_region)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def _wav_to_str(self, filename: str) -> typing.Optional[str]:
        audio_config = self.speechsdk.audio.AudioConfig(filename=filename)
        speech_recognizer = self.speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=audio_config)
        result = speech_recognizer.recognize_once()
        return result.text if result.reason == self.speechsdk.ResultReason.RecognizedSpeech else None

    def transform(self, X: dt.Frame) -> dt.Frame:
        # TODO: parallelize
        return X.to_pandas().astype(str).iloc[:, 0].apply(lambda s: self._wav_to_str(s))
