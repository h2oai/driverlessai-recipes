# SPEECH

## AudioMFCCTransformer

#### ➡️ Code
- [audio_mfcc_transformer.py](audio_MFCC_transformer.py)

#### ➡️ Description
Extracts numerical features from audio files using spectrograms and Mel-Frequency Cepstral Coefficients (MFCC)

#### ➡️ Inputs
- Single text column which contains full paths to audio files on the same machine running DAI

#### ➡️ Outputs
- Multiple numerical columns depending on the audio file

#### ➡️ Environment expectation
No limitations

#### ➡️ Dependenencies
- librosa

----

## AzureSpeechToText 

#### ➡️ Code
- [azure_speech_to_text.py](azure_speech_to_text.py)

#### ➡️ Description

An example of integration with Azure Speech Recognition Service. The
transform translate an audio file into text representation. The audio
file needs to be in PCM 16bit format and its lenght is limited by 15seconds.

#### ➡️ Inputs
- a string column with location of wav files PCM 16bit, max 15seconds

#### ➡️ Outputs
- a string column with translation of the wav files or None
    in case of a problem ()

#### ➡️ Environment expectation
- `DAI_CUSTOM_AzureSpeechToText_SERVICE_KEY`:
   Contains API key for Azure Speech Service

- `DAI_CUSTOM_AzureSpeechToText_SERVICE_REGION`:
   Azure region to access Speech Service.
   Default: `westus`


#### ➡️ Dependenencies
- `azure-cognitiveservices-speech`

