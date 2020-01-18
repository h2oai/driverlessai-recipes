"""
Speech to text using Mozilla's DeepSpeech

Settings for this recipe:

Assing MODEL_PATH global variable prior to usage

Assign WAV_COLNAME global variable with proper column name from your dataset.
This colums should contain absolute paths to .wav file which needs to be converted to text.

General requirements to .wav's:

1 channel (mono)
16 bit
16000 frequency

"""

import datatable as dt
import numpy as np
import pandas as pd
from h2oaicore.data import BaseData
import typing
import shlex
import subprocess
import sys
import wave
import os
from timeit import default_timer as timer
import logging


try:
    from shhlex import quote
except ImportError:
    from pipes import quote

_global_modules_needed_by_name = ["deepspeech-gpu==0.6.1"]   

#####Please fill up before usage
MODEL_PATH = "/home/dmitry/Desktop/DAI/deepspeech/deepspeech-0.6.1-models"
beam_width = 500 # Beam width for the CTC decoder
lm_alpha = 0.75 # Language model weight (lm_alpha)
lm_beta = 1.85 # Word insertion bonus (lm_beta)

WAV_COLNAME = "WAV"
LOG_FILE = "/var/tmp/MozillaDeepSpeechWav2Txt.log" # Put '' or None if don't want to log
MAX_SEC = 60 # Maximun length of .wav file in seconds, if .wav is longer it will be clipped
#####Please fill up before usage


def convert_samplerate(audio_path, desired_sample_rate):
    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little --compression 0.0 --no-dither - '.format(quote(audio_path), desired_sample_rate)
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
    except OSError as e:
        raise OSError(e.errno, 'SoX not found, use {}hz files or install it: {}'.format(desired_sample_rate, e.strerror))

    return desired_sample_rate, np.frombuffer(output, np.int16)

class MozillaDeepSpeechWav2Txt(BaseData):
    _modules_needed_by_name = ["deepspeech-gpu==0.6.1"] 

    @staticmethod
    def create_data(X: dt.Frame = None) -> dt.Frame:
        from deepspeech import Model
        
        try:
            logger = logging.getLogger(__name__)
            hdlr = logging.FileHandler(LOG_FILE)
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            hdlr.setFormatter(formatter)
            logger.addHandler(hdlr) 
            logger.setLevel(logging.INFO)
        except:
            logger = False
        
        X = X.to_pandas()
        if WAV_COLNAME in X.columns:
            model = os.path.join(MODEL_PATH,"output_graph.pbmm")
            lm = os.path.join(MODEL_PATH,"lm.binary")
            trie = os.path.join(MODEL_PATH,"trie")
            
            if logger:
                logger.info('Loading model from file {}'.format(model))
            model_load_start = timer()
            ds = Model(model, beam_width)
            model_load_end = timer() - model_load_start
            if logger:
                logger.info('Loaded model in {:.3}s.'.format(model_load_end))
            

            desired_sample_rate = ds.sampleRate()
            
            if logger:
                logger.info('Loading language model from files {} {}'.format(lm, trie))
                
            lm_load_start = timer()
            ds.enableDecoderWithLM(lm, trie, lm_alpha, lm_beta)
            lm_load_end = timer() - lm_load_start
            
            if logger:
                logger.info('Loaded language model in {:.3}s.'.format(lm_load_end))
                logger.info('Running inference.')
                

            results = []
            ds_len = len(X[WAV_COLNAME])
            for i, audio_fn in enumerate(X[WAV_COLNAME].values.tolist()):
                inference_start = timer()
                audio_length = 0
                fin = wave.open(audio_fn, 'rb')
                fs = fin.getframerate()
                if fs != desired_sample_rate:
                    if logger:
                        err_msg = 'Original sample rate ({}) is different than {}hz. '\
                                  'Resampling might produce erratic speech recognition.'
                        logger.warning(err_msg.format(fs, desired_sample_rate))
    
                    fs, audio = convert_samplerate(audio_fn, desired_sample_rate)
                else:
                    audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
                
                if MAX_SEC > 0:
                    audio = audio[:int(fs*MAX_SEC)]
                    
                audio_length = len(audio) * (1/fs)
                fin.close()
                
                try:
                    text = ds.stt(audio)
                except Exception as e:
                    text = ''
                    logger.error(e)
                    
                results.append(text)

                inference_end = timer() - inference_start
                if logger:
                    logger.info('Record {:d} of {:d}. Inference took {:0.3f}s for {:0.3f}s audio file.'.format(i, ds_len, inference_end, audio_length))
            
            X[WAV_COLNAME+"_txt"] = results
        
        return dt.Frame(X)
    
    