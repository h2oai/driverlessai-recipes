"""Extract MFCC and spectrogram features from audio files"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np

class AudioMFCCTransformer(CustomTransformer):

    _modules_needed_by_name = ['librosa==0.7.0']  
    _parallel_task = True  # if enabled, params_base['n_jobs'] will be >= 1 (adaptive to system), otherwise 1
    _can_use_gpu = True  # if enabled, will use special job scheduler for GPUs
    _can_use_multi_gpu = True  # if enabled, can get access to multiple GPUs for single transformer (experimental)
    _numeric_output = True

    @staticmethod
    def is_enabled():
        return False

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)
   

    @staticmethod
    def do_acceptance_test():
        return False
    
    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)
 
    #Return MFCC based on spectrogram of audio

    def get_mfcc(self,file_path):
        
        import librosa
        
        sampling_rate = 16000
        duration = 1 #in secs
        hop_length = 347*duration 
        fmin = 20 #min freq
        fmax = sampling_rate // 2 #max freq
        n_mels = 128  #number of mels
        n_fft = n_mels * 20 #fft window size
        padmode = 'constant'
        samples = sampling_rate * duration #number of samples
        n_mfcc = 13  #number of Mel FCC to use
        
        try:
            
            audio, sr = librosa.load(file_path, sr=sampling_rate)
            
            #Trim silence
            if len(audio)> 0: 
                audio, _ = librosa.effects.trim(audio) 
            
            #Trim if audio length > samples 
            if len(audio) > samples: 
                audio = audio[0:0+samples]
                
            #Else pad blanks if shorter 
            else: 
                padding = samples - len(audio)
                offset = padding // 2
                audio = np.pad(audio, (offset, samples - len(audio) - offset), padmode)
   
            #Get Mel spectogram of audio
            spectrogram = librosa.feature.melspectrogram(audio,
                                                 sr=sampling_rate,
                                                 n_mels=n_mels,
                                                 hop_length=hop_length,
                                                 n_fft=n_fft,
                                                 fmin=fmin,
                                                 fmax=fmax)
            #Convert to log scale (DB)
            spectrogram = librosa.power_to_db(spectrogram)
            
            #Get MFCC and second derivatives
            mfcc = librosa.feature.mfcc(S=spectrogram, n_mfcc=n_mfcc)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            
            #Append MFCC to spectrogram and flatten
            features = np.concatenate((spectrogram,mfcc,delta2_mfcc),axis=0)
            X = features.ravel()
            
            return X
        except:
            spectrogram = np.zeros(((n_mels+2*n_mfcc)*47), dtype=np.float32)
            X = spectrogram.ravel()
            return X
    
    def transform(self, X: dt.Frame):
        
        import pandas as pd
        

        mels = X.to_pandas().iloc[:, 0].apply(lambda x: self.get_mfcc(x))
        
        col_names = ['X_'+ str(i) for i in range (0,len(mels[0]))]
        rows = len(mels)
        cols = len(mels[0])
        output_df = pd.DataFrame(data=np.reshape(np.concatenate(mels),(rows,cols)),columns=col_names)

        return output_df
