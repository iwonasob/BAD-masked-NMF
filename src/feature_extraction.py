'''
Extracts appropriate features from the datasets
'''
import config as cfg
import os
import sys
import wavio
import pickle
import numpy as np
from scipy import signal
import librosa
from tqdm import tqdm

def readwav(path):
    Struct = wavio.read( path )
    wav = Struct.data.astype(float) / np.power(2, Struct.sampwidth*8-1)
    fs = Struct.rate
    return wav, fs


class FeatureExtraction:
    def __init__(self,
                 dataset_name):
                     
        """ Initialize class
        Args:
            dataset_name (string): Name of the dataset to prepare
        """
        self.dataset_name   = dataset_name
        self.root_path      = os.path.join(cfg.home_path, self.dataset_name)
        self.csv_path       = os.path.join(self.root_path, cfg.csv_path[self.dataset_name])
        self.wav_path       = os.path.join(self.root_path, "wav")
        self.feature        = cfg.feature
        self.feature_path   = os.path.join(self.root_path, self.feature)
        
        if not os.path.isdir(self.feature_path):
            os.makedirs(self.feature_path)
        
    def run(self):
        if not os.listdir(self.feature_path):
            self.extract_mel()
        else:
            print(self.feature +  " features are already extracted! ")

            
    # extract mel feature
    def extract_mel(self):
        """ Extract mel spectrograms
        """
        print("Extracting " + self.feature + " features")
        names = [ na for na in os.listdir(self.wav_path) if na.endswith('.wav') ]
        names = sorted(names)
        
        melW = librosa.filters.mel(cfg.fs, n_fft=cfg.win, n_mels=cfg.n_mels, fmin=0., fmax=22100 )
        melW /= np.max(melW, axis=-1)[:,None]
        
        for f in tqdm(names):
            file = os.path.join(self.wav_path, f)
            [wav, fs] = readwav(file)
            if (wav.ndim==2): 
                wav = np.mean( wav, axis=-1 )
            assert fs==cfg.fs
            ham_win = np.hamming(cfg.win)
            [fq, t, X] = signal.spectral.spectrogram( wav, window=ham_win, nperseg=cfg.win, noverlap=0, detrend=False, return_onesided=True, mode='magnitude' ) 
            X = X.T
            X = np.dot( X, melW.T )
            X=X/np.max(X)
            
            name=os.path.splitext(f)[0]
            out_path = os.path.join(self.feature_path, name+".f")
            pickle.dump( X, open(out_path, 'wb'))
        
