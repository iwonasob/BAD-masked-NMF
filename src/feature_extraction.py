'''
SUMMARY:  extracts the dataset
AUTHOR:   Iwona Sobieraj
Created:  2017.09.07
Modified: -
--------------------------------------
'''
import config as cfg
import os
import sys
import wavio
import cPickle
import numpy as np
from scipy import signal
import librosa

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
        self.feature        = "mel"
        self.feature_path   = os.path.join(self.root_path, self.feature)
        
        if not os.path.isdir(self.feature_path):
            os.makedirs(self.feature_path)
        
    def run(self):
        self.extract_mel(self.wav_path, self.feature_path )   
        

            
    # extract mel feature
    # Use preemphasis, the same as matlab
    def extract_mel(self, wav_fd, fe_fd):
        names = [ na for na in os.listdir(wav_fd) if na.endswith('.wav') ]
        names = sorted(names)
        cnt = 1
        for na in names:
            print(cnt, na)
            path = wav_fd + '/' + na
            wav, fs = readwav( path )
            if ( wav.ndim==2 ): 
                wav = np.mean( wav, axis=-1 )
            assert fs==44100
            ham_win = np.hamming(cfg.win)
            [f, t, X] = signal.spectral.spectrogram( wav, window=ham_win, nperseg=cfg.win, noverlap=0, detrend=False, return_onesided=True, mode='magnitude' ) 
            X = X.T
            
            # define global melW, avoid init melW every time, to speed up. 
            if globals().get('melW') is None:
                global melW
                melW = librosa.filters.mel( fs, n_fft=cfg.win, n_mels=40, fmin=0., fmax=22100 )
                melW /= np.max(melW, axis=-1)[:,None]
            
            X = np.dot( X, melW.T )
            X=X/np.max(X)
            
            # DEBUG. print mel-spectrogram
            # plt.matshow((X.T), origin='lower', aspect='auto')
            # plt.show()
            
            
            out_path = fe_fd + '/' + na[0:-4] + '.f'
            cPickle.dump( X, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
            cnt += 1
        