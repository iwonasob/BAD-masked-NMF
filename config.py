'''
SUMMARY:  config file
AUTHOR:   Qiuqiang Kong
Created:  2016.10.15
Modified: 
--------------------------------------
'''
# MODE
mode="dev"

# PATHS
home_path   = "/vol/vssp/AcousticEventsDetection/BAD"

wav_url     = {'warblrb' : "https://archive.org/download/warblrb10k_public/warblrb10k_public_wav.zip",
            'ff1010'  : "https://archive.org/download/ff1010bird/ff1010bird_wav.zip"}
csv_url     = {'warblrb' : "https://ndownloader.figshare.com/files/6035817",
            'ff1010'  : "https://ndownloader.figshare.com/files/6035814"} 
csv_path    = {'warblrb' : "warblrb10k_public_metadata.csv",
            'ff1010'  : "ff1010bird_metadata.csv"}
                       

# global params
feature ="mel"
win     = 1024
fs      = 44100.
n_mels  = 40
n_sh    = 4
n_duration = 440    # 44 frames per second, all together 10 seconds

# NMF training parameters
type        = '0_1' # available types: '01' for masked NMF, '0_1' for class-conditioned NMF, 'unsupervised' for unsupervised NMF
update_func = "kl"
iterations  = 200
rank_0      = 50
rank_1      = 10

# Classifier paramters
n_trees = 500