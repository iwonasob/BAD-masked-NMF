'''
SUMMARY:  config file
AUTHOR:   Qiuqiang Kong
Created:  2016.10.15
Modified: 
--------------------------------------
'''
# PATHS
home_path= "/vol/vssp/AcousticEventsDetection/BAD"

wav_url = {'warblrb' : "https://archive.org/download/warblrb10k_public/warblrb10k_public_wav.zip",
            'ff1010'  : "https://archive.org/download/ff1010bird/ff1010bird_wav.zip"}
csv_url = {'warblrb' : "https://ndownloader.figshare.com/files/6035817",
            'ff1010'  : "https://ndownloader.figshare.com/files/6035814"} 
csv_path = {'warblrb' : "warblrb10k_public_metadata.csv",
            'ff1010'  : "ff1010bird_metadata.csv"}
                       

# your workspace
scrap_fd = "/vol/vssp/AcousticEventsDetection/bird_song/bird_backup_scrap"     # you need modify this path
# mel_fd = "/vol/vssp/AcousticEventsDetection/bird_song/bird_backup_scrap"
mel_fd = "/vol/vssp/cvpwrkspc01/scratch/is0017"

# wbl dataset workspace
warblrb_cv10_csv_path = scrap_fd + "/warblrb_cv10.csv"
warblrb_denoise_wav_fd = scrap_fd + '/warblrb_denoise_wav'
warblrb_denoise_fe_fd = scrap_fd + '/warblrb_denoise_fe'
warblrb_denoise_fe_mel_fd = warblrb_denoise_fe_fd + '/warblrb_denoise_fe_mel'
warblrb_denoise_fe_fft_fd = warblrb_denoise_fe_fd + '/warblrb_denoise_fe_fft'
warblrb_dev_md_fd = scrap_fd + "/warblrb_dev_md"
warblrb_mel_fd = mel_fd + '/warblrb_fe_mel'
warblrb_fft_fd = scrap_fd + '/warblrb_fe_fft'

# ff dataset workspace
ff1010_cv10_csv_path = scrap_fd + "/ff1010_cv10.csv"
ff1010_denoise_wav_fd = scrap_fd + '/ff1010_denoise_wav'
ff1010_denoise_fe_fd = scrap_fd + '/ff1010_denoise_fe'
ff1010_denoise_fe_mel_fd = ff1010_denoise_fe_fd + '/ff1010_denoise_fe_mel'
ff1010_denoise_fe_fft_fd = ff1010_denoise_fe_fd + '/ff1010_denoise_fe_fft'
ff1010_dev_md_fd = scrap_fd + "/ff1010_dev_md"
ff1010_mel_fd = mel_fd + '/ff1010_fe_mel'
ff1010_fft_fd = scrap_fd + '/ff1010_fe_fft'

# test dataset workspace
test_wav_fd = "/vol/vssp/msos/qk/test_bird_wav"
test_mel_fd = mel_fd + '/test_fe_mel'

# global params
win = 1024
fs = 44100.
n_duration = 440    # 44 frames per second, all together 10 seconds