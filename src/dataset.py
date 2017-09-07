'''
SUMMARY:  downloads and extracts the dataset
AUTHOR:   Iwona Sobieraj
Created:  2017.09.07
Modified: -
--------------------------------------
'''
import config as cfg
import os
import sys
import requests
import zipfile
from clint.textui import progress


class DatasetCreator:
    def __init__(self,
                 dataset_name):
                     
        """ Initialize class
        Args:
            dataset_name (string): Name of the dataset to prepare
        """
        self.dataset_name   = dataset_name
        self.root_path      = os.path.join(cfg.home_path, self.dataset_name)
        self.wav_url        = cfg.wav_url[self.dataset_name]
        self.csv_url        = cfg.csv_url[self.dataset_name]
        path , zip_name     = os.path.split(self.wav_url)
        self.wav_zip        = os.path.join(cfg.home_path, zip_name)  
        self.csv_path       = os.path.join(self.root_path, cfg.csv_path[self.dataset_name])
        
        if not os.path.isdir(self.root_path):
            os.makedirs(self.root_path)
            
            
    def run(self):
        self.download()
        self.extract()
        
        
    def download(self):
        """ Download the dataset and annotation file

        """
        if not os.path.isfile(self.wav_zip): 
            print("Downloading the dataset "+ self.dataset_name)
            r = requests.get(self.csv_url, stream=True) # TODO fix the repetition!
            with open(self.csv_path, 'wb') as f:
                total_length = int(r.headers.get('content-length'))
                for chunk in progress.bar(r.iter_content(chunk_size=8192), expected_size=(total_length/8192) + 1): 
                    if chunk:
                        f.write(chunk)
                        f.flush()
            f.close()
            r = requests.get(self.wav_url, stream=True)
            with open(self.wav_zip, 'wb') as f:
                total_length = int(r.headers.get('content-length'))
                for chunk in progress.bar(r.iter_content(chunk_size=8192), expected_size=(total_length/8192) + 1): 
                    if chunk:
                        f.write(chunk)
                        f.flush()
            f.close()
        else:
            print(self.dataset_name + " has been already downloaded!")
            
        
    def extract(self):
        """ Extract the downloaded dataset

        """
        if not os.listdir(os.path.join(self.root_path,"wav")):
            print("Extracting the dataset "+ self.dataset_name)
            zip= zipfile.ZipFile(self.wav_zip)
            zip.extractall(self.root_path)
        else:
            print(self.dataset_name + " has been already extracted!")
            
