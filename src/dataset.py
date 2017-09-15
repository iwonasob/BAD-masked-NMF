'''
Download, extract and partition the datasets
'''
import config as cfg
import os
import sys
import requests
import zipfile
from clint.textui import progress
import numpy as np
np.random.seed(1515)
import pandas as pd


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

        self.wav_path       = os.path.join(self.root_path, "wav")
        self.csv_path       = os.path.join(self.root_path, cfg.csv_path[self.dataset_name])
        self.csv_10_path    = os.path.join(self.root_path, "cv10.csv")
        
        if not os.path.isdir(self.root_path):
            os.makedirs(self.root_path)

        if not os.path.isdir(self.wav_path):
            os.makedirs(self.wav_path)

            
    def run(self):
        self.download()
        self.extract()
        self.partition()
        
        
    def download(self):
        """ Download the dataset and annotation file

        """
        urls =[(self.csv_url,self.csv_path),(self.wav_url,self.wav_zip )]
        
        for u in urls:
            url=u[0]
            download_path=u[1]
            if not os.path.isfile(download_path): 
                print("Downloading the file "+ u[1])
                # open the link
                r = requests.get(url, stream=True) 
                # save the content to a file
                with open(u[1], 'wb') as f:
                    total_length = int(r.headers.get('content-length'))
                    for chunk in progress.bar(r.iter_content(chunk_size=8192), expected_size=(total_length/8192) + 1): 
                        if chunk:
                            f.write(chunk)
                            f.flush()
                f.close()
            else:
                print(download_path + " is already there!")
            
        
    def extract(self):
        """ Extract the downloaded dataset

        """
        if not os.listdir(os.path.join(self.root_path,"wav")):
            print("Extracting the dataset "+ self.dataset_name)
            zip= zipfile.ZipFile(self.wav_zip)
            zip.extractall(self.root_path)
        else:
            print(self.dataset_name + " has been already extracted!")
            
            
    def partition(self, n=10):
        """ Create a csv file with partitioning into n subsets
        Args:
            n:  number of subsets

        """
        if not os.path.isfile(self.csv_10_path): 
            data_list = pd.read_csv(self.csv_path)
            data_list['fold'] = np.random.randint( low=0, high=n, size=len(data_list))
            data_list.to_csv(self.csv_10_path)
            print("The partition into "+ str(n) + " is saved: "+ self.csv_10_path)
        else:
            print("The partition CSV file is already there! "+ self.csv_10_path)
            
