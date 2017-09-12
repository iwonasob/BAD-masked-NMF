'''
SUMMARY:  downloads and extracts datasets
AUTHOR:   Iwona Sobieraj
Created:  2017.09.07
Modified: -
--------------------------------------
'''
import config as cfg
import os
import pandas as pd
import cPickle
import numpy as np
from IPython.core.debugger import Tracer

class DataPreparation:
    def __init__(self,
                 datasets,
                 tr_fold,
                 test_fold):
                     
        """ Initialize class
        Args:
            datasets (list): List of datasets to use
        """
        self.datasets   = datasets
        self.csv        = "cv10.csv"
        self.tmp_path   = os.path.join(cfg.home_path, "tmp")
        self.tr_fold    = tr_fold
        self.test_fold  = test_fold
        
        if not os.path.isdir(self.tmp_path):
            os.makedirs(self.tmp_path)
            
    def run(self):
        data_dict = {}
        [data_dict['X_train'], data_dict['y_train'], data_dict['itemid_train']]=self.combine_datasets(self.tr_fold)   
        [data_dict['X_test'], data_dict['y_test'], data_dict['itemid_test']]=self.combine_datasets(self.test_fold)
        dump_file = os.path.join(self.tmp_path, "datadump_"+str(self.tr_fold)+"_"+str(self.test_fold)+".p")
        cPickle.dump( data_dict, open( dump_file, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
        print("Training and test data saved to: " + dump_file)
    
    def combine_datasets(self, fold_n):
        X_all,label_all,itemid_all=[],[],[]
        for dataset in self.datasets:
            [X, label, itemid] = self.load_data(dataset,fold_n)
            X_all.append(X)
            label_all.append(label)
            itemid_all.append(itemid)
        return [np.vstack(X_all), np.hstack(label_all), np.hstack(itemid_all)]    
    
    def load_data(self, dataset,fold_n):
        print("Loading dataset " + dataset + " fold "+ str(fold_n))
        csv_path=os.path.join(cfg.home_path, dataset, self.csv)
        data = pd.read_csv(csv_path)
        data_part=data[data['fold'].isin(fold_n)]
        label=data_part['hasbird'].values
        itemid=data_part['itemid'].values
        paths=[os.path.join(cfg.home_path, dataset, cfg.feature, str(i)+".f") for i in itemid]
        X = [cPickle.load(open(path, 'rb')) for path in paths]
        return [np.vstack(X), label, itemid]
