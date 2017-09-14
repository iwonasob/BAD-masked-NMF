'''
Main script to download and prepare the data and run Bird Audio Detection
'''

import config as cfg
from src.dataset import * 
from src.feature_extraction import *
from src.data_preparation import *
from src.train import *
from src.evaluate import *
import numpy as np

if __name__ == "__main__":
    
    if cfg.mode == 'dev':
        tr_fold=[1]
        test_fold=[2]
    elif cfg.mode == 'eval':
        tr_fold=range(10)
        test_fold=[]
    

    datasets=['warblrb','ff1010']
    
    for dataset_name in datasets:
        dataset = DatasetCreator(dataset_name)
        dataset.run()
        feature_extractor = FeatureExtraction(dataset_name)
        feature_extractor.run()
        
    data_preparation = DataPreparation(datasets,tr_fold,test_fold)
    [X_train_1, X_train_0, X_test, y_test] = data_preparation.run()
    
    trainer = Trainer()
    trainer.run_nmf(np.hstack(X_train_1),np.hstack(X_train_0))
    [clf,W]=trainer.run_rf(X_train_1,X_train_0)
    
    evaluator = Evaluator()
    evaluator.run(X_test, y_test, clf, W)
   


    


    
        
