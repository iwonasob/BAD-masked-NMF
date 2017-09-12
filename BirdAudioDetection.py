import config as cfg
from src.dataset import * 
from src.feature_extraction import *
from src.data_preparation import *

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
    data_preparation.run()
    
    
    
        
