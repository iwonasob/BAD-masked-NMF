import config as cfg
from src.dataset import * 
from src.feature_extraction import *

if __name__ == "__main__":
    
    datasets=['warblrb','ff1010']

    for dataset_name in datasets:
        dataset = DatasetCreator(dataset_name)
        dataset.run()
        feature_extractor = FeatureExtraction(dataset_name)
        feature_extractor.run()
    
        
    