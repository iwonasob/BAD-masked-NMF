'''
Run system on test files, evaluate with ROC curves
'''

import config as cfg
import os
import numpy as np
import cPickle
import multiprocessing
from sklearn import  ensemble, metrics
from joblib import Parallel, delayed
from nmf import NMF, process_parallel

np.random.seed(1515)
eps = np.spacing(1)
num_cores = multiprocessing.cpu_count()


class Evaluator:
    """
    Tests and evaluates the system
    """   
    def __init__(self):
        self.iterations     = cfg.iterations
        self.n_sh           = cfg.n_sh
        self.feature        = cfg.feature
        self.type           = cfg.type
        self.update_func    = cfg.update_func
        self.rank_0         = cfg.rank_0
        self.rank_1         = cfg.rank_1
        self.results_path   = os.path.join(cfg.home_path, "results")
      
        self.results_name   = os.path.join(self.results_path, 'W_'+self.feature+'_'+self.type+'_'+self.update_func+'_'+str(self.rank_1)+'p_'+str(self.rank_0)+'n_'+str(self.n_sh)+'sh.p')
        
        if not os.path.isdir(self.results_path):
            os.makedirs(self.results_path)
        
    def run(self,X_test,y_test,clf,W):
        """ Run evaluation
        Args:
            X_test: list of test spectrograms
            y_test: list of labels
            clf:    trained classifier
            W:      trained dictionary
        Output:
            NONE, the results are saved in a file
        """   
        print("NMF of test files")
        test_list = Parallel(n_jobs=num_cores)(delayed(process_parallel)(W.shape[1], f, W0 = W,  iterations=self.iterations) for f in X_test)
        test_data_pooled =[(np.hstack((np.mean(sample[1], axis=1), np.std(sample[1], axis=1)))) for sample in test_list]
        
        print("Predicting with Random Forest")
        y_scores=clf.predict_proba(np.array(test_data_pooled))
        cPickle.dump(y_scores, open(self.results_name, 'wb' ), protocol=cPickle.HIGHEST_PROTOCOL )
        
        if len(y_test) > 0:
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_scores[:,1])
            roc_auc = metrics.auc(fpr, tpr)
            print roc_auc