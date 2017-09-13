'''
SUMMARY:  downloads and extracts datasets
AUTHOR:   Iwona Sobieraj
Created:  2017.09.07
Modified: -
--------------------------------------
'''
import config as cfg
import os
import numpy as np
import cPickle
from nmf import NMF
np.random.seed(1515)
eps = np.spacing(1)

class Tester:
    def __init__(self):
                     
        """ Initialize class
        Args:
            dataset_name (string): Name of the dataset to prepare
        """
        self.n_sh           = cfg.n_sh
        self.feature        = cfg.feature
        self.type           = cfg.type
        self.update_func    = cfg.update_func
        self.iterations     = cfg.iterations
        self.rank_0         = cfg.rank_0
        self.rank_1         = cfg.rank_1
        self.W_path         = os.path.join(cfg.home_path, "W")
        self.W_name         = os.path.join(self.W_path, "W_"+self.feature+"_"+self.type+"_"+self.update_func+"_"+str(self.rank_1)+'p_'+str(self.rank_0)+'n_'+str(self.n_sh)+'sh.p')
        
    def run(self,X_test,y_test,clf):
        print("NMF of test files")
        test_list = Parallel(n_jobs=num_cores)(delayed(nmf_function.process)(W.shape[1], f, W0 = W,  iterations=100) for f in X_test)
        test_data_pooled =[(np.hstack((np.mean(sample[1], axis=1), np.std(sample[1], axis=1)))) for sample in test_list]
        
        print("Predicting with Random Forest")
        y_scores=clf.predict_proba(np.array(test_data_pooled))
        cPickle.dump( y_scores, open( results_name, 'wb' ), protocol=cPickle.HIGHEST_PROTOCOL )
        fpr, tpr, thresholds = metrics.roc_curve(te_y, y_scores[:,1])
        
        roc_auc = metrics.auc(fpr, tpr)
        
        ion()
        plt.plot(fpr,tpr,label=W_name)
        plt.legend(loc='best')
        print roc_auc