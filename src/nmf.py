"""
Non-negative Matrix Factorization algorithms.
Many can be unified under a single framework based on multiplicative updates.
Implementation is inspired by NMFlib by Graham Grindlay
"""
import numpy as np
np.random.seed(1515)

class NMF():
    
    def __init__(self, rank, update_func = "kl", iterations = 100, 
        threshold = None, norm_W = 2, norm_H = 0, 
        update_W = True, update_H = True, verbose=False):

        self.rank = rank
        self.update = getattr(self, update_func + "_updates")
        self.iterations = iterations
        self.threshold = threshold
        self.norm_W = norm_W
        self.norm_H = norm_H
        self.update_W = update_W
        self.update_H = update_H
        self.update_func = update_func
        self.verbose=verbose

    """
    Compute divergence between reconstruction and original
    """    
    def compute_error(self, V, W, H):
        eps = np.spacing(1)
        R=np.dot(W,H)
        if self.update_func == "kl" or self.update_func == "kls":
            err = np.sum(np.multiply(V,np.log((V+eps)/(R+eps))) - V + R)
        elif self.update_func == "is":
            err = np.sum((V+eps)/(R+eps)-np.log((V+eps)/(R+eps))-1)
        elif self.update_func == "eucl":
            err = np.sum((V-R)**2)
        else:
            raise ValueError("Unknown metric" + self.update_func )
        return err        

    """
    Normalize W and/or H depending on initialization options
    """    
    def normalize_W(self, W):        
        if self.norm_W == 1:            
            W = W/(np.sum(W,axis=0))
        if self.norm_W == 2:
            W= W/(np.sqrt(np.sum(W**2,axis=0)))
        return W
        
    def normalize_H(self, H):
        if self.norm_H == 1:            
            H = H/(np.sum(H,axis=1))
        if self.norm_H == 2:
            H= H/(np.sqrt(np.sum(H**2,axis=1)))
        return H

    def normalize(self, W, H):
        W = self.normalize_W(W)
        H = self.normalize_H(H)
        return [W,H]
    
    """
    Initialize and compute multiplicative updates iterations
    """                
    def process(self, V, lam=0, alpha=None, split_size=None, W0 = None, H0 = None):
         eps = np.spacing(1)
         W = W0 if W0 is not None else np.random.rand(V.shape[0],self.rank)+eps
         H = H0 if H0 is not None else np.random.rand(self.rank, V.shape[1])+eps
         self.ones = np.ones(V.shape)
         [W, H] = self.normalize(W, H)
         err=[]
         for i in range(self.iterations):
            [V, W, H] = self.update(V, W, H, lam, alpha, split_size)
            err = self.compute_error(V, W, H)
            if self.threshold is not None:
                err = self.compute_error(V, W, H)
                if err <= self.threshold:
                     return [W, H, err]
            if self.verbose == True:
                print err
         return [W, H, err]

    """
    Multiplicative updates functions
    """    

    """
    Optimize Kullback-Leibler divergence
    """    
    def kl_updates(self, V, W, H,lam, alpha, split_size):
        eps = np.spacing(1)
        if self.update_W:
            R = np.dot(W,H)
            W *= np.dot(np.divide(V, R + eps) , H.T) / (np.dot(self.ones, H.T) + eps)
            W = self.normalize_W(W)
        if self.update_H: 
            R = np.dot(W,H)
            H *= np.dot(W.T, np.divide(V, R + eps)) / (np.dot(W.T, self.ones) + eps)
            H = self.normalize_H(H)
        return [V, W, H]
        

    def kls_updates(self, V, W, H,lam,alpha,split_size):
        eps = np.spacing(1)
        if self.update_W:
            R = np.dot(W,H)
            W *= np.dot(np.divide(V, R + eps) , H.T) / (np.dot(self.ones, H.T) + eps)
            W = self.normalize_W(W)
        if self.update_H: 
            R = np.dot(W,H)
            H *= np.dot(W.T, np.divide(V, R + eps)) / (np.dot(W.T, self.ones) + eps+ lam)
            H = self.normalize_H(H)
        return [V, W, H]
        
    def klc_updates(self, V, W, H, lam, alpha, split_size):
        eps = np.spacing(1)
        [W1, W2] = np.split(W, split_size)
        [V1, V2] = np.split(V, split_size)
        if self.update_W:
            R1 = np.dot(W1,H)
            W1 *= np.dot(np.divide(V1, R1 + eps) , H.T) / (np.dot(np.ones(V1.shape), H.T) + eps)
            W1 = self.normalize_W(W1)
            R2 = np.dot(W2,H)
            W2 *= np.dot(np.divide(V2, R2 + eps) , H.T) / (np.dot(np.ones(V2.shape), H.T) + eps)
            W2 = self.normalize_W(W2)
        if self.update_H: 
            R1 = np.dot(W1,H)
            R2 = np.dot(W2,H)
            H *= ((1-alpha)*np.dot(W1.T, np.divide(V1, R1 + eps)) + alpha*np.dot(W2.T, np.divide(V2, R2 + eps)))/ (1-alpha)*(np.dot(W1.T, np.ones(V1.shape)) + alpha*np.dot(W2.T, np.ones(V2.shape))+ eps+ lam)
            H = self.normalize_H(H)
        W= np.vstack((W1,W2))
        return [V, W, H]
        

    """
    Optimize Itakura-Saito divergence
    """ 
    def is_updates(self, V, W, H, lam, alpha, split_size):
        eps = np.spacing(1)
        if self.update_W:
            R = np.dot(W,H)
            W*= ((np.dot(((R+eps)**(-2.0)*V), H.T) + eps)/(np.dot((R+eps)**(-1.0), H.T) + eps))
            W = self.normalize_W(W)
        if self.update_H: 
            R = np.dot(W,H)
            H*= (np.dot(W.T, (R+eps)**(-2.0)*V) + eps)/(np.dot(W.T, (R+eps)**(-1.0)) + eps)
            H = self.normalize_H(H)

        return [V, W, H]
        
    """
    Optimize Euclidean distance
    """ 
    def eucl_updates(self, V, W, H):
        eps = np.spacing(1)
        if self.update_W:
            R = np.dot(W,H)
            W*= ((np.dot(((R+eps)**(-2.0)*V), H.T) + eps)/(np.dot((R+eps)**(-1.0), H.T) + eps))
            W = self.normalize_W(W)
        if self.update_H: 
            R = np.dot(W,H)
            H*= (np.dot(W.T, (R+eps)**(-2.0)*V) + eps)/(np.dot(W.T, (R+eps)**(-1.0)) + eps)
            H = self.normalize_H(H)

        return [V, W, H]

#### function for multiprocessing
def process_parallel(rank, V, W0 = None, lam=0, H0 = None,verbose=False, iterations=100):
    eps = np.spacing(1)
    W = W0 if W0 is not None else np.random.rand(V.shape[0],rank)+eps
    H = H0 if H0 is not None else np.random.rand(rank, V.shape[1])+eps
    for i in range(iterations):
        [V, W, H] = update(V, W, H, lam)
        err = compute_error(V, W, H)
        if verbose == True:
            print err
    return [W, H, err]
        
   
def compute_error(V, W, H):
    eps = np.spacing(1)
    R=np.dot(W,H)
    err = np.sum(np.multiply(V,np.log((V+eps)/(R+eps))) - V + R)
    return err        

def update(V, W, H, lam):
    eps = np.spacing(1)
    # if self.update_W:
        # R = np.dot(W,H)
        # W *= np.dot(np.divide(V, R + eps) , H.T) / (np.dot(self.ones, H.T) + eps)
    R = np.dot(W,H)
    H *= np.dot(W.T, np.divide(V, R + eps)) / (np.dot(W.T, np.ones(V.shape)) + lam + eps)
    return [V, W, H]