Bird Audio Detection
=========================
Source code for Eusipco2017 paper: [Masked Non-negative Matrix Factorization for Bird Detection Using Weakly Labeled Data](http://epubs.surrey.ac.uk/842222/1/IwonaSobieraj_EUSIPCO2017.pdf) 


Authors:
- [Iwona Sobieraj](https://iwonasob.github.io/) (<i.sobieraj@surrey.ac.uk>)
- Quiqiang Kong 
- Mark Plumbley 

## 1. Installation


The system is developed for [Python 2.7.0](https://www.python.org/). The system is tested only with Linux operating system. 

Run to ensure that all external modules are installed

    pip install -r requirements.txt
    
    
## 2. Usage


To reproduce the results from the paper do the following:
    
1. Modify the working path home_path in config.py
2. Run the system. It will download the datasets, preapre the training data, train and test the system.

`python BirdAudioDetection.py`
    
    
