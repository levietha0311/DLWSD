import logging
import csv
import os
import sys
import numpy as np
import pandas as pd
import operator
import gc
import time
import datetime
import tensorflow as tf
import torch
from dateutil import parser

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import *
from tensorflow.keras.utils import to_categorical, normalize
from timeit import default_timer as timer
from fastai import *
from fastai.learner import *
from fastai.tabular import *
from fastai.tabular.all import *
from fastai.basics import *
from fastai.callback.all import *
from fastai.distributed import *
from fastprogress import fastprogress
from fastai.callback.mixup import *
from fastcore.script import *

torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 80

###################
#setting GPU Mem
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
#############################################
working_dir='/home/nnhoa/DLWSD'

dataPath = '/home/nnhoa/DLWSD/data'
resultPath = '/home/nnhoa/DLWSD/results'
modelPath = '/home/nnhoa/DLWSD/models'

prediction_model = 'multiclass2018'
#labels = ['Benign', 'BruteForce-Web', 'BruteForce-XSS', 'SQL-Injection']
# labels = ['Benign', 'Infilteration', 'DDOS attack-HOIC', 'DDOS attack-LOIC-UDP', 'DoS attacks-Slowloris', 'DoS attacks-GoldenEye', 'DoS attacks-Hulk', 'DoS attacks-SlowHTTPTest','SQL Injection', 'Brute Force - XSS', 'Brute Force - Web',  'Bot']
#labels = ['Benign', 'DoS attacks-SlowHTTPTest', 'DDOS attack-HOIC', 'DoS attacks-Hulk', 'Bot', 'Infilteration', 'DoS attacks-Slowloris', 'DDOS attack-LOIC-UDP', 'DoS attacks-GoldenEye', 'SQL Injection', 'Brute Force -Web', 'Brute Force -XSS']
# labels = ['Benign','Bot','Brute Force -Web','Brute Force -XSS','DDOS attack-HOIC','DDOS attack-LOIC-UDP','DoS attacks-GoldenEye','DoS attacks-Hulk','DoS attacks-SlowHTTPTest','DoS attacks-Slowloris','Infilteration','SQL Injection']
labels = ['Benign', 'Webshell']

cat_names = ['Dst Port', 'Protocol']
dep_var = 'Label'
cont_names = ['Timestamp', 'Flow Duration', 'Tot Fwd Pkts',
              'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
              'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
              'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean',
              'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean',
              'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot',
              'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
              'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
              'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
              'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s',
              'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean',
              'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt',
              'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt',
              'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
              'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg',
              'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg',
              'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts',
              'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
              'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts',
              'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max',
              'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']
#############################################
def loadData(fileName):
    dataFile = os.path.join(dataPath, fileName)
    df = pd.read_csv(dataFile) #, nrows=100000)
    df = df.dropna() #, nrows=1000    
    return df
def cleandata(data):
    data.drop(['Flow ID'],axis=1, inplace=True) 
    data.drop(['Src IP'],axis=1, inplace=True) 
    data.drop(['Src Port'],axis=1, inplace=True) 
    data.drop(['Dst IP'],axis=1, inplace=True) 
    data.drop(['Label'],axis=1, inplace=True) 
    data['Timestamp'] = data['Timestamp'].apply(changeTimestamp)
def changeTimestamp(t):
	return (parser.parse(t) - datetime(1970, 1, 1)).total_seconds()
#############################################
def loadModel(model_name):   
    global gmodel    
    modelFile= f"{working_dir}/models/{model_name}"    
    print ('Loading model: ',modelFile)
    gmodel = load_learner(modelFile) 
    print ("Finished loading the model.")    
##############################################    
def predict(cvsfile):    
    global gmodel
    global labels
    dataFile = os.path.join(dataPath, cvsfile)
    print("Predicting %s..." % dataFile)
    # data = pd.read_csv(dataFile, nrows=100).dropna()
    data = pd.read_csv(dataFile).dropna()
    data[dep_var] = data[dep_var].astype('category')	
    # yLabels = data.pop('Label').tolist()     		
    yLabels = data['Label']
    class_count_df = data.groupby(dep_var).count() #count classes
    n_0, n_1 = class_count_df.iloc[0, 0], class_count_df.iloc[1, 0]
    print(f"Benign {n_0} - Webshell {n_1}")       

    print("Preparing test_dl...")
    dl = gmodel.dls.test_dl(data, with_labels=True, drop_last=False)    
    print("Predicting...")    
    start = time.time()
    preds, tests, clas_idx = gmodel.get_preds(dl=dl, with_decoded=True)  
    elapsed = time.time() - start
    preds = preds.argmax(axis=1).tolist()
    tests = data[dep_var].apply(binary).tolist()    
    cm = confusion_matrix(tests, preds)
    print("Confusion Matrix:", cm)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)
    acc = accuracy_score(tests, preds)
    precision = precision_score(tests, preds)
    f1 = f1_score(tests, preds) 
    recall = recall_score(tests, preds)
    roc = roc_auc_score(tests, preds)
    print('Accuracy: {:.2f}%; Precision: {:.2f}%; F1: {:.2f}%; Recall: {:.2f}%; roc-auc: {:.2f}%; FPR: {:.2f}%; Elapsed: {:.2f} s'.format(acc*100, precision*100, f1*100, recall*100, roc*100, fpr*100, elapsed ))        
    print('Finish prediction!')

############################################## 
def normalizeLabel(l):
    if l == "Benign" :
        return l
    return "Webshell"
def binary(l):
    if l == "Benign" :
        return 0
    return 1    
############################################## 
def trainWS(dataFile, epochs=8, normalize=True):
    procs = [Categorify, FillMissing]
    if normalize:
        procs.append(Normalize)
    modelName = os.path.splitext(dataFile)[0]
    seed = 7
    np.random.seed(seed)
    # load data
    print('Loading dataset from %s...' % dataFile)
    df = loadData(dataFile)
    df[dep_var] = df[dep_var].apply(normalizeLabel)
    df[dep_var] = df[dep_var].astype('category')
    print('Setting model...' )
    # create model    
    data = TabularDataLoaders.from_df(df, path=dataPath, cat_names=cat_names, cont_names=cont_names, procs=procs, y_names=dep_var, bs=64)
    
    #config
    # tc = tabular_config(ps=[0.001, 0.01], embed_p=0.04)
    # create model and learn
    #for imbalance binary classify
    class_count_df = df.groupby(dep_var).count() #count classes
    n_0, n_1 = class_count_df.iloc[0, 0], class_count_df.iloc[1, 0]
    w_0 = (n_0 + n_1) / (2.0 * n_0) # compute weight for each class
    w_1 = (n_0 + n_1) / (2.0 * n_1)
    print(f"Malicious {n_0}/{n_1} Benign : Weight {w_0} / {w_1}")        
    class_weights=torch.cuda.FloatTensor([w_0, w_1]) # Convert Weights To FloatTensor
    # Instantiate RocAucBinary Score
    roc_auc = RocAucBinary()
    loss_func = CrossEntropyLossFlat(weight=class_weights)        
    model = tabular_learner(data, layers=[400, 200], loss_func=loss_func, metrics=[accuracy, Precision(), F1Score(), Recall(), roc_auc] )  
    # model = tabular_learner(data, layers=[400, 100], metrics=accuracy, config=tc ) #, callback_fns=ShowGraph) [accuracy_multi]
    # train the model, iterating on the data in batches of batch_size
    n_gpu = torch.cuda.device_count()
    ctx = model.distrib_ctx if num_distrib() and n_gpu else model.parallel_ctx
    
    print('Training model...' )
    with ctx(): model.fit_one_cycle(epochs, 1e-2)
    # model.fit(epochs, 1e-2)
    modelFile = os.path.join(modelPath, modelName)
    print('Saving model to file %s...' % modelFile)
    model.save(modelFile)
    modelExport = f"{working_dir}/models/{modelName}"
    model.export(modelExport)
    # evaluate the model
    # model.summary()
    loss, roc, acc = model.validate()
    print('loss {}: accuracy: {:.2f}%'.format(loss, acc*100))
    cvscores = []
    cvscores.append(acc*100)
    resultFile = os.path.join(resultPath, modelName)
    with open('{}.result'.format(resultFile), 'a') as fout:
        fout.write('accuracy: {:.2f} std-dev: {:.2f}\n'.format(np.mean(cvscores), np.std(cvscores)))
        print('accuracy: {:.2f} std-dev: {:.2f}\n'.format(np.mean(cvscores), np.std(cvscores)))
    print('Trained successully!')
    
    del data
    del df
    gc.collect()

def kfoldTest(dataFile, epochs=5, normalize=True, nfold=10):
    procs = [Categorify, FillMissing]
    if normalize:
        procs.append(Normalize)
    seed = 7
    np.random.seed(seed)
    # load data
    print('Loading dataset from %s...' % dataFile)
    data = loadData(dataFile)
    data[dep_var] = data[dep_var].apply(normalizeLabel)
    data[dep_var] = data[dep_var].astype('category')
    class_count_df = data.groupby(dep_var).count() #count classes
    n_0, n_1 = class_count_df.iloc[0, 0], class_count_df.iloc[1, 0]
    print("Total", len(data), " , Benign: ", n_0, " , WS: ", n_1 )    

    # define 6-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed)
    cvscores = []
    fold = 1
    for train_idx, test_idx in kfold.split(data.index, data[dep_var]):
        print('Running fold = ', fold ) #, train_idx, test_idx)        
        # create model
        data_fold = TabularDataLoaders.from_df(data, path=dataPath, cat_names=cat_names, cont_names=cont_names, procs=procs, y_names=dep_var, bs=64, valid_idx=test_idx, train_idx = train_idx)        
        print("Setting model...")
        # balancing dataset
        w_0 = (n_0 + n_1) / (2.0 * n_0) # compute weight for each class
        w_1 = (n_0 + n_1) / (2.0 * n_1)
        class_weights=torch.cuda.FloatTensor([w_0, w_1]) # Convert Weights To FloatTensor
        # # Instantiate RocAucBinary Score
        loss_func = CrossEntropyLossFlat(weight=class_weights)                
        roc_auc = RocAucBinary()
        model = tabular_learner(data_fold, layers=[400, 200], metrics=[accuracy, Precision(), F1Score(), Recall(), roc_auc], loss_func=loss_func)
        print('Training model...' )
        model.fit_one_cycle(epochs)
                
        # evaluate the model
        start = time.time()
        loss, acc, precision, f1, recall, roc = model.validate()
        elapsed = time.time() - start
        interp = ClassificationInterpretation.from_learner(model)
        upp, low = interp.confusion_matrix()
        tn, fp = upp[0], upp[1]
        fn, tp = low[0], low[1]
        fpr = fp / (fp + tn) 
        print("Confusion Matrix:")
        print(tn, fp)
        print(fn, tp)
        print('{}; accuracy: {:.2f}%; precision: {:.2f}%; F1: {:.2f}%; Recall: {:.2f}%; roc-auc: {:.2f}%; fpr: {:.2f}%; elapsed: {:.2f} s'.format(fold, acc*100, precision*100, f1*100, recall*100, roc*100, fpr*100, elapsed ))
        cvscores.append(acc*100)
        resultFile = os.path.join(resultPath, dataFile)
        with open('{}.result'.format(resultFile), 'a') as fout:
            fout.write('accuracy: {:.2f} std-dev: {:.2f}\n'.format(np.mean(cvscores), np.std(cvscores)))
        fold += 1
    del data    
    gc.collect()

           
############### MAIN APP ####################     
if __name__ == "__main__":
    
    if torch.cuda.is_available():
        print("GPU enabling...")
        torch.cuda.device('cuda')
    else:
        print("No GPU")
    
    if len(sys.argv) < 2:
        print('Usage: python DLWSD.py action [inputDataFile.csv]')
    elif sys.argv[1] == 'kfold':
        kfoldTest(sys.argv[2],  epochs=3, nfold=5, normalize=True)
    elif sys.argv[1] == 'train':
        trainWS(sys.argv[2],  epochs=6, normalize=True)
    elif sys.argv[1] == 'predict':
        # loadModel("WS1")
        # predict(sys.argv[2])
        loadModel("DLWSD")
        predict(sys.argv[2])
        loadModel("DLWSD+B")
        predict(sys.argv[2])
    else:
        print("Check syntax!")
        
