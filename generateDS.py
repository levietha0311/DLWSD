#pip3 install pandas numpy watchdog
#pip3 install tensorflow torch fastai
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
flowmeter = '/opt/netips/deepinspector/bin/flowmeter'
flowmeter_output =  '/opt/netips/deepinspector/flows/' 
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
names = ['Dst Port', 'Protocol', 'Timestamp', 'Flow Duration', 'Tot Fwd Pkts',
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

pcont_names = ['Timestamp','Flow Duration','Total Fwd Packet','Total Bwd packets','Total Length of Fwd Packet','Total Length of Bwd Packet','Fwd Packet Length Max','Fwd Packet Length Min','Fwd Packet Length Mean','Fwd Packet Length Std','Bwd Packet Length Max','Bwd Packet Length Min','Bwd Packet Length Mean','Bwd Packet Length Std','Flow Bytes/s','Flow Packets/s','Flow IAT Mean','Flow IAT Std','Flow IAT Max','Flow IAT Min','Fwd IAT Total','Fwd IAT Mean','Fwd IAT Std','Fwd IAT Max','Fwd IAT Min','Bwd IAT Total','Bwd IAT Mean','Bwd IAT Std','Bwd IAT Max','Bwd IAT Min','Fwd PSH Flags','Bwd PSH Flags','Fwd URG Flags','Bwd URG Flags','Fwd Header Length','Bwd Header Length','Fwd Packets/s','Bwd Packets/s','Packet Length Min','Packet Length Max','Packet Length Mean','Packet Length Std','Packet Length Variance','FIN Flag Count','SYN Flag Count','RST Flag Count','PSH Flag Count','ACK Flag Count','URG Flag Count','CWR Flag Count','ECE Flag Count','Down/Up Ratio','Average Packet Size','Fwd Segment Size Avg','Bwd Segment Size Avg','Fwd Bytes/Bulk Avg','Fwd Packet/Bulk Avg','Fwd Bulk Rate Avg','Bwd Bytes/Bulk Avg','Bwd Packet/Bulk Avg','Bwd Bulk Rate Avg','Subflow Fwd Packets','Subflow Fwd Bytes','Subflow Bwd Packets','Subflow Bwd Bytes','FWD Init Win Bytes','Bwd Init Win Bytes','Fwd Act Data Pkts','Fwd Seg Size Min','Active Mean','Active Std','Active Max','Active Min','Idle Mean','Idle Std','Idle Max','Idle Min']

gmodel = ''
sys.stderr = sys.stdout 
#logging.getLogger('tensorflow').disabled = True
wsfile = "/home/nnhoa/DLWSD/data/webshell.csv"
botfile = "/home/nnhoa/DLWSD/data/botattack.csv"
trainfile = "/home/nnhoa/DLWSD/data/trainN.csv"
testfile = "/home/nnhoa/DLWSD/data/testN.csv"

def changeTimestamp(t):
	return (parser.parse(t) - datetime(1970, 1, 1)).total_seconds()
def normalizeLabel(l):
    if l == "Benign" :
        return l
    return "Webshell"        
def binary(l):
    if l == "Benign" :
        return 0
    return 1
def generateDS(testsize=0.3):
    #webshell dataset
    print("Performing splitting webshell dataset...")
    df = pd.read_csv(wsfile) #, nrows=100000)
    df.replace([np.inf, - np.inf], np.nan, inplace = True)    
    df[dep_var] = df[dep_var].apply(normalizeLabel) 
    df = df.dropna() #, nrows=1000        
    classes = df.groupby(dep_var).count() #count classes
    n_0, n_1 = classes.iloc[0, 0], classes.iloc[1, 0]
    print("WS ", n_0, n_1)
    
    wstrain, wstest = train_test_split(df, test_size=testsize)
    classes = wstrain.groupby(dep_var).count() #count classes
    n_0, n_1 = classes.iloc[0, 0], classes.iloc[1, 0]
    print("WS-train ", n_0, n_1)
    classes = wstest.groupby(dep_var).count() #count classes
    n_0, n_1 = classes.iloc[0, 0], classes.iloc[1, 0]
    print("WS-test ", n_0, n_1)

    #botattack dataset
    print("Performing splitting botattack dataset...")
    df = pd.read_csv(botfile) #, nrows=100000)
    df.replace([np.inf, - np.inf], np.nan, inplace = True)    
    # df['Timestamp'] = df['Timestamp'].apply(changeTimestamp)      
    df[dep_var] = df[dep_var].apply(normalizeLabel) 
    df = df.dropna() #, nrows=1000  
    classes = df.groupby(dep_var).count() #count classes
    n_0, n_1 = classes.iloc[0, 0], classes.iloc[1, 0]
    print("Bot ", n_0, n_1)

    bottrain, bottest = train_test_split(df, test_size=testsize)
    classes = bottrain.groupby(dep_var).count() #count classes
    n_0, n_1 = classes.iloc[0, 0], classes.iloc[1, 0]
    print("Bot-train ", n_0, n_1)
    classes = bottest.groupby(dep_var).count() #count classes
    n_0, n_1 = classes.iloc[0, 0], classes.iloc[1, 0]
    print("Bot-test ", n_0, n_1)

    #merging data.iloc[train_idx]
    print("Merging dataset...")
    train = wstrain # wsdf.iloc[wstrain]
    train = train.append(bottrain) #botdf.iloc[bottrain])
    test = wstest # wsdf.iloc[wstest]
    test = test.append(bottest) #botdf.iloc[bottest])
    print("Saving training file...")
    train.to_csv(trainfile, index=False)
    print("Saving testing file...")
    test.to_csv(testfile, index=False)    
    print("Done!")

def loadData(fileName):
    dataFile = os.path.join(dataPath, fileName)
    pickleDump = '{}.pickle'.format(dataFile)
    df = pd.read_csv(dataFile) #, nrows=100000)
    df.replace([np.inf, - np.inf], np.nan, inplace = True)    
    df = df.dropna() #, nrows=1000    
    # df = df.astype(float)
    # df = shuffle(df)      
    # df.to_pickle(pickleDump)
    return df
    
    if os.path.exists(pickleDump):
        df = pd.read_pickle(pickleDump)
    else:
        df = pd.read_csv(dataFile)
        df = df.dropna()
        df = shuffle(df)
        df.to_pickle(pickleDump)
    return df
def cleandata(data):
    data.drop(['Flow ID'],axis=1, inplace=True) 
    data.drop(['Src IP'],axis=1, inplace=True) 
    data.drop(['Src Port'],axis=1, inplace=True) 
    data.drop(['Dst IP'],axis=1, inplace=True) 
    data.drop(['Label'],axis=1, inplace=True) 
    # data.columns = [names, 'Label']
    data['Timestamp'] = data['Timestamp'].apply(changeTimestamp)
#############################################
def getLabels(fileName, save=False):
    df = loadData(fileName)
    labels = df['Label'].unique()
    # labels.to_csv('classes.txt')
    print(labels)
    # np.savetxt('classes.txt', labels,delimiter=',')
    if (save):
        with open('classes.txt', 'w') as fout:
            for line in labels:
                fout.write(line + "\n")            
#############################################
def loadModel(model_name):    
    global gmodel    
    global gencoder
    global working_dir
    global labels
    
    modelFile= f"{working_dir}/models/{model_name}"    
    print ('Loading model: ',modelFile)
    gmodel = load_learner(modelFile) #learner.load(file=model_name) #file=modelFile)
    print ("Finished loading the model.")    

############################################## 
def train(dataFile, epochs=8, normalize=True, balancing=True):
    procs = [Categorify, FillMissing]
    if normalize:
        procs.append(Normalize)
    modelName = os.path.splitext(dataFile)[0]
    seed = 7
    np.random.seed(seed)
    # load data
    print('Loading dataset from %s...' % dataFile)
    df = loadData(dataFile)
    df[dep_var] = df[dep_var].astype('category')
    print('Setting model...' )
    # create model    
    data = TabularDataLoaders.from_df(df, path=dataPath, cat_names=cat_names, cont_names=cont_names, procs=procs, y_names=dep_var, bs=64)#, valid_idx=list(range(1,10000)))
                    #  .label_from_df(cols=dep_var).databunch()
    #config
    # tc = tabular_config(ps=[0.001, 0.01], embed_p=0.04)
    #for imbalance binary classify
    class_count_df = df.groupby(dep_var).count() #count classes
    n_0, n_1 = class_count_df.iloc[0, 0], class_count_df.iloc[1, 0]
    w_0 = (n_0 + n_1) / (2.0 * n_0) # compute weight for each class
    w_1 = (n_0 + n_1) / (2.0 * n_1)
    # y_batch = df[dep_var].tolist()
    # freq = np.histogram(y_batch)[0]
    # len_batch = len(y_batch)
    # loss_func.weight = torch.cuda.Tensor([[w[int(j[0])]] for j in y_batch])
    # loss_func.pos_weight = torch.cuda.FloatTensor([freq[-1]/len_batch, freq[0]/len_batch])
    print(f"Benign {n_0}/{n_1} Webshell : Weight {w_0} / {w_1}")        
    class_weights=torch.cuda.FloatTensor([w_0, w_1]) # Convert Weights To FloatTensor
    # Instantiate RocAucBinary Score
    roc_auc = RocAucBinary()
    loss_func = CrossEntropyLossFlat(weight=class_weights)            
    model = tabular_learner(data, layers=[400, 200], metrics=[accuracy, Precision(), F1Score(), Recall(), roc_auc])#, loss_func=loss_func)
    if balancing:
        model = tabular_learner(data, layers=[400, 200], metrics=[accuracy, Precision(), F1Score(), Recall(), roc_auc], loss_func=loss_func)
    print('Training model...' )
    model.fit_one_cycle(epochs)
    #model.fit_one_cycle(epochs, 1e-2)
    modelFile = os.path.join(modelPath, modelName)
    print('Saving model to file %s...' % modelFile)
    model.save(modelFile)
    modelExport = f"{working_dir}/models/{modelName}"
    model.export(modelExport)
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
    print('Accuracy: {:.2f}%; precision: {:.2f}%; F1: {:.2f}%; Recall: {:.2f}%; roc-auc: {:.2f}%; fpr: {:.2f}%; elapsed: {:.2f} s'.format(acc*100, precision*100, f1*100, recall*100, roc*100, fpr*100, elapsed ))        
    print('Trained successully!')
    del data
    del df
    gc.collect()

def predict(cvsfile):    
    global gencoder
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
    print('Elapsed time: {:.2f} s'.format(elapsed))    
    preds = preds.argmax(axis=1).tolist()
    # tests = tests.argmax(axis=1).tolist()
    tests = data[dep_var].apply(binary).tolist()    
    # print(preds, tests) 
    # print(yLabels, clas_idx)
    cm = confusion_matrix(tests, preds)
    print("Confusion Matrix:", cm)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)
    acc = accuracy_score(tests, preds)
    precision = precision_score(tests, preds)
    f1 = f1_score(tests, preds) 
    recall = recall_score(tests, preds)
    roc = roc_auc_score(tests, preds)
    print('Accuracy: {:.2f}%; precision: {:.2f}%; F1: {:.2f}%; Recall: {:.2f}%; roc-auc: {:.2f}%; fpr: {:.2f}%; elapsed: {:.2f} s'.format(acc*100, precision*100, f1*100, recall*100, roc*100, fpr*100, elapsed ))        
    print('Finish prediction!')
    del data
    # del df
    gc.collect()
#############################################
def kfoldTest(dataFile, epochs=5, normalize=True, nfold=10):
    procs = [Categorify, FillMissing]
    if normalize:
        procs.append(Normalize)
    seed = 7
    np.random.seed(seed)
    # load data
    print('Loading dataset from %s...' % dataFile)
    data = loadData(dataFile)
    # cleandata(data)    
    # data['Timestamp'] = data['Timestamp'].apply(changeTimestamp)        
    data[dep_var] = data[dep_var].apply(normalizeLabel)
    # data.to_csv('data.csv', index=False)
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
        # create model and learn
        # class_count_df = data.iloc[train_idx].groupby(dep_var).count() #count classes
        # n_0, n_1 = class_count_df.iloc[0, 0], class_count_df.iloc[1, 0]
        w_0 = (n_0 + n_1) / (2.0 * n_0) # compute weight for each class
        w_1 = (n_0 + n_1) / (2.0 * n_1)
        class_weights=torch.cuda.FloatTensor([w_0, w_1]) # Convert Weights To FloatTensor
        # # Instantiate RocAucBinary Score
        loss_func = CrossEntropyLossFlat(weight=class_weights)                
        roc_auc = RocAucBinary()
        model = tabular_learner(data_fold, layers=[400, 200], metrics=[accuracy, Precision(), F1Score(), Recall(), roc_auc], loss_func=loss_func)
        print('Training model...' )
        model.fit_one_cycle(epochs)
        
        # modelFile = os.path.splitext(dataFile)[0]
        # modelFile = os.path.join(modelPath, os.path.basename(modelFile))
        # modelFile = f"{modelFile}.f{fold}"
        # print("Saving model to ", modelFile)
        # model.save(modelFile)

        # train the model, iterating on the data in batches of batch_size
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
        # print('loss {}; accuracy: {:.2f}%; precision: {:.2f}%; F1: {:.2f}%; Recall: {:.2f}%; roc-auc: {:.2f}%;fpr: {:.2f}%; elapsed: {:.2f} s'.format(loss, acc*100, precision*100, f1*100, recall*100, roc*100, fpr*100, elapsed ))
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
        kfoldTest(sys.argv[2],  epochs=1, nfold=5, normalize=True)
    elif sys.argv[1] == 'label':
        getLabels(sys.argv[2])
    elif sys.argv[1] == 'DS':
        generateDS(testsize=0.3)
    elif sys.argv[1] == 'train':
        train(sys.argv[2],  epochs=1, normalize=True, balancing=False)
    elif sys.argv[1] == 'predict':
        loadModel("trainN")
        predict(sys.argv[2])
    else:
        print("Check syntax!")
        
