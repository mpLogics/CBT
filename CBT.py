import pandas as pd
from train import Train
from Preprocessing import Preprocess
from Visualizer import Visualize
import numpy as np
from sklearn.model_selection import train_test_split


class CBT():
    def __init__(self):
        self.MAX_SEQUENCE_LENGTH = 250
        self.EMBEDDING_DIM = 100
        self.MAX_NB_WORDS = 50000    
        self.filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
    
    def readData(self,path = 'Try3.csv'):
        data = pd.read_csv(path)
        print(data.head())
        return data
    
if __name__=='__main__':
    predictors = 3
    OP_Dim =[]
    epochs = 10
    batch_size = 10
    therapy = CBT()
    path_default = r'data/Try3.csv'
    
    try:
        path = input("Enter Data path: ")
        data = therapy.readData(path)
    except FileNotFoundError:
        print("Incorrect path entered. Using default path")
        path = path_default
        data = therapy.readData()
    
    Visualize.plotData(data)
    
    X,Y,tokenizer,labels = Preprocess.preProcess(data)
    input_shape = X.shape[1]

    Infer_labels=[]
    for i in range(1,len(labels)):
        Infer_labels.append(list(pd.get_dummies(data[labels[i][0]]).columns))
    for i in range(1,len(labels)):
        OP_Dim.append((np.unique(data[labels[i][0]])))                 

    
    DataGen = {}
    ModelStack = {}
    embedding_dims = [100, 100, 100]

    for i in range(1,len(labels)):
        model = therapy.buildModel(input_shape,OP_Dim[i-1].shape[0], embedding_dims[i-1])
        X_train, X_unseen, Y_train, Y_unseen = train_test_split(X,Y[labels[i][1]], test_size = 0.10, random_state = 42)
        logs, model = Train.train(model,X_train,Y_train,epochs,batch_size)
        ModelStack[labels[i][1]] = [logs, model]
        DataGen[labels[i][1]] = [X_unseen, Y_unseen]

    Visualize.infer(ModelStack,tokenizer,Infer_labels,DataGen)


    from sklearn import svm
    models_test = []
    models_acc = []
    for i in range(1,len(labels)):
        
        X_train, X_unseen, Y_train, Y_unseen = train_test_split(X,Y[labels[i][1]], test_size = 0.10, random_state = 42)
        clf = svm.SVC()
        clf.fit(X_train, np.argmax(Y_train, axis = 1))
        models_acc.append(clf.score(X_train, np.argmax(Y_train, axis = 1)))
        models_test.append(clf.score(X_unseen, np.argmax(Y_unseen, axis = 1)))
        print(labels[i][1], "Train Acc: {:0.2f} Test Acc: {:0.2f}".format(models_acc[-1]*100, models_test[-1]*100))