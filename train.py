from keras.callbacks import EarlyStopping
from tensorflow import keras
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D


class Train():
    
    def __init__(self) -> None:
        pass
    
    def buildModel(self,input_shape,OP_Dim, embedding_dim):
        model = keras.Sequential()
        model.add(Embedding(self.MAX_NB_WORDS, embedding_dim, input_length=input_shape))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100,dropout = 0.2, recurrent_dropout = 0.2, return_sequences = False))
        model.add(Dense(OP_Dim, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

    def train(self,model,X_train,Y_train,epochs,batch_size):
        history = model.fit(X_train, Y_train, 
                            epochs=epochs, 
                            batch_size=batch_size,
                            validation_split=0.1,
                            callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
        return history, model
    
    def evaluateModels(self,ModelStack,DataGen):
        testAcc = []
        for i in range(1,len(self.labels)):
            accr = ModelStack[self.labels[i][1]][1].evaluate(DataGen[self.labels[i][1]][0],DataGen[self.labels[i][1]][1])
            print(self.labels[i][1],': Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
            testAcc.append(accr[1])
        return testAcc