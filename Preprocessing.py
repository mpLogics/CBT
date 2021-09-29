from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd


class Preprocess():
    def __init__(self) -> None:
        self.MAX_SEQUENCE_LENGTH = 250
        self.EMBEDDING_DIM = 100
        self.MAX_NB_WORDS = 50000
        
        self.labels = {0:['Sample','Samples'],
                      1:['Situation','Situations'],
                      2:['Emotion','Emotions'],
                      3:['Thinking_Error','Thinking Error']}
        
        self.filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
    
    def preProcess(self,data):
        #Finding unique tokens in the corpora using tokenization
        # The maximum number of words to be used. (most frequent)
        tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS, filters=self.filters, lower=True)
        for i in range(len(self.labels)):
            tokenizer.fit_on_texts(data[self.labels[i][0]].values)
            word_index = tokenizer.word_index
            
        X = tokenizer.texts_to_sequences(data[self.labels[0][0]].values)
        X = pad_sequences(X, maxlen=self.MAX_SEQUENCE_LENGTH)
        Y = {}
        for i in range(1,len(self.labels)):
            Y[self.labels[i][1]] = pd.get_dummies(data[self.labels[i][0]]).values
        return X,Y,tokenizer,self.labels