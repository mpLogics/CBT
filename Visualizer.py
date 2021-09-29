import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

class Visualize():
    def __init__(self) -> None:
        self.labels = {0:['Sample','Samples'],
                      1:['Situation','Situations'],
                      2:['Emotion','Emotions'],
                      3:['Thinking_Error','Thinking Error']}
        self.MAX_SEQUENCE_LENGTH = 250

    def plotData(self,data):
        for i in range(1,len(self.labels)):
            data[self.labels[i][0]].value_counts().sort_values(ascending=False).iplot(kind='bar', yTitle='Number of Samples', title=self.labels[i][1]) 
    
    def plotLogs(self,ModelStack):
        plt.figure(figsize=(15,20))
        k = 1
        for i in range(1,len(self.labels)):
            plt.subplot(3,2,k)
            plt.title("Loss vs Epochs - " + (str)(self.labels[i][1]))
            plt.plot(ModelStack[self.labels[i][1]][0].history['loss'],label = "Training")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.plot(ModelStack[self.labels[i][1]][0].history['val_loss'],label = "Validation")
            plt.legend()
            k+=1
            plt.subplot(3,2,k)
            plt.title("Accuracy vs Epochs - " + (str)((self.labels[i][1])))
            plt.plot(ModelStack[self.labels[i][1]][0].history['accuracy'],label = "Training")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.plot(ModelStack[self.labels[i][1]][0].history['val_accuracy'],label = "Validation")
            plt.xlabel("Epochs")
            plt.legend()
            k+=1
        plt.show()
    
    def infer(self,ModelStack,tokenizer,infer_labels,DataGen):
        Predicted = {}
        for i in range(1,len(self.labels)):
            model_OP = []
            for j in range(len(DataGen[self.labels[i][1]][0])):
                pred = ModelStack[self.labels[i][1]][1].predict(DataGen[self.labels[i][1]][0][j].reshape(1,self.MAX_SEQUENCE_LENGTH))
                val = (np.argmax(pred,axis=1))[0]
                model_OP.append(str(infer_labels[i-1][val]))
            Predicted[self.labels[i][1]] = model_OP
        
        Final_Inference = pd.DataFrame(tokenizer.sequences_to_texts((DataGen[self.labels[i][1]][0])))
        
        for i in range(1,len(infer_labels)+1):
            YY = infer_labels[i-1]
            Test = []
            for j in range(len(DataGen[self.labels[i][1]][0])):
                Test.append((YY[(np.argmax(DataGen[self.labels[i][1]][1][j]))]))
            col_a = self.labels[i][1] + ' (actual)'
            col_p = self.labels[i][1] + ' (predicted)'
            Final_Inference[col_a] = Test
            Final_Inference[col_p] = Predicted[self.labels[i][1]]
        return Final_Inference