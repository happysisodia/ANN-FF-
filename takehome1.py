# -*- coding: utf-8 -*-
# takehome1.py
# written by Happy Sisodia 
# Clemson University ID: hsisodi
# For the purpose of ANN Take-Home #1
# ECE8720 taught by Robert J Schalkoff

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import random

data_AEP = np.loadtxt('db4_Diff1f_100_128w_AEP2_gp2_ver1-1_ece856.txt')
data_NAEP = np.loadtxt('db4_Diff1f_100_128w_nonAEP2_gp2_ver1-1_ece856.txt')

class NeuralNetwork():
    
    random.seed(1000)
    def __init__(self):
        self.input_node    = 27
        self.Output_node   = 1
        self.hidden_node   = 55
        self.learning_rate = 0.325
        self.Momentum      = 0.25        
        
        self.W1 = np.random.normal(0.0, pow(self.input_node, -0.5),(self.input_node,self.hidden_node))
        self.W2 = np.random.normal(0.0, pow(self.hidden_node, -0.5), (self.hidden_node, self.Output_node))
        
        self.b1 = np.ones((1,self.hidden_node)) - 0.5
        self.b2 = np.ones((1,self.Output_node)) - 0.5
    
    def smote(self, sample, N, k):
        self.sample = sample
        self.k = k
        self.T = len(self.sample)
        self.N = N
        self.newIndex = 0
        self.synthetic = []
        self.neighbors = NearestNeighbors(n_neighbors=self.k).fit(self.sample)
        
        if self.N < 100:
            self.T = (self.N / 100) * self.T
            self.N = 100
        self.N = int(self.N / 100)

        for i in range(0, self.T):
            nn_array = self.compute_k_nearest(i)
            self.populate(self.N, i, nn_array)        
        
    def compute_k_nearest(self, i):
        nn_array = self.neighbors.kneighbors([self.sample[i]], self.k, return_distance=False)
        if len(nn_array) == 1:
            return nn_array[0]
        else:
            return []   
    
    def populate(self, N, i, nn_array):
        while N != 0:
            nn = random.randint(0, self.k - 1)
            self.synthetic.append([])
            for attr in range(0, len(self.sample[i])):
                dif = self.sample[nn_array[nn]][attr] - self.sample[i][attr]
                gap = random.random()
                self.synthetic[self.newIndex].append(self.sample[i][attr] + gap * dif)
            self.newIndex += 1
            N -= 1    
                
    def sse_loss(self,y_true, y_pred):
        TS = np.sum((y_true - y_pred) **2)
        return TS
    
    def sigmoid(self,s):
        return 1/(1 + np.exp(-s))
    
    def sigmoid_derivative(self,s):
        return self.sigmoid(s) * (1 - self.sigmoid(s))

    def tanh_derivative(self, z):
        return (1 - np.power(np.tanh(z), 2))     
    
    def forward(self,X):
        self.z1 = np.dot(X,self.W1) + self.b1
        self.o1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.o1,self.W2)  + self.b2
        o2 = self.sigmoid(self.z2)     
        return o2
    
    def backpropagation(self,X,y,o):
        self.output_delta_old = 0
        self.output2_delta_old = 0
        self.output_error = y - o
        self.output_delta = self.output_error * self.sigmoid_derivative(o)
        self.output1_error = self.output_delta.dot(self.W2.T)
        self.output2_delta = self.output1_error * self.sigmoid_derivative(self.o1)
        
        self.W1 += self.Momentum * self.output_delta_old + self.learning_rate * X.T.dot(self.output2_delta)
        self.W2 += self.Momentum * self.output2_delta_old + self.learning_rate * self.o1.T.dot(self.output_delta)
        
        self.output_delta_old  = X.T.dot(self.output2_delta)
        self.output2_delta_old = self.o1.T.dot(self.output_delta)                
        
    def train(self,X,y):
        o = self.forward(X)
        self.backpropagation(X,y,o)
    
    def evaluate(self,X,y):
        pred = self.forward(X)
        loss_error = self.sse_loss(pred,y)
        acc = np.equal(np.argmax(pred, axis=1), np.argmax(y, axis=1)).mean()
        return loss_error,acc
    
    def normalize(self,x):
         return x/np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    
    def counfusion_matrix(self,X,y):
        y_pred = self.forward(X)
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        
        for i in range(0,y_pred.shape[0]):
            for j in range(0,y_pred.shape[1]):
                if y_pred[i][j] <= 0.5:
                   y_pred[i][j] = 0.01
                elif y_pred[i][j] > 0.5:
                   y_pred[i][j] = 0.99
                j += 1
            i += 1
              
        for i in range(0,y_pred.shape[0]):
            for j in range(0,y_pred.shape[1]):  
                if y[i][j] == y_pred[i][j]:
                    if y_pred[i][j] == 0.99:
                        TP += 1
                    elif y_pred[i][j] == 0.01:
                        TN += 1
                elif y[i][j] != y_pred[i][j]:
                    if y_pred[i][j] == 0.99:
                        FP += 1
                    elif y_pred[i][j] == 0.01:
                        FN += 1
                
        return TP,TN,FP,FN    
    
if __name__ == '__main__':
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    Acccuracy = 0
    NN = NeuralNetwork()
    s = NN.smote(sample=data_AEP, N=3000, k=5)
    aep = np.array(NN.synthetic)
    aep_train,aep_test = aep[:2000,:],aep[2001:,:]
    naep_train,naep_test = data_NAEP[:2000,:],data_NAEP[2001:,:]
    
    x_train = np.concatenate((naep_train,aep_train))
    x_test  = np.concatenate((aep_test,naep_test))
    
    x_train =  NN.normalize(x_train)
    x_test  =  NN.normalize(x_test)
        
    output_aep = np.zeros((1,2000)) + 0.99
    y1 = output_aep.reshape((2000,1))

    output_naep = np.zeros((1,2000)) + 0.01
    y2 = output_naep.reshape((2000,1))
    y_output = np.concatenate((y2,y1))

    output_taep = np.zeros((1,485)) + 0.99
    yt1 = output_taep.reshape((485,1))

    output_tnaep = np.zeros((1,485)) + 0.01
    yt2 = output_tnaep.reshape((485,1))
    yt_output = np.concatenate((yt1,yt2))
    
    BATCH_SIZE = 100
    
    loss_train = []
    acc_train  = []
    loss_tr    = 0 
    EPOCH = 1200
    for epochs in range(EPOCH):
        
        index = np.random.permutation(x_train.shape[0])
        batch_count = 0
        
        while batch_count* BATCH_SIZE < x_train.shape[0]:
            batch_X , batch_Y = x_train[batch_count*BATCH_SIZE:(batch_count+1)*BATCH_SIZE], y_output[batch_count*BATCH_SIZE:(batch_count+1)*BATCH_SIZE]
            batch_count += 1
            NN.train(batch_X,batch_Y)
            
        loss_tr ,accn = NN.evaluate(x_train,y_output)
        loss_tst,acct = NN.evaluate(x_test,yt_output)
        acc_train.append(accn)
        
        if epochs % 100 == 0:
            loss_train.append(loss_tr)
            print("Epoch Number: %d" % epochs)
            print("loss for the epoch %.3f\n" % loss_tr)

TP , TN , FP , FN = NN.counfusion_matrix(x_test,yt_output)

Acccuracy     = ((TP + TN) / (TP + TN + FP + FN)) * 100
sensistivity  = TP / (TP + FN)
specificity   = TN / (TN + FP)
precision     = TP / (TP + FP)
F1            = (2* precision * sensistivity)/ (precision + sensistivity)

print("True Positive: %d"  % TP)
print("True Negative: %d"  % TN)
print("False Positive: %d" % FP)
print("False Negative: %d" % FN)
print("Acccuracy: %.3f"      % Acccuracy  )
print("sensistivity: %.3f"   % sensistivity)
print("specificity: %.3f"    % specificity)
print("precision: %.3f"      % precision)
print("F1: %.3f"             % F1)
                      
plt.subplots()
plt.plot(loss_train, label='Train')
plt.legend()
plt.xlabel('Epochs')
plt.title('Loss')
plt.show()