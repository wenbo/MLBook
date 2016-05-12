import numpy 
from scipy import sparse as S
from matplotlib import pyplot as plt 
from scipy.sparse.csr import csr_matrix 
import pandas 

def normalize(x):
    V = x.copy()
    V -= x.min(axis=1).reshape(x.shape[0],1)
    V /= V.max(axis=1).reshape(x.shape[0],1)
    return V
    
def sigmoid(x):
    #return x*(x > 0)
    #return numpy.tanh(x)
    return 1.0/(1+numpy.exp(-x)) 

class RBM():
    def __init__(self, n_visible=None, n_hidden=None, W=None, learning_rate = 0.1, weight_decay=1,cd_steps=1,momentum=0.5):
        if W == None:
            self.W =  numpy.random.uniform(-.1,0.1,(n_visible,  n_hidden)) / numpy.sqrt(n_visible + n_hidden)
            self.W = numpy.insert(self.W, 0, 0, axis = 1)
            self.W = numpy.insert(self.W, 0, 0, axis = 0)
        else:
            self.W=W 
        self.learning_rate = learning_rate 
        self.momentum = momentum
        self.last_change = 0
        self.last_update = 0
        self.cd_steps = cd_steps
        self.epoch = 0 
        self.weight_decay = weight_decay  
        self.Errors = []
         
            
    def fit(self, Input, max_epochs = 1, batch_size=100):  
        if isinstance(Input, S.csr_matrix):
            bias = S.csr_matrix(numpy.ones((Input.shape[0], 1))) 
            csr = S.hstack([bias, Input]).tocsr()
        else:
            csr = numpy.insert(Input, 0, 1, 1)
        for epoch in range(max_epochs): 
            idx = numpy.arange(csr.shape[0])
            numpy.random.shuffle(idx)
            idx = idx[:batch_size]  
                   
            self.V_state = csr[idx] 
            self.H_state = self.activate(self.V_state)
            pos_associations = self.V_state.T.dot(self.H_state) 
  
            for i in range(self.cd_steps):
              self.V_state = self.sample(self.H_state)  
              self.H_state = self.activate(self.V_state)
              
            neg_associations = self.V_state.T.dot(self.H_state) 
            self.V_state = self.sample(self.H_state) 
            
            # Update weights. 
            w_update = self.learning_rate * ((pos_associations - neg_associations) / batch_size) 
            total_change = numpy.sum(numpy.abs(w_update)) 
            self.W += self.momentum * self.last_change  + w_update
            self.W *= self.weight_decay 
            
            self.last_change = w_update
            
            RMSE = numpy.mean((csr[idx] - self.V_state)**2)**0.5
            self.Errors.append(RMSE)
            self.epoch += 1
            # print("Epoch %s: RMSE = %s; ||W||: %6.1f; Sum Update: %f" % (self.epoch, RMSE, numpy.sum(numpy.abs(self.W)), total_change))  
        return self 
        
    def learning_curve(self):
        plt.ion()
        #plt.figure()
        E = numpy.array(self.Errors)
        plt.plot(pandas.rolling_mean(E, 50)[50:])  
        plt.show()
     
    def activate(self, X):
        if X.shape[1] != self.W.shape[0]:
            if isinstance(X, S.csr_matrix):
                bias = S.csr_matrix(numpy.ones((X.shape[0], 1))) 
                csr = S.hstack([bias, X]).tocsr()
            else:
                csr = numpy.insert(X, 0, 1, 1) 
        else:
            csr = X
        p = sigmoid(csr.dot(self.W)) 
        p[:,0]  = 1.0 
        return p  
        
    def sample(self, H, addBias=True): 
        if H.shape[1] == self.W.shape[0]:
            if isinstance(H, S.csr_matrix):
                bias = S.csr_matrix(numpy.ones((H.shape[0], 1))) 
                csr = S.hstack([bias, H]).tocsr()
            else:
                csr = numpy.insert(H, 0, 1, 1)
        else:
            csr = H
        p = sigmoid(csr.dot(self.W.T)) 
        p[:,0] = 1
        return p
      
if __name__=="__main__":
    data = numpy.random.uniform(0,1,(100,10))
    rbm = RBM(10,15)
    rbm.fit(data,1000)
    rbm.learning_curve()