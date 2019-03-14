import numpy as np
from scipy.optimize import minimize

class LinearRegression():

    def __init__(self,reg=None,lmbda=0.1,itration=1000):
        self.reg = reg
        self.lmbda = lmbda
        self.itration = itration
        if self.reg not in ['l1','l2',None] :
            raise Exception("Regularization should be l1 , l2 or None not ",self.reg)
        print("reg : ",self.reg,"    lmbda : ",self.lmbda,"    itration : ",self.itration)

        
    def h(self,x,theta):
        return np.dot(x,theta)


    def regularization(self,theta,m):
        if self.reg == 'l2':
            r = (self.lmbda/(2 * m)) * np.sum(np.square(theta))
        if self.reg == 'l1':
            r = (self.lmbda/(2 * m)) * np.sum(theta)
        if self.reg == None :
            r = 0
        return r


    def cost_function(self,theta,x,y):
        m = x.shape[0]
        loss = self.h(x,theta) - y
        error = (1/(2 * m)) * np.sum(np.square(loss))
        r = self.regularization(theta,m)
        reg_cost = error + r
        return reg_cost


    def gradient(self,theta,x,y):
        m = x.shape[0]
        loss = self.h(x,theta)
        loss = loss.reshape([m,1])
        loss = loss - y
        grad = (1/m)*np.dot(x.T,loss)
        r = self.regularization(theta,m)
        grad += r
        return grad


    def predict(self,x,theta):
        m,n = x.shape
        ones = np.ones([m,1],dtype=np.int32)
        x = np.concatenate((ones,x),axis=1)
        y_pred = np.dot(x,theta)
        return y_pred


    def fit(self,x,y) :
        m,n = x.shape
        ones = np.ones([m,1])
        x = np.concatenate((ones,x),axis=1)
        y = y.reshape([m,1])
        theta = np.random.randn(n+1,1)
        result = minimize(self.cost_function, theta, args = (x, y),method = 'TNC' ,
                          jac = self.gradient,options={'maxiter':self.itration})
        return result.x
