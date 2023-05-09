#!/usr/bin/env python
# coding: utf-8



import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt




class Basic_Regression:
    
    def __init__(self, data):
        '''
        inputs
            data : your data
            x: independent variables
            y: dependent variable
        '''
        self.data=data
        self.x=data[:,1:]
        self.y=data[:,0:1]
        self.x_mean=np.mean(self.x)
        self.y_mean=np.mean(self.y)
        self.X = np.column_stack((np.ones(len(self.x)), self.x)) #adding constant columns
        self.m=self.X.shape[1] # degree of freedom
        self.n=len(self.y) # number of observation
        self.x_trans = self.X.transpose() #transpoze of X 
        self.Xx=self.x_trans.dot(self.X) # X'X
        self.x_inv = np.linalg.inv(self.Xx) # invert of X
        self.xy=self.x_trans.dot(self.y) # multipy X'y
        self.b=self.x_inv.dot(self.xy) # X(-1)*X'y ---> coefficent matrix
        self.n=len(self.x)
        self.m=self.X.shape[1]

    def show(self):
        print(tabulate((self.data), headers=['y','x']))
        print(f'\n Mean of Y: {self.y_mean} ----- Mean of X: {self.x_mean}')
        print(f'\n {"X Matrix:".center(10)} \n {self.x}')
        print(f'\n {"X Matrix With Constant:".center(10)} \n {self.X}')
        print(f'\n {"Y Matrix:".center(10)} \n {self.y}')
        print(f'\n {"X Transpose:".center(10)} \n {self.x_trans}')
        print("\nX'X Matrix")
        print(np.asmatrix(self.Xx))
        print("\nInvert of X")
        print(np.asmatrix(self.x_inv))
        print("\n X'y Matrix")
        print(np.asmatrix(self.xy))
        print("\nCoefficent Matrix")
        print(np.asmatrix(self.b))
    

    def basic_reg_model(self,a):
            return self.b[0, 0] + a * self.b[1, 0]
                
        
    def basic_regression(self): 
        self.predict=self.basic_reg_model(self.x)
        a=np.column_stack((self.y,self.x))
        c=np.column_stack((a,self.predict))
        print('Values')
        print(tabulate(c, headers=['Y', 'X', 'Predicted Y']),'\n')
        
        y_dif_y_mean=[]
        for i in self.y:
            y_dif=(i-self.y_mean)**2
            y_dif_y_mean.append(y_dif)
        y_dif_y_mean=np.array(y_dif_y_mean)
        TKT=np.sum(y_dif_y_mean)

        y_diff_y_mean=[]
        for i in self.predict:
            y_diff=(i-self.y_mean)**2
            y_diff_y_mean.append(y_diff)
        y_diff_y_mean=np.array(y_diff_y_mean)
        RKT=np.sum(y_diff_y_mean)
 
        Vb=((TKT-RKT)/(self.n-self.m))*self.x_inv
        print('Covarince Matris')
        print(Vb,'\n')

#Showing Regression Performance
        headers2=['R', 'R Square', 'Adjusted R Square']
        data2=[
            [np.sqrt((RKT/TKT)),
             RKT/TKT,
             1-((1-(RKT/TKT))*(self.n-1)/(self.n-self.m-1))]
            ]
        table2=tabulate(data2, headers2,tablefmt='pipe' )
        print(f'{table2}\n')

#ANOVA        
        headers = ['Change', 'Sum Square', 'Degree of Freedom','Mean Square','F']
        dataAn=[
       ['Regresyon', RKT, self.m-1, RKT/(self.m-1), (RKT/(self.m-1))/((TKT-RKT)/(self.n-self.m)) ],
       ['Residual', (TKT-RKT), self.n-self.m, (TKT-RKT)/(self.n-self.m)],
       ['Total', TKT, self.n-1]
       ]
        table = tabulate(dataAn, headers, tablefmt='pipe')
        print('ANOVA')
        print(table,'\n')


#Showing Coefficient Statistics
        x_dif_x_mean=[]
        for i in self.x:
            x_dif=(i-self.x_mean)**2
            x_dif_x_mean.append(x_dif)
        x_dif_x_mean=np.array(x_dif_x_mean)
        sumX=np.sum(x_dif_x_mean)
        
        headers1=['variables', 'B values','t values', 'variance', 'standart errors']
        dataCof=[
       ['constant',self.b[0,0], self.b[0,0]/np.sqrt(Vb[0,0]), Vb[0,0], np.sqrt(Vb[0,0])],
       ['x1',
        self.b[1,0],
        self.b[1,0]/np.sqrt(((np.sum(y_dif_y_mean) - np.sum(y_diff_y_mean))/(self.n-self.m))/sumX),
        ((np.sum(y_dif_y_mean) - np.sum(y_diff_y_mean))/(self.n-self.m))/sumX,
        ((np.sqrt(TKT-RKT)/(self.n-self.m))/sumX)]]
        table1=tabulate(dataCof, headers1, tablefmt='pipe')
        print(table1)
     
    def bsc_plot(self):
        fig, ax=plt.subplots()
        plt.plot(self.x, self.basic_reg_model(self.x), color='black')
        plt.scatter(self.x,self.y)
        plt.show()
     
            
        
        
        
        
        
        
        
      