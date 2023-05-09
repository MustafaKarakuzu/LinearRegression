# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 01:56:18 2023

@author: MUSTAFA KARAKUZU
"""

#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import math

class Exponantial_Reg:
    
    def __init__(self, data):
        '''
        inputs
            data : your data
            x: independent variables
            y: dependent variable
        '''
        self.data=data
        self.x=np.array(data.iloc[:,1:])
        self.y=np.array(data.iloc[:,0:1])
        self.ylog=np.log(self.y)
        self.ylog_mean=np.mean(self.ylog)
        self.c=np.ones(len(self.x))
        self.x_mean=np.mean(self.x)
        self.y_mean=np.mean(self.y)
        self.X = np.column_stack((self.c,self.x)) #adding constant columns
        self.m=self.X.shape[1] # degree of freedom
        self.n=len(self.y) # number of observation
        self.x_trans = self.X.transpose() #transpoze of X 
        self.Xx=self.x_trans.dot(self.X) # X'X
        self.x_inv = np.linalg.inv(self.Xx) # invert of X
        self.xy=self.x_trans.dot(self.ylog) # multipy X'y
        self.b=self.x_inv.dot(self.xy) # X(-1)*X'y ---> coefficent matrix
        self.n=len(self.x)
        self.m=self.X.shape[1]
        self.const=math.exp(self.b[0,0])
        self.beta=math.exp(self.b[1,0])      

    def show(self):
        print(tabulate((self.data), headers=['y','x']))
        print(f'\n Mean of Y: {self.y_mean} ----- Mean of X: {self.x_mean}')
        print(f'\n {"X Matrix:".center(10)} \n {self.X}')
        print(f'\n {"Y Matrix:".center(10)} \n {self.y}')
        print(f'\n {"X Transpose:".center(10)} \n {self.x_trans}')
        print("\nInvert of X")
        print(np.asmatrix(self.x_inv))
        print("\n X'y Matrix")
        print(np.asmatrix(self.xy))
        print("\nCoefficent Matrix")
        print(self.b)


    def exponantial_reg_model(self,a):
        return self.const*(self.beta**a)
     
    def exponantial_regression(self): 
        self.predict=self.exponantial_reg_model(self.x)
        self.predict=np.array(self.predict)
        a=np.column_stack((self.y,self.x,self.predict))
        print('Values')
        print(tabulate(a, headers=['Y', 'X', 'Predicted Y']),'\n')
     
        y_dif_y_mean=[]
        for i in self.ylog:
            y_dif=(i-self.ylog_mean)**2
            y_dif_y_mean.append(y_dif)
        y_dif_y_mean=np.array(y_dif_y_mean)
        TKT=np.sum(y_dif_y_mean)

        y_diff_y_mean=[]
        for i in self.predict:
            y_diff=(np.log(i)-self.ylog_mean)**2
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
       ['constant',self.const, self.const/np.sqrt(Vb[0,0]), Vb[0,0], np.sqrt(Vb[0,0])],
       ['x1', self.b[1,0], self.b[1,0]/np.sqrt(Vb[1,1]), Vb[1,1],np.sqrt(Vb[1,1])],
       ]
        table1=tabulate(dataCof, headers1, tablefmt='pipe')
        print(table1)
        
    def exp_plot(self):
        fig, ax=plt.subplots()
        plt.plot(self.x, self.exponantial_reg_model(self.x), color='black')
        plt.scatter(self.x,self.y)
        plt.show()
         
       
        
        
        
        
        
        
      