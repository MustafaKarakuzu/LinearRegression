import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt


class Quadratic_Reg:

    def __init__(self, data):
        '''
        inputs
            data : your data
            x: independent variables
            y: dependent variable
        '''
        self.data = data
        self.x = np.array(self.data.iloc[:, 1:])
        self.y = np.array(self.data.iloc[:, 0:1])
        self.y_mean=np.mean(self.y)
        self.x_mean=np.mean(self.x)
        self.yx = self.x*self.y
        self.Xsq = self.x**2
        self.yXsq = self.y*self.x**2
        self.Xqb = self.x**3
        self.Xqua = self.x**4
        self.n=len(self.x)
        self.m=3
        self.Xcons = np.column_stack((np.ones(len(self.x)), self.x, self.Xsq))
        self.x_trans = self.Xcons.transpose() #transpoze of X 
        self.Xx=self.x_trans.dot(self.Xcons) # X'X
        self.x_inv = np.linalg.inv(self.Xx)

    def show(self):
        a=np.column_stack((self.x,self.y,self.yx,self.Xsq,self.yXsq,self.Xqb,self.Xqua))
        table=tabulate(a, headers=['Y','X','Y*X','X^2','Y*X^2','X^3','X^4'])
        print(table,'\n')
    
    def coef(self):
        sub=np.array([
            [self.n,np.sum(self.x),np.sum(self.Xsq)],
            [np.sum(self.x),np.sum(self.Xsq),np.sum(self.Xqb)],
            [np.sum(self.Xsq),np.sum(self.Xqb),np.sum(self.Xqua)]
                     ])
        B0=np.array([
            [np.sum(self.y),np.sum(self.x),np.sum(self.Xsq)],
            [np.sum(self.yx),np.sum(self.Xsq),np.sum(self.Xqb)],
            [np.sum(self.yXsq),np.sum(self.Xqb),np.sum(self.Xqua)]
                     ])
        B1=np.array([
            [self.n,np.sum(self.y),np.sum(self.Xsq)],
            [np.sum(self.x),np.sum(self.yx),np.sum(self.Xqb)],
            [np.sum(self.Xsq),np.sum(self.yXsq),np.sum(self.Xqua)]
                     ])
        B2=np.array([
            [self.n,np.sum(self.x),np.sum(self.y)],
            [np.sum(self.x),np.sum(self.Xsq),np.sum(self.yx)],
            [np.sum(self.Xsq),np.sum(self.Xqb),np.sum(self.yXsq)]
                     ])
        self.b0=np.linalg.det(B0)/np.linalg.det(sub)
        self.b1=np.linalg.det(B1)/np.linalg.det(sub)
        self.b2=np.linalg.det(B2)/np.linalg.det(sub)
        
        
    def quadratic_reg_model(self,a):
        return self.b0+self.b1*a+self.b2*a**2

    def quadratic_reg(self):
        self.predict=self.quadratic_reg_model(self.x)
        a=np.column_stack((self.y,self.x,self.predict))
        print(tabulate(a, headers=['Y', 'X', 'Predicted Y']),'\n')
        
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
        print(Vb[0,0])
        print(Vb[1,1])
        print(Vb[2,2])
        
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
       ['constant',self.b0, self.b0/np.sqrt(Vb[0,0]), Vb[0,0], np.sqrt(Vb[0,0])],
       ['x1',
        self.b1,
        self.b1/np.sqrt(Vb[1,1]),
        Vb[1,1],
        np.sqrt(Vb[1,1])
        ],
       ['x^2',
       self.b2,
       self.b2/np.sqrt(Vb[2,2]),
       Vb[2,2],
       np.sqrt(Vb[2,2])]
       ]
       
        table1=tabulate(dataCof, headers1, tablefmt='pipe')
        print(table1)
    
    def qua_plot(self):
        fig, ax=plt.subplots()
        plt.plot(self.x, self.quadratic_reg_model(self.x), color='black')
        plt.scatter(self.x,self.y)
        plt.show()
