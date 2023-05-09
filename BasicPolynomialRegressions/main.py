# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 18:15:45 2023

@author: MUSTAFA KARAKUZU
"""
#Imported Library
import numpy as np
import pandas as pd

# Created Library
from basic_reg import Basic_Regression as bsreg
from quadratic_reg import Quadratic_Reg as quareg
from qubic_reg import Qubic_Reg as qbreg
from exponantial_reg import Exponantial_Reg as expreg


# BASIC REGRESSION

data0=bsreg(data=np.genfromtxt('data\data_basicreg.txt', delimiter='')) 
data0.basic_regression() #----->  to show result of regression
data0.bsc_plot() # -----> to show plot of basic regression


# QUADRATIC REGRESSION

data1=quareg(data=pd.read_excel('data\data_quadratic_reg.xlsx'))
data1.show() #-----> to show core values of variables
data1.coef() #------> to calculate the predicters with Cramer method
data1.quadratic_reg() #------> to show result of regression
data1.qua_plot() #------> to show plot of quadratic regression


# CUBIC REGRESSION

data2=qbreg(data=pd.read_excel('data\data_qubic_reg.xlsx'))
data2.show() #-----> to show core values of variables
data2.qubic_regression() #------> to show result of regression
data2.qbc_plot() #------> to show plot of qubic regression


# EXPONANTIAL REGRESSION

data3=expreg(data=pd.read_excel('data\data_exponantial_reg.xlsx'))
data3.show()  #-----> to show core values of variables
data3.exponantial_regression() #------> to show result of regression
data3.exp_plot() #------> to show plot of exponantial regression
