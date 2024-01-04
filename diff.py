# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:14:29 2022

@author: chenyon
"""
import numpy as np
import compecon as ce

np.set_printoptions(precision=15)

def f(x):
    x1, x2 = x
    y = [np.exp(x1)- x2, 
         x1+x2**2,
         (1-x1)*np.log(x2)]
    return np.array(y)

ce.jacobian(f,[0, 1])


def f(x):
    x1,x2=x
    return x1**2*np.exp(-x2)
ce.jacobian(f,[1, 0])
ce.hessian(f,[1,0])


# need to install following: https://numdifftools.readthedocs.io/en/latest/tutorials/install.html
# In prompt command window, enter:
#    pip install numdifftools 

import numdifftools as nd
import matplotlib.pylot as plt

f = nd.Derivative(np.exp, full_output = True)
val, info = f(0)
print(val)

df = nd.Derivative(np.sin, n=2)
print(df(0))

nd.Gradient(f)([1,0])
nd.Hessian(f)([1,0])

x = np.linspace(-2, 2, 100)
for i in range(3):
    df = nd.Derivative(np.tanh, n=i)
    y = df(x)
    h = plt.plot(x, y/np.abs(y).max())
plt.show()