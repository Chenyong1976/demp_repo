# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 15:23:11 2022

@author: chenyon
"""
# https://pythonnumericalmethods.berkeley.edu/notebooks/chapter22.06-Python-ODE-Solvers.html
# solve: S'(t)=cos(t), s0=0 for t \in [0, 2pi]
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

plt.style.use('seaborn-poster')

#matplotlib inline

F = lambda t, s: np.cos(t)

t_eval = np.arange(0, np.pi, 0.1)
sol = solve_ivp(F, [0, np.pi], [0], t_eval=t_eval)

plt.figure(figsize =(12,4))
plt.subplot(121)
plt.plot(sol.t,sol.y[0])
plt.xlabel('t')
plt.ylabel('S(t)')
plt.subplot(122)
plt.plot(sol.t, sol.y[0]-np.sin(sol.t))
plt.xlabel('t')
plt.ylabel('S(t)-sin(t)')
plt.tight_layout()
plt.show()

# change absolute and relative tolerance
sol = solve_ivp(F, [0, np.pi], [0], t_eval=t_eval, rtol = 1e-8, atol=1e-8)
# chagne method used
sol = solve_ivp(F, [0, np.pi], [0], method='RK45', t_eval=t_eval, rtol = 1e-8, atol=1e-8)


plt.figure(figsize =(12,4))
plt.subplot(121)
plt.plot(sol.t,sol.y[0])
plt.xlabel('t')
plt.ylabel('S(t)')
plt.subplot(122)
plt.plot(sol.t, sol.y[0]-np.sin(sol.t))
plt.xlabel('t')
plt.ylabel('S(t)-sin(t)')
plt.tight_layout()
plt.show()

# solve S'(t)=-S(t), S0=1 for t\in [0,1]
F = lambda t, s: -s
t_eval = np.arange(0, 1.01, 0.01)
sol = solve_ivp(F, [0,1], [1], t_eval=t_eval)

plt.figure(figsize =(12,4))
plt.subplot(121)
plt.plot(sol.t,sol.y[0])
plt.xlabel('t')
plt.ylabel('S(t)')
plt.subplot(122)
plt.plot(sol.t, sol.y[0]-np.exp(-sol.t))
plt.xlabel('t')
plt.ylabel('S(t)-exp(-t)')
plt.tight_layout()
plt.show()

# 2d ODE
# S'(t)= [[0, t**2], [-t, 0]] @ S(t)
F = lambda t, s: np.dot(np.array([[0, t**2], [-t, 0]]),s)

t_eval = np.arange(0, 10.01, 0.01)
sol = solve_ivp(F, [0, 10], [1, 1], t_eval = t_eval)

plt.figure(figsize =(12,8))
plt.plot(sol.y.T[:,0], sol.y.T[:,1]) # t is column vector, sol.y is row vector so transpose is necessary.
plt.xlabel('x')
plt.ylabel('y')
plt.show()