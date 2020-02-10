"""
Created on Mon Feb  3 18:20:15 2020

@author: Yohan Reyes
"""

# =============================================================================
# %% PATHS
# =============================================================================

PATH_DATA = '../Coronavirus Analysis/'


# =============================================================================
# %% Libs
# =============================================================================

import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import datetime
import pandas as pd
import numpy as np
import sklearn 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as st
from scipy.optimize import curve_fit
from scipy.optimize import basinhopping

import scipy as sp

import os

# =============================================================================
# %% Defs
# =============================================================================

def expFunc(x, a, b, c):
    return a * np.exp(b * x) + c

def polyFunc(x, a, b, d, c):
    return a * np.power(x,b) + x*d + c

def stretchedExpFunc(x, a, m, s, l, c):
    return a*np.exp(-m* np.power((x/s),m)*(1/l)) + c
#    return a*np.exp(-m*((x/s)**m))

def E_func(params):
    ret = stretchedExpFunc(X_hat, params[0], params[1], params[2], params[3], params[4])

    n = float(len(ret))
    er = np.sum(np.power((ret - y_),2))/ n

    if np.isnan(er):
        er = 1e10

    return er


def parabolicFractalFunc(x,a,b,c,d):
#    return a*np.power(x,-b)*np.exp(-d*np.power(np.log(x),2)) + c
    return np.multiply(a*np.power(x,-b),np.exp(-d*np.power(np.log(x),2)) + c)

# =============================================================================
# %% Load Data
# =============================================================================


data = pd.read_csv(PATH_DATA+'Data.csv')
data.columns = ['Date','Infected','Percent Change']
data.index = list(data['Date'].values)


# =============================================================================
# %% Plot Data and log data
# =============================================================================

plt.figure()
#plt.plot(data['Date'], data['Infected'])
plt.plot(data['Infected'].values)

#plt.figure()
#plt.plot(np.log(data['Infected']))

X = list(range(1,len(data)+1))
X_ = np.asmatrix(X)
y = data['Infected'].values
y_hat = []

# =============================================================================
# %% Fit every one of the functions
# =============================================================================

#X_hat = list(range(1,(len(data)+10)))

delta = 180

dates = []
# temp = datetime.datetime.now()
# str(temp.date())
temp_ = datetime.datetime(2020, 2, 6)
temp = datetime.datetime(2020, 2, 6)

dates.append(str(temp.date()))
for i in range(1,delta+1):
    temp = temp_ + datetime.timedelta(days=i)
    dates.append(str(temp.date()))
dates = pd.DataFrame(dates)
dates.columns = ['Date']
dates = pd.concat((data['Date'],dates['Date']),axis = 0)
dates = dates.reset_index(drop = True)


X_hat = list(range(1,(len(data)+1)))
X_hat = np.asmatrix(X_hat)

X_hat_ = list(range(1,(len(data)+delta+2)))
X_hat_ = np.asmatrix(X_hat_)

y_hat = []
y_hat_all = []

opt = []
e = []
mse = []


popt, pcov = curve_fit(expFunc, X, y)
opt.append(popt)
plt.figure(figsize = (20,10))
plt.subplot(3, 1, 1)
plt.plot(data['Infected'].values)
#plt.title('Ajuste de Curvas con una funcion exponencial: ' + str(popt[0]) + 'exp(' + str(popt[1]) + ') ' +  str(popt[2]) )
plt.title('Ajuste de Curvas con una funcion exponencial: A*exp(b*X) + C'  )
y_hat_ = pd.DataFrame(np.transpose(expFunc(X_hat,popt[0],popt[1],popt[2])), columns = ['Pred'])
y_hat__ = pd.DataFrame(np.transpose(expFunc(X_hat_,popt[0],popt[1],popt[2])), columns = ['Pred'], index = dates)
y_hat.append(y_hat_)
y_hat_all.append(y_hat__)
plt.plot(y_hat__.values)
e_ = y_hat_.values-np.reshape(data['Infected'].values,[-1,1])
e.append(e_)
# plt.plot(e_)
mse_ = np.sum(np.power(e_,2))/len(e_)
mse.append(mse_)


plt.subplot(3, 1, 2)
plt.plot(data['Infected'].values)
plt.title('Ajuste de Curvas con una funcion polinomial: A*X^b + C'  )
popt, pcov = curve_fit(polyFunc, X, y)
opt.append(popt)
y_hat_ = pd.DataFrame(np.transpose(polyFunc(X_hat,popt[0],popt[1],popt[2],popt[3])), columns = ['Pred'])
y_hat__ = pd.DataFrame(np.transpose(polyFunc(X_hat_,popt[0],popt[1],popt[2],popt[3])), columns = ['Pred'], index = dates)
y_hat.append(y_hat_)
y_hat_all.append(y_hat__)
plt.plot(y_hat__.values)
e_ = y_hat_.values-np.reshape(data['Infected'].values,[-1,1])
e.append(e_)
#plt.plot(e_)
mse_ = np.sum(np.power(e_,2))/len(e_)
mse.append(mse_)


p0 = [4.70438908e+06, -5.18777553e-01,  2.52895759e+01, -1.13516528e-01, 1.80811964e+02]


plt.subplot(3, 1, 3)
plt.plot(data['Infected'].values)
# popt, pcov = curve_fit(stretchedExpFunc, X, y)
#ret = basinhopping(stretchedExpFunc, p0, take_step=mystep, accept_test=mybound)
y_= data['Infected'].values
ret = basinhopping(E_func, p0)

popt = []
popt.append(ret['x'][0])
popt.append(ret['x'][1])
popt.append(ret['x'][2])
popt.append(ret['x'][3])
popt.append(ret['x'][4])

opt.append(popt)
y_hat_ = pd.DataFrame(np.transpose(stretchedExpFunc(X_hat,popt[0],popt[1],popt[2],popt[3],popt[4])), columns = ['Pred'])
y_hat__ = pd.DataFrame(np.transpose(stretchedExpFunc(X_hat_,popt[0],popt[1],popt[2],popt[3],popt[4])), columns = ['Pred'], index = dates)
y_hat.append(y_hat_)
y_hat_all.append(y_hat__)
plt.plot(y_hat__.values)
e_ = y_hat_.values-np.reshape(data['Infected'].values,[-1,1])
e.append(e_)
#plt.plot(e_)
mse_ = np.sum(np.power(e_,2))/len(e_)
mse.append(mse_)

rmse = []

rmse = np.sqrt(mse)


# =============================================================================
# %% Definitive Plot
# =============================================================================


plt.figure()

x = pd.DatetimeIndex(dates.values)
data_ = pd.DataFrame(([np.NaN]*(delta+1)),columns = ['Infected'])
data_ = pd.concat((data['Infected'],data_['Infected']),axis = 0)
data_ = data_.reset_index(drop = True)
data_.index = x

# plt.plot(data_, linestyle='-.', color = 'crimson', linewidth = 3)
plt.plot(data_, color = 'crimson', linewidth = 5)


popt, pcov = curve_fit(stretchedExpFunc, X, y)
opt.append(popt)
y_hat_ = pd.DataFrame(np.transpose(stretchedExpFunc(X_hat,ret['x'][0],ret['x'][1],ret['x'][2],ret['x'][3],ret['x'][4])), columns = ['Pred'])
y_hat__ = pd.DataFrame(np.transpose(stretchedExpFunc(X_hat_,ret['x'][0],ret['x'][1],ret['x'][2],ret['x'][3],ret['x'][4])), columns = ['Pred'])
y_hat.append(y_hat_)
y_hat__.index = list(dates.values)
plt.plot(x.values,y_hat__,color = 'dodgerblue', linestyle = '--')
e_ = y_hat_.values-np.reshape(data['Infected'].values,[-1,1])
e.append(e_)
#plt.plot(e_)
mse_ = np.sum(np.power(e_,2))/len(e_)
mse.append(mse_)

# plt.axvline(x=str(data_.index.values[len(data)])[:10], color = 'k')

# plt.xticks(rotation=45)
plt.xlabel('Fecha (Meses)', fontsize=18)
plt.ylabel('Numero de Casos de Infeccion de Coronavirus',fontsize=18)
plt.title('Pronostico de Infeccion por Novel coronavirus (2019-nCoV)', fontsize = 24)
plt.grid()


# =============================================================================
# %% END
# =============================================================================




