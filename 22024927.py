# -*- coding: utf-8 -*-
"""
Created on Thu May 11 22:44:39 2023

@author: Shiva Kumar Sundara Murthy
Student ID : 22024927
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.cluster import KMeans

def converttonumbers(df, columnlist):
    df = df.replace(to_replace="..", value="0")
    df[columnlist] = df[columnlist].apply(pd.to_numeric)
    return df

def equation(x, a, b, c):
    '''
    This is the function which returns the quadratic function.
    Quadratic equation used for curve fitting.

    Args:
        x (numpy.ndarray): Input array.
        a (float): Coefficient of x^2 term.
        b (float): Coefficient of x term.
        c (float): Constant term.

    Returns:
        numpy.ndarray: Result of the quadratic equation.
    '''
    return a * x**2 + b * x + c

def curveFit(df_CO2, df_elec, country_code, years):
    '''
    Perform curve fitting and plot the curve.

    Args:
        df_CO2 (pandas.DataFrame): DataFrame containing CO2 emission data.
        df_elec (pandas.DataFrame): DataFrame containing electricity production data.
        country_code (str): Country code for the specific country.
        years (list): List of years to consider for curve fitting.

    Returns:
        None
    '''
    df_CO2 = df_CO2.replace(to_replace="..", value="0")
    df_CO2 = df_CO2.loc[df_CO2['Country Code'] == country_code][years]
    df_CO2 = df_CO2.dropna(axis=1)
    
    df_elec = df_elec.replace(to_replace="..", value="0")
    df_elec = df_elec.loc[df_elec['Country Code'] == country_code][years]
    df_elec = df_elec.dropna(axis=1)
    
    years = df_CO2.columns[4:].tolist()
    
    x, _ = opt.curve_fit(equation, df_CO2[years].values.flatten(), df_elec[years].values.flatten())
    
    df_elec[years] = equation(df_CO2[years].values.flatten(), *x)
    
    plt.figure()
    plt.plot(df_CO2[years].values.flatten(), df_elec[years].values.flatten(), color='green')
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend(["Expected"], loc="upper right")
    plt.title("Curve Fit of " + country_code)
    plt.show()

data_CO2 = pd.read_csv("CO2Emission.csv", skiprows=4)
df_CO2 = pd.DataFrame(data_CO2)

data_elec = pd.read_csv("Electricityproductionfromcoal.csv", skiprows=4)
df_elec = pd.DataFrame(data_elec)

years = [str(year) for year in range(1990, 2015)]

country_code = "GBR"  # Country code for United Kingdom
labels = ["CO2 Emission", "Electricity Production from Coal Sources"]

curveFit(df_CO2, df_elec, country_code, years)

years = ["2015"]
# invoking converttonumbers method to change the data of 2015 to numeric form
df = converttonumbers(data_CO2, years)
df1 = converttonumbers(data_elec, years)
x = df["2015"]
y = df1["2015"]

# Remove inf and NaN values from x and y
x = x[np.isfinite(x)]
y = y[np.isfinite(y)]

# Align x and y arrays based on common indices
common_indices = np.intersect1d(x.index, y.index)
x = x.loc[common_indices]
y = y.loc[common_indices]

# using curve fit
popt, covar = opt.curve_fit(equation, x, y)
x.name = "CO2 Emission"
y.name = "Electricity Production from Coal Sources"
# using pandas series merge()
df_fit = pd.merge(x, y, right_index=True, left_index=True)

# applying clustering for 2015 data
noclusters = 4
kmeans = KMeans(n_clusters=noclusters)
# fitting the data
kmeans.fit(df_fit.values)
# getting labels
labels = kmeans.labels_
# finding centers
cen = kmeans.cluster_centers_

# plotting the figure
plt.figure(figsize=(6.0, 6.0))
# Individual colors can be assigned to symbols. The label l is used to select the l-th number from the color table.
plt.scatter(df_fit["CO2 Emission"], df_fit["Electricity Production from Coal Sources"])
plt.xlabel("CO2 Emission")
plt.ylabel("Electricity Production from Coal Sources")
plt.title("Clustering of 2015 Data")
plt.show()
