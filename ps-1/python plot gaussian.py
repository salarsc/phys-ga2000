# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np 
import matplotlib.pyplot as plt

#the constants
pi=np.pi

#defining the function gaussian
"""
calculates gaussian
x is a variablr ,sig is standard devitation , mu is the average
it return a normalized gaussian with mean mu and deviation sig
"""

def g(x, mu, sig):
    return (1 / (sig * np.sqrt(2 * pi))) * np.exp(-((x - mu) ** 2) / (2 * (sig ** 2)))
#now we creat an array :

x=np.linspace(-10,10 ,1000)
#now we create the gaussian output using our function

y=  g(x,0,3) 

#now we plot:

plt.scatter(x,y,s=1,c="black") 
plt.title("gaussian")
plt.xlabel("input")
plt.ylabel("output")
  

#now we save
plt.savefig("gaussian.png")


