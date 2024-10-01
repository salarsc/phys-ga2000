from numpy import linspace
from pylab import *
sys.path.append('cpresources')
from gaussxw import gaussxw

#Part b (calculating the integral)
#first we initiate the gauss quadretue weights and points:
    
x,w= gaussxw(20)

#now we define the function  


N = 100
m = 1
v = lambda x: x**4
f=lambda x,a,M: sqrt(8*m)/sqrt(v(a)-v(x))

def T(aa):
    
    a=0
    b=aa
    
    #scaling:
    xp=0.5*(b-a)*x + 0.5*(b+a) 
    wp = 0.5*(b-a)*w
    
    I=sum(wp*f(xp,b,m))
    return I


A = linspace(0,2,100)
Ts = [T(As) for As in A]

plot(A,Ts)
xlabel('initial amplitude(A)')
ylabel('period (T)')    
title("the curve for period of the anharmonic motion vs the first amplitite place")
savefig("q 2")
