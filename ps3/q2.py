import numpy as np
from matplotlib.pyplot import plot,savefig,title,show,legend,xlabel,ylabel
from random import random
random()

NBI13=1000
NTI=0
NPB=0
NBI9=0


#time cycle
h=1
tmax=20000

#calculating the decay probablities in each time slot for each :
ppb=1-2**(-h/3.3/60)    
pti=1-2**(-h/2.2/60) 
pbi13=1-2**(-h/46/60) 

#plot point lists
 

bi9list=[]
pblist=[]
tilist=[]
bi13list=[]
tlist=np.arange(0.0,tmax,h)
for s in tlist:
    
    
    
    bi9list.append(NBI9)
    pblist.append(NPB)
    tilist.append(NTI)
    bi13list.append(NBI13)
    
    #decayings
    
    for i in range(NPB):
        if random()<ppb:
            NPB-=1
            NBI9+=1
    
    for i in range(NTI):
        if random()<pti:
            NTI-=1
            NPB+=1
    for i in range(NBI13):
        if random()<pbi13:
            NBI13-=1
            if random()<0.9791:
                NPB+=1
            else:
                NTI+=1
        
plot(tlist,bi9list,label='Bi209')
plot(tlist,pblist,label='Pb209')
plot(tlist,tilist,label='Ti209')
plot(tlist,bi13list,label='Bi213')
title("the exponentional decay chain of BI213")
legend()
xlabel('time, sec')
ylabel('Number')
savefig("figq2.png")
show()
