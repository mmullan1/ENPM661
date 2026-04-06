import matplotlib.pyplot as plt
import numpy as np
import math

def cost(Xi,Yi,Thetai,UL,UR):
    t = 0
    r = 0.033
    L = 0.287
    dt = 0.1
    Xn=Xi
    Yn=Yi
    Thetan = 3.14 * Thetai / 180


    # Xi, Yi,Thetai: Input point's coordinates
    # Xs, Ys: Start point coordinates for plot function
    # Xn, Yn, Thetan: End point coordintes
    D=0
    while t<1:
        t = t + dt
        Xs = Xn
        Ys = Yn
        Delta_Xn = 0.5*r * (UL + UR) * math.cos(Thetan) * dt
        Delta_Yn = 0.5*r * (UL + UR) * math.sin(Thetan) * dt
        Thetan += (r / L) * (UR - UL) * dt
        D=D+ math.sqrt(math.pow((0.5*r * (UL + UR) * math.cos(Thetan) * dt),2)+math.pow((0.5*r * (UL + UR) * math.sin(Thetan) * dt),2))
    Thetan = 180 * (Thetan) / 3.14
    return Xn, Yn, Thetan, D
    
def run_actions(Xi, Yi, Thetai, RPM1, RPM2):
    actions=[[0, RPM1], [RPM1, 0],[RPM1, RPM1],[0, RPM2],[RPM2, 0],[RPM2, RPM2], [RPM1, RPM2], [RPM2, RPM1]]
        
    for action in actions:
        k=cost(Xi,Yi, Thetai, action[0],action[1]) 
        print(k)

run_actions(0, 0, 0, 5, 10)

    
