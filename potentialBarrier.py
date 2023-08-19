import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal 
from matplotlib.animation import FuncAnimation
l=2
j=0
V= lambda x: 1000 if 1<x<1.5 else 0
V= np.vectorize(V)
x= np.linspace(-5,5,1000)
dx2= (x[1]-x[0])**2
diagonalElements= 2*(1+dx2*V(x[1:-1])) #we dont want first and last element
secondDiagonal=-1*np.ones(len(diagonalElements)-1)

psi0= np.exp(-(x+4)**2)*np.exp(1j*50*x)
normalPsi0= psi0/(simps(np.absolute(psi0)**2,x))**0.5
eigenValues, eigenFunctions= eigh_tridiagonal(diagonalElements, secondDiagonal)
energy_n=lambda n:eigenValues[n]
print(energy_n(10))
psi_n=lambda n : eigenFunctions.T[n]/(simps(eigenFunctions.T[n]**2,x[1:-1]))**0.5

Cn= lambda n: simps(psi_n(n)*normalPsi0[1:-1],x[1:-1])
psi_nt= lambda t:sum([Cn(n)*psi_n(n)*np.exp(-1j*t *energy_n(n)) for n in range(1,200)])


fig, ax= plt.subplots()

#lets simulate the wave funtion
def animate(t):
    ax.clear()
    ax.set_ylim([-4,5])
    ax.plot(x,V(x))
    ax.plot(x[1:-1], psi_nt(t*50), label="real part")
    ax.plot(x[1:-1], np.imag(psi_nt(t*50)), label="imaginary part")

plt.legend()
ani= FuncAnimation(fig=fig,func=animate,frames=50,interval=20)
ani.save("hua.mp4",'ffmpeg',fps=5)
plt.show()
