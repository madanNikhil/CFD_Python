# -*- coding: utf-8 -*-
"""
Created on Thu Jan 08 14:45:38 2015

@author: nikhil madan

# CFD Tool to analyze Trapped Vortex Combution chamber design - Master thesis at

Warsaw University of Technology, Poland, March 2015.

Discretization of Navier Stokes Eqn. and solving it numerically on the defined mesh.
"""
# import things

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

# define parameters

nx = 41			# no. of nodes in x-direction
ny = 41			# no. of nodes in y-direction
nt = 100		# no. of time steps
nit=50			# no. of iterations
c = 1			# normalized input velocity
dx = 8.0/(nx-1) # unit x-distance
dy = 8.0/(ny-1)	# unit y-distance

# create mesh

x = np.linspace(0,8,nx)
y = np.linspace(0,12,ny)
Y,X = np.meshgrid(y,x)

# fluid properties -- air in this case
rho = 1
nu = .1
dt = .001
k= 0.9
m= 1
cp = 1.004
c = 0.71
h  = 280.3
gamma = 1.4

#init velocities, pressures, tempratures

u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx)) 
b = np.zeros((ny, nx)) # intermediatory variable for the ease of splitting a longer equation
T = np.zeros((ny, nx))


# function to evaluate intermediatory variable

def buildUpB(b, rho, dt, u, v, dx, dy):
    
    b[1:-1,1:-1]=rho*(1/dt*((u[2:,1:-1]-u[0:-2,1:-1])/(2*dx)+(v[1:-1,2:]-v[1:-1,0:-2])/(2*dy))-\
		((u[2:,1:-1]-u[0:-2,1:-1])/(2*dx))**2-\
		2*((u[1:-1,2:]-u[1:-1,0:-2])/(2*dy)*(v[2:,1:-1]-v[0:-2,1:-1])/(2*dx))-\
		((v[1:-1,2:]-v[1:-1,0:-2])/(2*dy))**2)
	
    return b

# function to evaluate pressure variable

def presPoisson(p, dx, dy, b):
    pn = np.empty_like(p)
    pn = p.copy()
    
    for q in range(nit):
		pn = p.copy()
		p[1:-1,1:-1] = ((pn[2:,1:-1]+pn[0:-2,1:-1])*dy**2+(pn[1:-1,2:]+pn[1:-1,0:-2])*dx**2)/\
			(2*(dx**2+dy**2)) -\
			dx**2*dy**2/(2*(dx**2+dy**2))*b[1:-1,1:-1]
		
		p[-1,:] =0		##dp/dy = 0 at y = 2
		p[0,:] = p[1,:]	 	##dp/dy = 0 at y = 0
		p[:,0]=p[:,1]		   ##dp/dx = 0 at x = 0
		p[:,-1]= p[:,-2]		       ##p = 0 at x = 2		
		  
        
    return p
    
# function to evaluate output parameters

def trappedCavityFlow(nt, u, v, dt, dx, dy, p, rho, nu, T):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    Tn = np.empty_like(T)
    
    b = np.zeros((ny, nx))
    
    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        Tn = T.copy()
        
        
        b = buildUpB(b, rho, dt, u, v, dx, dy)
        p = presPoisson(p, dx, dy, b)
        
        # x-velocity component

        u[1:-1,1:-1] = un[1:-1,1:-1]-\
            un[1:-1,1:-1]*dt/dx*(un[1:-1,1:-1]-un[0:-2,1:-1])-\
            vn[1:-1,1:-1]*dt/dy*(un[1:-1,1:-1]-un[1:-1,0:-2])-\
            dt/(2*rho*dx)*(p[2:,1:-1]-p[0:-2,1:-1])+\
            nu*(dt/dx**2*(un[2:,1:-1]-2*un[1:-1,1:-1]+un[0:-2,1:-1])+\
            dt/dy**2*(un[1:-1,2:]-2*un[1:-1,1:-1]+un[1:-1,0:-2]))

         # y-velocity component   
	
        v[1:-1,1:-1] = vn[1:-1,1:-1]-\
            un[1:-1,1:-1]*dt/dx*(vn[1:-1,1:-1]-vn[0:-2,1:-1])-\
            vn[1:-1,1:-1]*dt/dy*(vn[1:-1,1:-1]-vn[1:-1,0:-2])-\
            dt/(2*rho*dy)*(p[1:-1,2:]-p[1:-1,0:-2])+\
            nu*(dt/dx**2*(vn[2:,1:-1]-2*vn[1:-1,1:-1]+vn[0:-2,1:-1])+\
            (dt/dy**2*(vn[1:-1,2:]-2*vn[1:-1,1:-1]+vn[1:-1,0:-2])))
         
         # thermal component   

        T[1:-1,1:-1]=Tn[1:-1,1:-1]-\
            (2.0/(c*(1-2*gamma)))*un[1:-1,1:-1]*(u[1:-1,1:-1]-un[1:-1,1:-1])-(2.0/(c*(1-2*gamma)))*vn[1:-1,1:-1]*(v[1:-1,1:-1]-vn[1:-1,1:-1])+\
            (2*k*dt/(c*(1-2*gamma)*rho*dx**2))*(Tn[2:,1:-1]-2*Tn[1:-1,1:-1]+Tn[0:-2,1:-1])+\
            (2*k*dt/(c*(1-2*gamma)*rho*dy**2))*(Tn[1:-1,2:]-2*Tn[1:-1,1:-1]+Tn[1:-1,0:-2])
            
        # Defined boundry conditions
        
        u[0,:((ny-1)/3)] = 0
        u[0,((ny-1)/3):(2*(ny-1)/3)]= 3
        u[0,(2*(ny-1)/3):ny] = 0
        u[:,0] = 0
        u[:,-1] = 0
        v[:((nx-1)/16),0] = 3
        v[((nx-1)/16):nx,0] = 0
        v[-1,:]=0
        v[:,0] = 0
        v[:((nx-1)/16),-1] = -3
        v[((nx-1)/16):nx,-1] = 0
        u[-1,:] = 0
        T[:,0] = 300
        T[:,-1] = 300
        T[-1, : ((ny-1)/4)] = 300
        T[0,: ((ny-1)/6)] = 1000
        T[0,((ny-1)/6):((ny-1)/3)] = 300
        T[0,(2*(ny-1)/3):(5*(ny-1)/6)] = 300
        T[0,(5*(ny-1)/6):(ny)]= 1000
        
    return u, v, p, T

# Output of results

u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))
nt = 500
u, v, p, T = trappedCavityFlow(nt, u, v, dt, dx, dy, p, rho, nu, T)
#fig = plt.figure(figsize=(11,7), dpi=100)
#plt.contourf(X,Y,p,alpha=0.5)    ###plnttong the pressure field as a contour
#plt.colorbar()
#plt.contour(X,Y,p)               ###plotting the pressure field outlines
#plt.quiver(X[::,::],Y[::,::],u[::,::],v[::,::]) ##plotting velocity
#plt.xlabel('X')
#plt.ylabel('Y')
print "u is",u[1:,1:]
print "v is",v[1:,1:]

