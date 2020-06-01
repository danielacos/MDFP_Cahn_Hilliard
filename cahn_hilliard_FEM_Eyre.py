"""
Heat equation with Dirichlet homogeneous conditions.

  u'= Laplace(u) in the unit square
  u = 0            on the boundary
  u = alpha*e^(x^2+y^2)            at t = 0

We will comupute the energy functional E=\int_\omega u^2 in each time step
"""

from __future__ import print_function
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import random

T = 2.0            # final time
num_steps = 10     # number of time steps
dt = T / num_steps # time step size
eps = 0.01
gamma = 0.0001

# Create mesh and define function space
nx = ny = 10 # Boundary points
mesh = UnitSquareMesh(nx,ny)

plot(mesh)
plt.show()

deg = 1 # Degree of polynomials in discrete space
V = FunctionSpace(mesh, "Lagrange", deg)

# Define boundary condition
u_D = Constant(0)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

random.seed(867658767987)
u_0 =  Expression('0.02*(0.5- rand())', degree=deg) # Random values between -0.01 and 0.01
u_n = interpolate(u_0,V)
print('max = %f' % (u_n.vector().get_local().max()))
print('min = %f' % (u_n.vector().get_local().min()))
c = plot(u_n)
plt.colorbar(c)
plt.show()

w_n = - eps**2 * div(grad(u_n)) + pow(u_0,3) - 3 * pow(u_0,2) + 2 * u_0
plot(w_n)
plt.show()

# Define the energy vector
E = []

# Define variational problem
u = TrialFunction(V) # Meaningless function used to define the variational formulation
v = TestFunction(V) # Meaningless function used to define the variational formulation

a1 = u*v*dx + dt*dot(grad(u), grad(v))*dx
L = u_n*v*dx

# Time-stepping
u = Function(V)
t = 0
for n in range(num_steps):

    # Update current time
    t += dt

    # Compute solution
    solve(a == L, u, bc)

    # Plot solution
    pic = plot(u)
    # ,mode='color')
    plt.title("Ecuación del calor en t = %.2f" %(t))
    plt.colorbar(pic)
    plt.show()


    # Compute the energy
    energy = assemble(0.5*u*u*dx)
    E.append(energy)
    print('E =',energy)


    # Update previous solution
    u_n.assign(u)

plt.plot(np.linspace(0,T,num_steps),E, color='red')
plt.title("Funcional de energía")
plt.xlabel("Tiempo")
plt.ylabel("Energía")
plt.show()
