"""
Heat equation with Dirichlet homogeneous conditions.

  u'= Laplace(u) in the unit square
  u = 0            on the boundary
  u = alpha*e^(x^2+y^2)            at t = 0

We will comupute the energy functional E=\int_\omega u^2 in each time step
"""

from __future__ import print_function
from dolfin import *
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import random

# Class representing the intial conditions
class IC(UserExpression):
    def eval(self,values,x):
        values[0] = 1.0*np.random() + 0.25
    def value_shape(self):
        return(1,)

def u_init(x):
    """Initialise values for c and mu."""
    values = np.zeros((1, x.shape[1]))
    values[0] = 0.63 + 0.02 * (0.5 - np.random.rand(x.shape[1]))
    return values

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

# Define initial value
u_0 =  Expression('1.0*random() + 0.25', degree=deg)
u_n = interpolate(u_0,V)
plot(u_n)
plt.show()

w_n = - eps**2 * div(grad(u_n)) + pow(u_0,3) - 3 * pow(u_0,2) + 2 * u_0



plot(w_n)
plt.show()

# Define the energy vector
E = []

# Define variational problem
u = TrialFunction(V) # Meaningless function used to define the variational formulation
v = TestFunction(V) # Meaningless function used to define the variational formulation

a = u*v*dx + dt*dot(grad(u), grad(v))*dx
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
