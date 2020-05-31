"""
FEniCS tutorial demo program: Heat equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.

  u'= Laplace(u) in the unit square
  u = 0            on the boundary
  u = alpha*e^(x^2+y^2)            at t = 0
"""

from __future__ import print_function
from fenics import *
from mshr import * # Paquete para crear las mallas
import numpy as np
import matplotlib.pyplot as plt

T = 2.0            # final time
num_steps = 10     # number of time steps
dt = T / num_steps # time step size
alpha = 1

# Create mesh and define function space
nf = 100 # Boundary points
resolution = 60 # Mesh resolution
circle = Circle(Point(0.,0.),1.0,nf)
mesh = generate_mesh(circle,resolution)

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
u_0 = Expression('alpha * exp(x[0]*x[0]+x[1]*x[1])',degree = deg,alpha = alpha)
u_n = interpolate(u_0, V) # Both interpolates u_D into V
#u_n = project(u_D, V)

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
    plot(u)
    plt.show()


    # Compute the energy
    energy = assemble(0.5*u*u*dx)
    print('E =',energy)


    # Update previous solution
    u_n.assign(u)
