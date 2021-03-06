"""
FEniCS tutorial demo program: Heat equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.

  u'= Laplace(u) + f  in the unit square
  u = u_D             on the boundary
  u = u_0             at t = 0

  u = 1 + x^2 + alpha*y^2 + \beta*t
  f = beta - 2 - 2*alpha
"""

from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib.pyplot as plt

T = 2.0            # final time
num_steps = 10     # number of time steps
dt = T / num_steps # time step size
alpha = 3          # parameter alpha
beta = 1.2         # parameter beta

# Create mesh and define function space
nx = ny = 8
mesh = UnitSquareMesh(nx, ny)
deg = 1

plot(mesh)
plt.show()

V = FunctionSpace(mesh, "Lagrange", deg)

# Define boundary condition
u_D = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                 degree = deg, alpha=alpha, beta=beta, t=0) # degree indicates the projection degree of u_D onto the mesh

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define initial value
u_n = interpolate(u_D, V) # Both interpolates u_D into V
#u_n = project(u_D, V)

# Define variational problem
u = TrialFunction(V) # Meaningless function used to define the variational formulation
v = TestFunction(V) # Meaningless function used to define the variational formulation
f = Constant(beta - 2 - 2*alpha)

a = u*v*dx + dt*dot(grad(u), grad(v))*dx
L = (u_n + dt*f)*v*dx


# Time-stepping
u = Function(V)
t = 0
for n in range(num_steps):

    # Update current time
    t += dt
    u_D.t = t

    # Compute solution
    solve(a == L, u, bc)

    # Plot solution
    plot(u)
    plt.show()

    # Compute error at vertices
    u_e = interpolate(u_D, V)
    error = np.abs(u_e.vector().get_local() - u.vector().get_local()).max()
    print('t = %.2f: error = %.3g' % (t, error))

    # Update previous solution
    u_n.assign(u)
