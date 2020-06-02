"""
Cahn-Hilliard equation with Neumann homogeneous conditions.

  phi'= gamma * Laplace(w)                          in the unit square
  w = - epsilon^2 * Laplace(phi) + (phi^2-1)^2      in the unit square
  grad(phi) * n = grad(w) * n = 0                   on the boundary
  phi = random data between -0.01 and 0.01          at t = 0

We will comupute the energy functional

E = epsilon^2/2 * \int_\Omega |\nabla \phi|^2 + \int_\Omega (phi^2-1)^2

in each time step.
"""

from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib.pyplot as plt

T = 1.0            # final time
num_steps = 50     # number of time steps
dt = T / num_steps # time step size
eps = 0.01
gamma = 1

# Create mesh and define function space
nx = ny = 100 # Boundary points
mesh = UnitSquareMesh(nx,ny)

plot(mesh)
plt.show()

deg = 1 # Degree of polynomials in discrete space
P = FiniteElement("Lagrange", mesh.ufl_cell(), deg) # Space of polynomials
W = FunctionSpace(mesh, MixedElement([P,P])) # Space of functions

# Random initial data
u_0 = Expression(('0.02*(0.5- rand())','0'), degree=deg) # Random values between -0.01 and 0.01
u_n = interpolate(u_0,W)

phi_n,w_n = u_n.split(True)
print('max = %f' % (phi_n.vector().get_local().max()))
print('min = %f' % (phi_n.vector().get_local().min()))
c = plot(phi_n)
plt.colorbar(c)
plt.show()

print('mass = %f' % (assemble(phi_n*dx)))

# Define the energy vector
E = []

# Define variational problem
u = TrialFunction(W) # Meaningless function used to define the variational formulation
v = TestFunction(W) # Meaningless function used to define the variational formulation

phi, w = split(u)
barw, barphi = split(v)

a1 = phi * barw * dx + dt * gamma * dot(grad(w),grad(barw)) * dx
L1 = phi_n * barw * dx

a2 = w * barphi * dx - pow(eps,2) * dot(grad(phi),grad(barphi)) * dx - 2 * phi * barphi * dx
L2 = pow(phi_n,3) * barphi * dx - 3 * phi_n * barphi * dx

a = a1 + a2
L = L1 + L2

# Time-stepping
u = Function(W)
t = 0
for n in range(num_steps):

    # Update current time
    t += dt

    # Compute solution
    solve(a == L, u)

    phi, w = u.split(True)

    # Plot solution
    pic = plot(phi,mode='color')
    plt.title("Ecuación del Cahn-Hilliard en t = %.2f" %(t))
    plt.colorbar(pic)
    plt.show()

    # Compute the mass
    print('mass = %f' % (assemble(phi*dx)))

    # Compute the energy
    energy = assemble(0.5*pow(eps,2)*dot(grad(phi),grad(phi))*dx + pow(pow(phi,2)-1,2)*dx)
    E.append(energy)
    print('E =',energy)


    # Update previous solution
    phi_n.assign(phi)

plt.plot(np.linspace(0,T,num_steps),E, color='red')
plt.title("Funcional de energía")
plt.xlabel("Tiempo")
plt.ylabel("Energía")
plt.show()