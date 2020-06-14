"""
Cahn-Hilliard equation with Neumann homogeneous conditions.

  phi'= gamma * Laplace(w)                          in the unit square
  w = - epsilon^2 * Laplace(phi) + F'(phi)          in the unit square
  grad(phi) * n = grad(w) * n = 0                   on the boundary
  phi = random data between -0.01 and 0.01          at t = 0

where F(phi) = (phi^2-1)^2.

We will comupute the energy functional

E = epsilon^2/2 * \int_\Omega |\nabla \phi|^2 + \int_\Omega (phi^2-1)^2

in each time step.

FEM semidiscrete space scheme and Eyre semidicrete time scheme
"""

from fenics import *
import random
import numpy as np
import matplotlib.pyplot as plt

T = 0.05            # final time
num_steps = 100     # number of time steps
dt = T / num_steps # time step size
eps = 0.01
gamma = 1.0

savepic = 0 # Indicates if pictures are saved or not

print("dt = %f" %(dt))

# Create mesh and define function space
nx = ny = 100 # Boundary points
print("nx = ny = %d" %(nx))

mesh = UnitSquareMesh(nx,ny)

plot(mesh)
plt.show()

deg = 1 # Degree of polynomials in discrete space
P = FiniteElement("Lagrange", mesh.ufl_cell(), deg) # Space of polynomials
W = FunctionSpace(mesh, MixedElement([P,P])) # Space of functions
V = FunctionSpace(mesh, P)

# Random initial data
random.seed(1)
class Init_u(UserExpression):
    def eval(self, values, x):
        values[0] = random.uniform(-0.01,0.01)

phi_0 = Init_u(degree=deg) # Random values between -0.01 and 0.01
phi_n = interpolate(phi_0,V)

pic = plot(phi_n)
plt.title("Condición inicial")
plt.colorbar(pic)
plt.show()

print('max = %f' % (phi_n.vector().get_local().max()))
print('min = %f' % (phi_n.vector().get_local().min()))
print('mass = %f' % (assemble(phi_n*dx)))

# Define the energy vector
E = []
energy = assemble(0.5*pow(eps,2)*dot(grad(phi_n),grad(phi_n))*dx + 0.25 * pow(pow(phi_n,2)-1,2)*dx)
E.append(energy)
print('E =',energy)

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

print("Iteraciones:")

for i in range(num_steps):

    print("\nIteración %d:" %(i+1))

    # Update current time
    t += dt

    # Compute solution
    solve(a == L, u)

    phi, w = u.split(True)

    # Plot solution
    if(savepic):
        if(i==0 or i==(num_steps/2-1) or i==(num_steps-1)):
            pic = plot(phi)
            plt.title("Función de campo de fase en t = %.4f" %(t))
            plt.colorbar(pic)
            plt.savefig("fig/FEM-Eyre_nt-%d_t-%.4f.png" %(num_steps,t))
            plt.close()

    # Compute the mass
    print('mass = %f' % (assemble(phi*dx)))

    # Update previous solution
    phi_n.assign(phi)

    # Compute the energy
    energy = assemble(0.5*pow(eps,2)*dot(grad(phi),grad(phi))*dx + 0.25 * pow(pow(phi,2)-1,2)*dx)
    E.append(energy)
    print('E =',energy)

pic = plot(phi)
plt.title("Función de campo de fase en t = %.2f" %(t))
plt.colorbar(pic)
plt.show()

plt.plot(np.linspace(0,T,num_steps+1),E, color='red')
plt.title("Energía discreta")
plt.xlabel("Tiempo")
plt.ylabel("Energía")
if(savepic):
    plt.savefig("fig/FEM-Eyre_nt-%d_energia.png" %(num_steps))
plt.show()
