"""
Cahn-Hilliard equation with Neumann homogeneous conditions.

  phi'= gamma * Laplace(w) + s(x,t)                 in the unit square
  w = - epsilon^2 * Laplace(phi) + F'(phi)          in the unit square
  grad(phi) * n = grad(w) * n = 0                   on the boundary
  phi = 0.1 * sin(0.5 * x[0]) * sin(0.5 * x[1]      at t = 0

where F(phi) = (phi^2-1)^2.

We will comupute the energy functional

E = epsilon^2/2 * \int_\Omega |\nabla \phi|^2 + \int_\Omega U^2

in each time step.

DG semidiscrete space scheme and EQ semidicrete time scheme
"""

from fenics import *
import numpy as np
import matplotlib.pyplot as plt

T = 0.05           # final time
num_steps = 1000     # number of time steps
dt = T / num_steps # time step size
eps = Constant(0.1)
gamma = Constant(1.0)

print("dt = %f" %(dt))

# Create mesh and define function space
nx = ny = 16 # Boundary points
print("nx = ny = %f" %(nx))

mesh = RectangleMesh(Point(-pi,3*pi), Point(3 * pi, -pi), nx, ny, "right/left")

plot(mesh)
plt.show()

print("h = %f" %(mesh.hmax()))

deg = 2 # Degree of polynomials in discrete space
P = FiniteElement("Lagrange", mesh.ufl_cell(), deg) # Space of polynomials
W = FunctionSpace(mesh, MixedElement([P,P])) # Space of functions
V = FunctionSpace(mesh, P)

# Source term
g1 = Expression('0.1 * exp(-0.25 * t) * sin(0.5 * x[0]) * sin(0.5 * x[1])', degree = deg, t=0) # exact solution
g2 = Expression('pow(0.1 * exp(-0.25 * t) * cos(0.5 * x[0]) * sin(0.5 * x[1]),2) + pow(0.1 * exp(-0.25 * t) * sin(0.5 * x[0]) * cos(0.5 * x[1]),2)',degree=deg, t=0)
s = Expression('- 0.25 * g1 + pow(eps,2) * g1 * 0.25 - 1.5 * g1 * g2 + 1.5 * pow(g1,3) - 0.5 * g1', degree=deg, g1=g1, g2=g2, eps=eps) # source term
#s = Expression('- 0.25 * (0.1 * exp(-0.25 * t) * sin(0.5 * x[0]) * sin(0.5 * x[1])) + pow(eps,2) * (0.1 * exp(-0.25 * t) * sin(0.5 * x[0]) * sin(0.5 * x[1])) * 0.25 - 1.5 * (0.1 * exp(-0.25 * t) * sin(0.5 * x[0]) * sin(0.5 * x[1])) * (pow(0.1 * exp(-0.25 * t) * cos(0.5 * x[0]) * sin(0.5 * x[1]),2) + pow(0.1 * exp(-0.25 * t) * sin(0.5 * x[0]) * cos(0.5 * x[1]),2)) + 1.5 * pow(0.1 * exp(-0.25 * t) * sin(0.5 * x[0]) * sin(0.5 * x[1]),3) - 0.5 * (0.1 * exp(-0.25 * t) * sin(0.5 * x[0]) * sin(0.5 * x[1]))', degree=deg, t=0, eps=eps) # source term

# Initial data

phi_n = interpolate(g1,V)

c = plot(phi_n)
plt.title("Condición inicial")
plt.colorbar(c)
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
L1 = phi_n * barw * dx + dt * s * barw * dx

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

    # Update source term
    g1.t = t
    g2.t = t
    s.g1 = g1
    s.g2 = g2
    #s.t = t

    # Compute solution
    solve(a == L, u)

    phi, w = u.split(True)

    #fa = FunctionAssigner([V, V], W)
    #phi, w = Function(V), Function(V)
    #fa.assign([phi, w], u)

    # Plot solution
    #pic = plot(phi)
    #plt.title("Ecuación del Cahn-Hilliard en t = %.2f" %(t))
    #plt.colorbar(pic)
    #plt.show()

    # Compute the mass
    print('mass = %f' % (assemble(phi * dx)))


    # Update previous solution
    phi_n.assign(phi)

    # Compute the energy
    energy = assemble(0.5*pow(eps,2)*dot(grad(phi),grad(phi))*dx + 0.25 * pow(pow(phi,2)-1,2)*dx)
    E.append(energy)
    print('E =',energy)

pic = plot(phi)
plt.title("Ecuación de Cahn-Hilliard en t = %.2f" %(t))
plt.colorbar(pic)
plt.show()

print("Error en norma L2 = %.10f" %(sqrt(assemble(pow(phi-g1,2)*dx))))
print("Error en norma L_inf = %.10f" %(np.abs(phi.vector().get_local() - interpolate(g1,V).vector().get_local()).max()))

pic = plot(interpolate(g1,V))
plt.title("Ecuación de Cahn-Hilliard en t = %.2f" %(t))
plt.colorbar(pic)
plt.show()

plt.plot(np.linspace(0,T,num_steps+1),E, color='red')
plt.title("Funcional de energía")
plt.xlabel("Tiempo")
plt.ylabel("Energía")
plt.show()
