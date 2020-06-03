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
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

def cahn_hilliard_DG_EQ_test():
    T = 0.01            # final time
    num_steps = 1000     # number of time steps
    dt = T / num_steps # time step size
    eps = Constant(0.1)
    gamma = Constant(1.0)
    sigma = Constant(10.0) # penalty parameter
    B  = Constant(1.0)

    # Create mesh and define function space
    nx = ny = 7 # Boundary points
    mesh = RectangleMesh(Point(-pi,3*pi), Point(3 * pi, -pi), nx, ny, "right/left")

    plot(mesh)
    plt.show()

    deg = 1 # Degree of polynomials in discrete space
    P = FiniteElement('DG', mesh.ufl_cell(), deg) # Space of polynomials
    W = FunctionSpace(mesh, MixedElement([P,P])) # Space of functions
    V = FunctionSpace(mesh, P)

    n = FacetNormal(mesh)
    h = CellDiameter(mesh)

    # Source term
    g1 = Expression('0.1 * exp(-t * 4) * sin(x[0]/2) * sin(x[1]/2)', degree = deg, t=0)
    g2 = Expression('pow(0.1 * exp(-t * 4) * cos(x[0]/2) * sin(x[1]/2),2) + pow(0.1 * exp(-t * 4) * sin(x[0]/2) * cos(x[1]/2),2)',degree=deg, t=0)
    s = Expression('- 0.25 * g1 + pow(eps,2) * g1 * 0.25 - 1.5 * g1 * g2 + 1.5 * g1 + 1.5 * pow(g1,3) - 0.5 * g1', degree=deg, g1=g1, g2=g2, eps=eps)

    # Initial data

    phi_n = interpolate(g1,V)
    print('max = %f' % (phi_n.vector().get_local().max()))
    print('min = %f' % (phi_n.vector().get_local().min()))
    c = plot(phi_n)
    plt.colorbar(c)
    plt.show()

    print('mass = %f' % (assemble(phi_n*dx)))

    U_n = project(sqrt(0.25 * pow(pow(phi_n,2) - 1.0,2) + B),V)

    # Define function H
    H = project((pow(phi_n,3) - phi_n)/sqrt(0.25 * pow(pow(phi_n,2) - 1.0,2) + B),V)

    # Define the energy vector
    E = []

    # Define variational problem
    u = TrialFunction(W) # Meaningless function used to define the variational formulation
    v = TestFunction(W) # Meaningless function used to define the variational formulation

    phi, w = split(u)
    barw, barphi = split(v)

    a1 = phi * barw * dx \
        + dt * gamma * (dot(grad(w),grad(barw)) * dx \
        - dot(avg(grad(w)),n('+'))*jump(barw) * dS \
        - dot(avg(grad(barw)),n('+'))*jump(w) * dS \
        + sigma/h('+') * dot(jump(w), jump(barw)) * dS)
    L1 = phi_n * barw * dx + dt * s * barw * dx

    a2 = w * barphi * dx \
        - pow(eps,2) * (dot(grad(phi),grad(barphi))*dx \
        - dot(avg(grad(phi)),n('+'))*jump(barphi) * dS \
        - dot(avg(grad(barphi)),n('+'))*jump(phi) * dS \
        + sigma/h('+') * dot(jump(phi), jump(barphi)) * dS) \
        - 0.5 * pow(H,2) * phi * barphi * dx
    L2 = H * U_n * barphi * dx \
        - 0.5 * pow(H,2) * phi_n * barphi * dx

    a = a1 + a2
    L = L1 + L2

    # Time-stepping
    u = Function(W)
    t = 0

    for i in range(num_steps):

        # Update current time
        t += dt

        g1.t = t
        g2.t = t

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
        U_n.assign(project(U_n+ 0.5 * H * (phi - phi_n ),V))
        H.assign(project((pow(phi ,3) - phi)/sqrt(0.25 * pow(pow(phi,2) - 1.0,2) + B),V))
        phi_n.assign(phi)

        # Compute the energy
        energy = assemble(0.5*pow(eps,2)*(dot(grad(phi),grad(phi))*dx - 2.0 * dot(avg(grad(phi)),n('+'))*jump(phi) * dS  + sigma/h('+') * pow(jump(phi),2) * dS) + pow(U_n,2) * dx)
        E.append(energy)
        print('E =',energy)

    pic = plot(phi)
    plt.title("Ecuación del Cahn-Hilliard en t = %.2f" %(t))
    plt.colorbar(pic)
    plt.show()

    print("Error = %f" %(assemble(pow(phi-g1,2)*dx)))

    plt.plot(np.linspace(0,T,num_steps),E, color='red')
    plt.title("Funcional de energía")
    plt.xlabel("Tiempo")
    plt.ylabel("Energía")
    plt.show()
