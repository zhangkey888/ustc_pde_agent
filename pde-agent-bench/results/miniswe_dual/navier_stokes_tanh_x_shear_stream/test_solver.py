import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import basix.ufl
import ufl
from petsc4py import PETSc
import time

comm = MPI.COMM_WORLD
nu_val = 0.16

for N in [32, 48, 64, 96]:
    t0 = time.time()
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    cell_name = domain.topology.cell_name()
    
    degree_u = 2
    V_el = basix.ufl.element("Lagrange", cell_name, degree_u, shape=(2,))
    Q_el = basix.ufl.element("Lagrange", cell_name, 1)
    ME = basix.ufl.mixed_element([V_el, Q_el])
    W = fem.functionspace(domain, ME)
    V = fem.functionspace(domain, basix.ufl.element("Lagrange", cell_name, degree_u, shape=(2,)))
    
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    u_exact_0 = pi * ufl.tanh(6.0 * (x[0] - 0.5)) * ufl.cos(pi * x[1])
    u_exact_1 = -6.0 * (1.0 - ufl.tanh(6.0 * (x[0] - 0.5))**2) * ufl.sin(pi * x[1])
    u_exact = ufl.as_vector([u_exact_0, u_exact_1])
    p_exact = ufl.sin(pi * x[0]) * ufl.cos(pi * x[1])
    
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    
    f = (ufl.grad(u_exact) * u_exact
         - nu * ufl.div(ufl.grad(u_exact))
         + ufl.grad(p_exact))
    
    F_form = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda x: np.stack([
        np.pi * np.tanh(6.0 * (x[0] - 0.5)) * np.cos(np.pi * x[1]),
        -6.0 * (1.0 - np.tanh(6.0 * (x[0] - 0.5))**2) * np.sin(np.pi * x[1])
    ]))
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    bcs = [bc_u]
    
    # Initial guess: exact solution
    w.sub(0).interpolate(lambda x: np.stack([
        np.pi * np.tanh(6.0 * (x[0] - 0.5)) * np.cos(np.pi * x[1]),
        -6.0 * (1.0 - np.tanh(6.0 * (x[0] - 0.5))**2) * np.sin(np.pi * x[1])
    ]))
    w.sub(1).interpolate(lambda x: np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]))
    
    problem = petsc.NonlinearProblem(
        F_form, w, bcs=bcs,
        petsc_options_prefix=f"ns{N}_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_rtol": 1e-12,
            "snes_atol": 1e-14,
            "snes_max_it": 30,
            "ksp_type": "preonly",
            "pc_type": "lu",
        }
    )
    
    problem.solve()
    w.x.scatter_forward()
    
    n_newton = problem.solver.getIterationNumber()
    
    u_h = w.sub(0).collapse()
    
    error_form = fem.form(ufl.inner(u_h - u_exact, u_h - u_exact) * ufl.dx)
    l2_error = np.sqrt(comm.allreduce(fem.assemble_scalar(error_form), op=MPI.SUM))
    
    elapsed = time.time() - t0
    print(f"N={N:3d}: L2 error = {l2_error:.2e}, Newton iters = {n_newton}, time = {elapsed:.2f}s")
