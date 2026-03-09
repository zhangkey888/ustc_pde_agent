import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import basix.ufl
import ufl
from petsc4py import PETSc
import time

comm = MPI.COMM_WORLD
nu_val = 0.15

for N in [80, 96, 128]:
    t_start = time.time()
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    cell_name = domain.topology.cell_name()
    P2_vec = basix.ufl.element("Lagrange", cell_name, 2, shape=(domain.geometry.dim,))
    P1_el = basix.ufl.element("Lagrange", cell_name, 1)
    ME = basix.ufl.mixed_element([P2_vec, P1_el])
    W = fem.functionspace(domain, ME)
    V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim,)))
    
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    x = ufl.SpatialCoordinate(domain)
    pi_ufl = ufl.pi
    u_exact = ufl.as_vector([
        pi_ufl * ufl.exp(2 * x[0]) * ufl.cos(pi_ufl * x[1]),
        -2.0 * ufl.exp(2 * x[0]) * ufl.sin(pi_ufl * x[1])
    ])
    p_exact = ufl.exp(x[0]) * ufl.cos(pi_ufl * x[1])
    
    grad_u_exact = ufl.grad(u_exact)
    convection_exact = ufl.dot(grad_u_exact, u_exact)
    laplacian_u_exact = ufl.div(ufl.grad(u_exact))
    grad_p_exact = ufl.grad(p_exact)
    f = convection_exact - nu_val * laplacian_u_exact + grad_p_exact
    
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    
    F_form = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.dot(ufl.grad(u), u), v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda x: np.vstack([
        np.pi * np.exp(2 * x[0]) * np.cos(np.pi * x[1]),
        -2.0 * np.exp(2 * x[0]) * np.sin(np.pi * x[1])
    ]))
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    bcs = [bc_u]
    
    w.x.array[:] = 0.0
    w.x.scatter_forward()
    
    problem = petsc.NonlinearProblem(
        F_form, w, bcs=bcs, petsc_options_prefix=f"ns{N}_",
        petsc_options={
            "snes_type": "newtonls", "snes_linesearch_type": "bt",
            "snes_rtol": 1e-12, "snes_atol": 1e-14, "snes_max_it": 50,
            "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
        }
    )
    
    problem.solve()
    reason = problem.solver.getConvergedReason()
    newton_its = problem.solver.getIterationNumber()
    w.x.scatter_forward()
    
    u_sol = w.sub(0).collapse()
    error_form = fem.form(ufl.inner(u_sol - u_exact, u_sol - u_exact) * ufl.dx)
    error_local = fem.assemble_scalar(error_form)
    error_global = np.sqrt(comm.allreduce(error_local, op=MPI.SUM))
    
    elapsed = time.time() - t_start
    print(f"N={N}: L2 error={error_global:.6e}, Newton its={newton_its}, reason={reason}, time={elapsed:.3f}s")
