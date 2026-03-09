import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

comm = MPI.COMM_WORLD
N = 64
deg = 2

for eps_val in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", deg))
    x_coord = ufl.SpatialCoordinate(domain)
    
    f_ufl = 3.0 * ufl.cos(3.0 * ufl.pi * x_coord[0]) * ufl.sin(2.0 * ufl.pi * x_coord[1])
    
    def ic_func(X):
        return 0.2 * np.sin(3.0 * np.pi * X[0]) * np.sin(2.0 * np.pi * X[1])
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    bcs = [bc]
    
    u = fem.Function(V, name="u")
    u_n = fem.Function(V, name="u_n")
    v = ufl.TestFunction(V)
    
    u_n.interpolate(ic_func)
    u.x.array[:] = u_n.x.array[:]
    
    dt_val = 0.005
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))
    eps_c = fem.Constant(domain, PETSc.ScalarType(eps_val))
    
    # Allen-Cahn: R(u) = (1/eps)(u^3 - u)
    R_u = (1.0 / eps_c) * (u**3 - u)
    
    F_form = (
        (u - u_n) / dt_c * v * ufl.dx
        + eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + R_u * v * ufl.dx
        - f_ufl * v * ufl.dx
    )
    
    J_form = ufl.derivative(F_form, u)
    
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-9,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "ksp_rtol": 1e-9,
    }
    
    try:
        problem = petsc.NonlinearProblem(
            F_form, u, bcs=bcs, J=J_form,
            petsc_options_prefix=f"test{eps_val}_",
            petsc_options=petsc_options,
        )
        
        snes = problem.solver
        t = 0.0
        n_steps = int(round(0.2 / dt_val))
        converged_all = True
        
        for step in range(n_steps):
            t += dt_val
            u.x.array[:] = u_n.x.array[:]
            problem.solve()
            reason = snes.getConvergedReason()
            if reason < 0:
                converged_all = False
                break
            u_n.x.array[:] = u.x.array[:]
        
        print(f"eps={eps_val:.3f}: range=[{u.x.array.min():.6f}, {u.x.array.max():.6f}], converged={converged_all}")
    except Exception as e:
        print(f"eps={eps_val:.3f}: FAILED - {e}")
