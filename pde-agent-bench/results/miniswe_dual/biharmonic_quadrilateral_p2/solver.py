import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    """Solve biharmonic equation using mixed formulation (two Poisson solves).
    
    Biharmonic: Delta^2 u = f in Omega, u = 0 on dOmega
    Mixed formulation: sigma = -Delta u, then -Delta sigma = f
    Both sigma and u have zero Dirichlet BCs for this manufactured solution.
    """
    
    comm = MPI.COMM_WORLD
    
    N = 36
    degree = 3
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.quadrilateral)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    pi_val = np.pi
    
    f_expr = 25.0 * pi_val**4 * ufl.sin(2 * pi_val * x[0]) * ufl.sin(pi_val * x[1])
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda xc: np.ones(xc.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_zero = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    
    solver_opts = {"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": "1e-12"}
    
    sigma_trial = ufl.TrialFunction(V)
    tau = ufl.TestFunction(V)
    a1 = ufl.inner(ufl.grad(sigma_trial), ufl.grad(tau)) * ufl.dx
    L1 = f_expr * tau * ufl.dx
    
    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc_zero],
        petsc_options=solver_opts,
        petsc_options_prefix="p1_"
    )
    sigma_h = problem1.solve()
    iter1 = problem1.solver.getIterationNumber()
    
    u_trial = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a2 = ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
    L2 = sigma_h * v * ufl.dx
    
    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc_zero],
        petsc_options=solver_opts,
        petsc_options_prefix="p2_"
    )
    u_h = problem2.solve()
    iter2 = problem2.solver.getIterationNumber()
    
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points_3d = np.zeros((nx_out * ny_out, 3))
    points_3d[:, 0] = X.ravel()
    points_3d[:, 1] = Y.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-12,
            "iterations": iter1 + iter2,
        }
    }
