"""
Solver for steady incompressible Navier-Stokes using dolfinx 0.10.0
  u·∇u - ν∇²u + ∇p = f   in Ω=[0,1]²
  ∇·u = 0                 in Ω
  u = g                   on ∂Ω
Case: navier_stokes_low_nu_small_amplitude
Manufactured: u = [0.2π cos(πy) sin(2πx), -0.4π cos(2πx) sin(πy)], p = 0
ν = 0.01, Taylor-Hood Q3/Q2 on quadrilaterals, Newton with ν-continuation + MUMPS.
"""

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, nls
from dolfinx.fem import petsc
from basix.ufl import element, mixed_element
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    params = pde.get("params", {})
    nu_val = float(params.get("viscosity", 0.01))

    N = 48
    degree_u = 3
    degree_p = 2
    newton_rtol = 1e-10
    newton_atol = 1e-12
    ksp_type = "preonly"
    pc_type = "lu"

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.quadrilateral)
    tdim = domain.topology.dim
    fdim = tdim - 1

    cell_name = domain.topology.cell_name()
    vel_el = element("Lagrange", cell_name, degree_u, shape=(2,))
    pres_el = element("Lagrange", cell_name, degree_p)
    mel = mixed_element([vel_el, pres_el])
    W = fem.functionspace(domain, mel)

    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi

    u_exact = ufl.as_vector([
        0.2 * pi * ufl.cos(pi * x[1]) * ufl.sin(2.0 * pi * x[0]),
        -0.4 * pi * ufl.cos(2.0 * pi * x[0]) * ufl.sin(pi * x[1])
    ])
    # p_exact = 0

    # Use a fem.Constant for ν so we can do continuation
    nu = fem.Constant(domain, ScalarType(nu_val))

    # Source term: f = u_exact·∇u_exact - ν∇²u_exact (+ ∇p_exact = 0)
    # Note: source term also depends on nu via the diffusion of the manufactured solution
    f_source = ufl.grad(u_exact) * u_exact - nu * ufl.div(ufl.grad(u_exact))

    # Weak form residual
    F = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + q * ufl.div(u) * ufl.dx
         - ufl.inner(f_source, v) * ufl.dx)

    # Dirichlet BCs: u = u_exact on entire boundary
    V_col, col_map_u = W.sub(0).collapse()
    u_bc_func = fem.Function(V_col)
    u_bc_func.interpolate(lambda x: np.vstack([
        0.2 * np.pi * np.cos(np.pi * x[1]) * np.sin(2.0 * np.pi * x[0]),
        -0.4 * np.pi * np.cos(2.0 * np.pi * x[0]) * np.sin(np.pi * x[1])
    ]))

    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs_u = fem.locate_dofs_topological((W.sub(0), V_col), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    # Pin one pressure DOF at origin to fix gauge (p_exact = 0 everywhere)
    Q_col, col_map_p = W.sub(1).collapse()
    p_bc_func = fem.Function(Q_col)
    p_bc_func.interpolate(lambda x: np.zeros_like(x[0]))

    def at_origin(x):
        tol = 1e-10
        return np.logical_and(np.abs(x[0]) < tol, np.abs(x[1]) < tol)

    dofs_p = fem.locate_dofs_geometrical((W.sub(1), Q_col), at_origin)
    bc_p = fem.dirichletbc(p_bc_func, dofs_p, W.sub(1))

    bcs = [bc_u, bc_p]

    w.x.array[:] = 0.0
    w.x.scatter_forward()

    # Newton solver
    problem = petsc.NonlinearProblem(F, w, bcs=bcs)
    solver = nls.petsc.NewtonSolver(comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = newton_rtol
    solver.atol = newton_atol
    solver.max_it = 50

    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")

    # Viscosity continuation for robustness: high ν → target ν
    total_newton = 0
    nu_stages = []
    if nu_val < 0.1:
        nu_stages = [1.0, 0.1, nu_val]
    elif nu_val < 1.0:
        nu_stages = [1.0, nu_val]
    else:
        nu_stages = [nu_val]

    for nu_step in nu_stages:
        nu.value = nu_step
        n_newton, converged = solver.solve(w)
        assert converged, f"Newton did not converge at nu={nu_step} after {n_newton} iters"
        w.x.scatter_forward()
        total_newton += n_newton

    # Extract velocity into collapsed space for evaluation
    u_sol = fem.Function(V_col)
    u_sol.x.array[:] = w.x.array[col_map_u]
    u_sol.x.scatter_forward()

    # Sample velocity magnitude on 50×50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    points_3d = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx_out * ny_out)])

    vel_values = _probe_vector(u_sol, points_3d, domain)
    vel_mag = np.sqrt(vel_values[:, 0]**2 + vel_values[:, 1]**2)
    u_grid = vel_mag.reshape(nx_out, ny_out)

    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": newton_rtol,
        "iterations": total_newton,
        "nonlinear_iterations": [int(total_newton)],
    }

    return {"u": u_grid, "solver_info": solver_info}


def _probe_vector(u_func, points_array, domain):
    """Evaluate vector function u_func at points (shape (3, N)). Returns (N, dim) array."""
    gdim = domain.geometry.dim
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_array.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_array.T)

    pts, cells, idx_map = [], [], []
    for i in range(points_array.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            pts.append(points_array.T[i])
            cells.append(links[0])
            idx_map.append(i)

    n_pts = points_array.shape[1]
    values = np.full((n_pts, gdim), np.nan)
    if pts:
        vals = u_func.eval(np.array(pts), np.array(cells, dtype=np.int32))
        values[idx_map] = vals[:, :gdim]

    comm = domain.comm
    if comm.size > 1:
        all_vals = comm.gather(values, root=0)
        if comm.rank == 0:
            combined = np.full_like(values, np.nan)
            for pv in all_vals:
                mask = ~np.isnan(pv[:, 0])
                combined[mask] = pv[mask]
            values = combined
        values = comm.bcast(values, root=0)

    return values