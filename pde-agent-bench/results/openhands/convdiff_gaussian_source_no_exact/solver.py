"""
Solver for steady convection-diffusion equation using dolfinx 0.10.0
Problem: -ε ∇²u + β·∇u = f in Ω, u = 0 on ∂Ω
Case ID: convdiff_gaussian_source_no_exact
SUPG stabilization for high Péclet number.
"""

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    # ---- Parameters from case_spec ----
    pde = case_spec.get("pde", {})
    params = pde.get("params", {})
    epsilon = params.get("epsilon", 0.02)
    beta_list = params.get("beta", [8.0, 3.0])

    # ---- Agent-selected parameters ----
    mesh_resolution = 128
    element_degree = 1
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10

    # ---- Create mesh and function space ----
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution,
                                     cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    xc = ufl.SpatialCoordinate(domain)

    # ---- Convection velocity ----
    beta = ufl.as_vector([beta_list[0], beta_list[1]])
    beta_mag = ufl.sqrt(beta_list[0]**2 + beta_list[1]**2)

    # ---- Source term ----
    f_ufl = ufl.exp(-250.0 * ((xc[0] - 0.3)**2 + (xc[1] - 0.7)**2))

    # ---- Trial / test functions ----
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # ---- Boundary conditions (homogeneous Dirichlet on all boundaries) ----
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim,
        lambda x: np.logical_or.reduce([
            np.isclose(x[0], 0.0), np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0), np.isclose(x[1], 1.0),
        ])
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)

    # ---- SUPG stabilization parameter ----
    # For P1 elements, ∇²u_h = 0, so residual R(u_h) = β·∇u_h - f
    h = ufl.CellDiameter(domain)
    Pe_local = beta_mag * h / (2.0 * epsilon)
    # Optimal SUPG parameter: τ = h/(2|β|) * (coth(Pe) - 1/Pe)
    # For high Pe, coth(Pe) ≈ 1, so τ ≈ h/(2|β|)
    # Use a smooth approximation: τ = h/(2|β|) * min(1, Pe/3)
    # But since Pe >> 1 here, simplify to:
    tau = h / (2.0 * beta_mag)

    # ---- Galerkin terms ----
    a_gal = (epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
             + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx)
    L_gal = ufl.inner(f_ufl, v) * ufl.dx

    # ---- SUPG stabilization terms ----
    # Test function modification: v_supg = τ * (β·∇v)
    # For P1: residual = β·∇u - f (Laplacian of P1 is zero)
    r_u = ufl.dot(beta, ufl.grad(u))  # residual operator applied to u (without -f)
    v_supg = tau * ufl.dot(beta, ufl.grad(v))

    a_supg = ufl.inner(r_u, v_supg) * ufl.dx
    L_supg = ufl.inner(f_ufl, v_supg) * ufl.dx

    # ---- Combined forms ----
    a_form = a_gal + a_supg
    L_form = L_gal + L_supg

    # ---- Solve ----
    problem = petsc.LinearProblem(
        a_form, L_form, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
        },
        petsc_options_prefix="convdiff_"
    )
    u_sol = problem.solve()

    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()

    # ---- Sample solution on 50×50 grid ----
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    points_3d = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx_out * ny_out)])

    u_grid = _probe(u_sol, points_3d, domain).reshape(nx_out, ny_out)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
        "stabilization": "SUPG",
    }

    return {"u": u_grid, "solver_info": solver_info}


def _probe(u_func, points_array, domain):
    """Evaluate u_func at points_array (shape (3, N)). Returns (N,) array."""
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

    values = np.full(points_array.shape[1], np.nan)
    if pts:
        vals = u_func.eval(np.array(pts), np.array(cells, dtype=np.int32))
        values[idx_map] = vals.flatten()

    comm = domain.comm
    if comm.size > 1:
        all_vals = comm.gather(values, root=0)
        if comm.rank == 0:
            combined = np.full_like(values, np.nan)
            for pv in all_vals:
                mask = ~np.isnan(pv)
                combined[mask] = pv[mask]
            values = combined
        values = comm.bcast(values, root=0)

    return values