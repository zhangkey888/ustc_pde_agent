"""
Solver for steady convection-diffusion equation using dolfinx 0.10.0
Problem: -ε ∇²u + β·∇u = f in Ω=[0,1]²
Case ID: convdiff_mixed_p3_supg
Manufactured solution: u = sin(πx)sin(2πy)
SUPG stabilization for high Péclet number (~1265), P3 elements.
"""

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    params = pde.get("params", {})
    epsilon = float(params.get("epsilon", 0.01))
    beta_list = params.get("beta", [12.0, 4.0])
    beta_list = [float(b) for b in beta_list]

    # Agent-selected parameters
    N = 80
    p = 3
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", p))

    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi

    # Convection velocity
    beta = ufl.as_vector([beta_list[0], beta_list[1]])
    beta_mag = float(np.sqrt(beta_list[0]**2 + beta_list[1]**2))

    # Manufactured solution and derived source term
    u_exact = ufl.sin(pi * x[0]) * ufl.sin(2 * pi * x[1])
    f_ufl = -epsilon * ufl.div(ufl.grad(u_exact)) + ufl.dot(beta, ufl.grad(u_exact))

    # Trial / test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Boundary conditions: u = sin(πx)sin(2πy) = 0 on all boundaries of [0,1]²
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim,
        lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)

    # SUPG stabilization parameter
    h = ufl.CellDiameter(domain)
    # For high-order elements, scale tau by 1/p
    Pe_cell = beta_mag * h / (2.0 * epsilon * p)
    # Double asymptotic: xi ≈ min(Pe/3, 1-1/Pe)
    xi = ufl.conditional(ufl.gt(Pe_cell, 1.0),
                         1.0 - 1.0 / Pe_cell,
                         Pe_cell / 3.0)
    tau = h / (2.0 * beta_mag * p) * xi

    # Galerkin weak form
    a_gal = (epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
             + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx)
    L_gal = ufl.inner(f_ufl, v) * ufl.dx

    # SUPG stabilization: full residual including -ε∇²u for P3
    supg_test = tau * ufl.dot(beta, ufl.grad(v))
    strong_op_u = -epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    a_supg = ufl.inner(strong_op_u, supg_test) * ufl.dx
    L_supg = ufl.inner(f_ufl, supg_test) * ufl.dx

    a_form = a_gal + a_supg
    L_form = L_gal + L_supg

    # Solve
    problem = petsc.LinearProblem(
        a_form, L_form, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "10000",
            "ksp_gmres_restart": "150",
        },
        petsc_options_prefix="convdiff_"
    )
    u_sol = problem.solve()

    ksp = problem.solver
    iterations = ksp.getIterationNumber()

    # Sample solution on 50×50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    points_3d = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx_out * ny_out)])

    u_grid = _probe(u_sol, points_3d, domain).reshape(nx_out, ny_out)

    solver_info = {
        "mesh_resolution": N,
        "element_degree": p,
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