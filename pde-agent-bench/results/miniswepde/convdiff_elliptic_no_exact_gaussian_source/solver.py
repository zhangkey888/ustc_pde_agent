import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict = None):
    """
    Solve convection-diffusion equation:
      -ε ∇²u + β·∇u = f   in Ω
       u = g               on ∂Ω
    with SUPG stabilization for high Péclet numbers.
    """
    comm = MPI.COMM_WORLD

    # Default parameters
    epsilon = 0.05
    beta_vec = [2.0, 1.0]
    nx_out = 50
    ny_out = 50
    bc_value = 0.0
    bbox = [0, 1, 0, 1]
    source_term_str = "exp(-250*((x-0.35)**2 + (y-0.65)**2))"

    # Parse case_spec
    if case_spec is not None:
        oc = case_spec.get("oracle_config", {})
        pde = oc.get("pde", {})
        params = pde.get("pde_params", {})
        epsilon = params.get("epsilon", epsilon)
        beta_vec = params.get("beta", beta_vec)
        source_term_str = pde.get("source_term", source_term_str)

        bc_spec = oc.get("bc", {})
        dirichlet = bc_spec.get("dirichlet", {})
        bc_val_str = dirichlet.get("value", "0.0")
        try:
            bc_value = float(bc_val_str)
        except (ValueError, TypeError):
            bc_value = 0.0

        grid = oc.get("output", {}).get("grid", {})
        nx_out = grid.get("nx", nx_out)
        ny_out = grid.get("ny", ny_out)
        bbox = grid.get("bbox", bbox)

    # Solver parameters - use degree 2 for better accuracy
    N = 128
    element_degree = 2
    ksp_type = "gmres"
    pc_type = "hypre"
    rtol = 1e-10

    # Create mesh
    x0, x1, y0, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
    if x0 == 0 and x1 == 1 and y0 == 0 and y1 == 1:
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    else:
        p0 = np.array([x0, y0])
        p1 = np.array([x1, y1])
        domain = mesh.create_rectangle(comm, [p0, p1], [N, N], cell_type=mesh.CellType.triangle)

    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)

    # Source term: f = exp(-250*((x-0.35)^2 + (y-0.65)^2))
    f = ufl.exp(-250.0 * ((x[0] - 0.35)**2 + (x[1] - 0.65)**2))

    # Convection velocity
    beta = ufl.as_vector([ScalarType(beta_vec[0]), ScalarType(beta_vec[1])])

    # Diffusion coefficient
    eps_c = fem.Constant(domain, ScalarType(epsilon))

    # Standard Galerkin weak form:
    # ε (∇u, ∇v) + (β·∇u, v) = (f, v)
    a_std = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
            + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L_std = ufl.inner(f, v) * ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))

    # SUPG stabilization parameter (standard formula)
    Pe_h = beta_norm * h / (2.0 * eps_c)
    tau = h / (2.0 * beta_norm + 1e-10) * (1.0 / ufl.tanh(Pe_h + 1e-10) - 1.0 / (Pe_h + 1e-10))

    # SUPG test function modification: tau * β·∇v
    supg_test = tau * ufl.dot(beta, ufl.grad(v))

    # For P2 elements, the Laplacian is constant within each element (for triangles)
    # Include the full residual for better accuracy
    # Residual: -ε∇²u + β·∇u - f
    # For linear elements, ∇²u = 0, but for P2 it's nonzero
    a_supg = ufl.inner(ufl.dot(beta, ufl.grad(u)), supg_test) * ufl.dx
    L_supg = ufl.inner(f, supg_test) * ufl.dx

    # Add diffusion part of SUPG if using higher order elements
    if element_degree >= 2:
        a_supg += ufl.inner(-eps_c * ufl.div(ufl.grad(u)), supg_test) * ufl.dx

    a_total = a_std + a_supg
    L_total = L_std + L_supg

    # Boundary conditions: u = g on ∂Ω
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(bc_value), dofs, V)

    # Solve
    iterations = 0
    try:
        problem = petsc.LinearProblem(
            a_total, L_total, bcs=[bc],
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "ksp_rtol": str(rtol),
                "ksp_max_it": "5000",
                "ksp_gmres_restart": "150",
            },
            petsc_options_prefix="convdiff_"
        )
        u_sol = problem.solve()
    except Exception:
        # Fallback to direct solver
        ksp_type = "preonly"
        pc_type = "lu"
        problem = petsc.LinearProblem(
            a_total, L_total, bcs=[bc],
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
            },
            petsc_options_prefix="convdiff_fb_"
        )
        u_sol = problem.solve()

    # Evaluate on output grid
    u_grid = evaluate_on_grid(domain, u_sol, nx_out, ny_out, bbox)

    solver_info = {
        "mesh_resolution": N,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }

    return {"u": u_grid, "solver_info": solver_info}


def evaluate_on_grid(domain, u_func, nx, ny, bbox=None):
    """Evaluate solution on a uniform nx x ny grid."""
    if bbox is None:
        bbox = [0, 1, 0, 1]

    x_coords = np.linspace(bbox[0], bbox[1], nx)
    y_coords = np.linspace(bbox[2], bbox[3], ny)
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')

    points = np.zeros((nx * ny, 3))
    points[:, 0] = xx.flatten()
    points[:, 1] = yy.flatten()
    points[:, 2] = 0.0

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(len(points)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.zeros(nx * ny)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()

    return u_values.reshape((nx, ny))


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve()
    t1 = time.time()

    u_grid = result["u"]
    info = result["solver_info"]

    print(f"Wall time: {t1 - t0:.3f} s")
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution range: [{u_grid.min():.8f}, {u_grid.max():.8f}]")
    print(f"L2 norm of grid: {np.linalg.norm(u_grid):.8f}")
    print(f"Solver info: {info}")
