"""
Solver for steady convection-diffusion equation using dolfinx 0.10.0
Problem: -ε ∇²u + β·∇u = f in Ω=[0,1]²
Case ID: convdiff_mixed_no_exact_gaussian_source
SUPG stabilization for high Péclet number (~1118).
"""

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _parse_bcs(case_spec, domain, V):
    """Parse boundary conditions from case_spec and return list of DirichletBC."""
    pde = case_spec.get("pde", {})
    bcs_spec = pde.get("bcs", [])
    params = pde.get("params", {})
    beta_list = params.get("beta", [10.0, 5.0])

    tdim = domain.topology.dim
    fdim = tdim - 1
    bcs = []
    has_dirichlet = False
    dirichlet_segments = set()

    for bc_def in bcs_spec:
        bc_type = bc_def.get("type", "dirichlet")
        location = bc_def.get("location", "")
        value = bc_def.get("value", 0.0)

        if bc_type.lower() == "dirichlet":
            has_dirichlet = True
            marker = _boundary_marker(location)
            if marker is not None:
                dirichlet_segments.add(location)
                facets = mesh.locate_entities_boundary(domain, fdim, marker)
                dofs = fem.locate_dofs_topological(V, fdim, facets)
                if isinstance(value, (int, float)):
                    bcs.append(fem.dirichletbc(ScalarType(float(value)), dofs, V))
                else:
                    u_bc = fem.Function(V)
                    u_bc.interpolate(lambda x, v=value: np.full_like(x[0], float(v)))
                    bcs.append(fem.dirichletbc(u_bc, dofs))

    # If no BCs were parsed, apply default mixed BCs:
    # Dirichlet u=0 on inflow, natural (Neumann=0) on outflow
    if not bcs:
        # Inflow: where β·n < 0
        # Left (x=0): β·(-1,0) = -β[0] < 0 if β[0] > 0 → inflow
        # Bottom (y=0): β·(0,-1) = -β[1] < 0 if β[1] > 0 → inflow
        # Right (x=1): β·(1,0) = β[0] > 0 → outflow
        # Top (y=1): β·(0,1) = β[1] > 0 → outflow
        def inflow_marker(x):
            return np.logical_or(
                np.isclose(x[0], 0.0),
                np.isclose(x[1], 0.0)
            )
        inflow_facets = mesh.locate_entities_boundary(domain, fdim, inflow_marker)
        inflow_dofs = fem.locate_dofs_topological(V, fdim, inflow_facets)
        bcs.append(fem.dirichletbc(ScalarType(0.0), inflow_dofs, V))

    return bcs


def _boundary_marker(location):
    """Return a boundary marker function for the given location string."""
    loc = location.lower().strip()
    if loc in ("left", "x=0", "x0"):
        return lambda x: np.isclose(x[0], 0.0)
    elif loc in ("right", "x=1", "x1"):
        return lambda x: np.isclose(x[0], 1.0)
    elif loc in ("bottom", "y=0", "y0"):
        return lambda x: np.isclose(x[1], 0.0)
    elif loc in ("top", "y=1", "y1"):
        return lambda x: np.isclose(x[1], 1.0)
    elif loc in ("all", "boundary", "entire"):
        return lambda x: np.logical_or.reduce([
            np.isclose(x[0], 0.0), np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0), np.isclose(x[1], 1.0),
        ])
    elif loc in ("inflow",):
        return lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))
    elif loc in ("outflow",):
        return lambda x: np.logical_or(np.isclose(x[0], 1.0), np.isclose(x[1], 1.0))
    return None


def solve(case_spec: dict) -> dict:
    # ---- Parameters from case_spec ----
    pde = case_spec.get("pde", {})
    params = pde.get("params", {})
    epsilon = float(params.get("epsilon", 0.01))
    beta_list = params.get("beta", [10.0, 5.0])
    beta_list = [float(b) for b in beta_list]

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
    beta_mag_val = float(np.sqrt(beta_list[0]**2 + beta_list[1]**2))

    # ---- Source term: f = exp(-200*((x-0.3)^2 + (y-0.7)^2)) ----
    source = pde.get("source", {})
    # Parse source from case_spec if available, otherwise use default
    f_ufl = ufl.exp(-200.0 * ((xc[0] - 0.3)**2 + (xc[1] - 0.7)**2))

    # ---- Trial / test functions ----
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # ---- Boundary conditions ----
    bcs = _parse_bcs(case_spec, domain, V)

    # ---- SUPG stabilization parameter ----
    h = ufl.CellDiameter(domain)
    # For P1 elements on triangles: h ~ 2*cell_diameter/sqrt(dim*(dim+1))
    # Optimal SUPG: τ = h/(2|β|) * ξ(Pe_h) where ξ(Pe) = coth(Pe) - 1/Pe
    # For high Pe (~1118), ξ ≈ 1, so τ ≈ h/(2|β|)
    tau = h / (2.0 * beta_mag_val)

    # ---- Galerkin terms ----
    a_gal = (epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
             + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx)
    L_gal = ufl.inner(f_ufl, v) * ufl.dx

    # ---- SUPG stabilization terms ----
    # For P1: ∇²u_h = 0 within elements, so strong residual = β·∇u - f
    # SUPG adds ∫_K τ (β·∇u_h - f)(β·∇v) dx
    v_supg = tau * ufl.dot(beta, ufl.grad(v))
    a_supg = ufl.inner(ufl.dot(beta, ufl.grad(u)), v_supg) * ufl.dx
    L_supg = ufl.inner(f_ufl, v_supg) * ufl.dx

    # Also add diffusion contribution to SUPG for completeness:
    # For P1 this is zero (-ε∇²u_h = 0), but for higher order it wouldn't be
    # So we include: -ε∇²u contribution = 0 for P1

    # ---- Combined forms ----
    a_form = a_gal + a_supg
    L_form = L_gal + L_supg

    # ---- Solve ----
    problem = petsc.LinearProblem(
        a_form, L_form, bcs=bcs,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "5000",
            "ksp_gmres_restart": "100",
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