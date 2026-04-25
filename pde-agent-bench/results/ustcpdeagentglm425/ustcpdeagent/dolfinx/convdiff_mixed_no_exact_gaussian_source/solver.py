import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # --- Extract problem parameters ---
    pde = case_spec["pde"]
    eps = pde["diffusion"]       # 0.01
    beta = pde["velocity"]       # [10.0, 5.0]
    
    out = case_spec["output"]
    grid_info = out["grid"]
    nx_out = grid_info["nx"]
    ny_out = grid_info["ny"]
    bbox = grid_info["bbox"]  # [xmin, xmax, ymin, ymax]
    
    # --- Create mesh ---
    N = 256  # mesh resolution
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # --- Function space ---
    deg = 2  # polynomial degree
    V = fem.functionspace(domain, ("Lagrange", deg))
    
    # --- Boundary conditions: u = 0 on all boundaries ---
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # --- Define variational form with SUPG stabilization ---
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Source term f
    f = ufl.exp(-200.0 * ((x[0] - 0.3)**2 + (x[1] - 0.7)**2))
    
    # Convection velocity as UFL vector
    beta_ufl = ufl.as_vector([beta[0], beta[1]])
    
    # Mesh element size (for SUPG)
    h = ufl.CellDiameter(domain)
    
    # Local Peclet number
    beta_norm = ufl.sqrt(ufl.inner(beta_ufl, beta_ufl))
    Pe_loc = beta_norm * h / (2.0 * eps)
    
    # SUPG stabilization parameter (Brooks-Hughes formula)
    tau_high = h / (2.0 * beta_norm + 1e-30)
    tau_low = h**2 / (12.0 * eps + 1e-30)
    tau_supg = ufl.conditional(Pe_loc > 3.0, tau_high, tau_low)
    
    # SUPG test function: v + tau * beta . grad(v)
    v_supg = v + tau_supg * ufl.inner(beta_ufl, ufl.grad(v))
    
    # Bilinear form: eps * grad(u) . grad(v_supg) + beta . grad(u) * v_supg
    a = eps * ufl.inner(ufl.grad(u), ufl.grad(v_supg)) * ufl.dx \
      + ufl.inner(ufl.dot(beta_ufl, ufl.grad(u)), v_supg) * ufl.dx
    
    # Linear form: f * v_supg
    L = f * v_supg * ufl.dx
    
    # --- Solve ---
    rtol = 1e-10
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "ksp_rtol": rtol,
            "ksp_atol": 1e-12,
            "ksp_max_it": 2000,
        },
        petsc_options_prefix="convdiff_"
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    # --- Get solver info ---
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # --- Sample solution on output grid ---
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((3, nx_out * ny_out), dtype=np.float64)
    pts[0] = XX.ravel()
    pts[1] = YY.ravel()
    pts[2] = 0.0
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.zeros((pts.shape[1],), dtype=ScalarType)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(
            np.array(points_on_proc),
            np.array(cells_on_proc, dtype=np.int32)
        )
        u_values[eval_map] = vals.flatten()
    
    # Communicate results across processes
    u_values_global = np.zeros_like(u_values)
    comm.Allreduce(u_values, u_values_global, op=MPI.SUM)
    
    u_grid = u_values_global.reshape(ny_out, nx_out)
    
    # --- Build solver_info ---
    solver_info = {
        "mesh_resolution": N,
        "element_degree": deg,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": rtol,
        "iterations": int(iterations),
    }
    
    # Check for time info
    if "time" in pde:
        solver_info["dt"] = 0.0
        solver_info["n_steps"] = 0
        solver_info["time_scheme"] = "none"
    
    return {"u": u_grid, "solver_info": solver_info}
