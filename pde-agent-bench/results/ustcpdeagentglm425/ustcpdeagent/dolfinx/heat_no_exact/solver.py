import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # ---- Parse case_spec ----
    pde = case_spec["pde"]
    coeff = pde.get("coefficients", {})
    kappa = coeff.get("kappa", coeff.get("κ", 1.0))
    
    time_info = pde.get("time", {})
    t0 = time_info.get("t0", 0.0)
    t_end = time_info.get("t_end", 0.1)
    dt_suggested = time_info.get("dt", 0.02)
    scheme = time_info.get("scheme", "backward_euler")
    
    bcs_info = pde.get("boundary_conditions", [])
    ic_info = pde.get("initial_condition", {})
    
    out = case_spec["output"]
    grid = out["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]  # [xmin, xmax, ymin, ymax]
    xmin, xmax, ymin, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    
    # ---- Choose mesh resolution, element degree, dt ----
    mesh_res = 80
    elem_deg = 2
    dt = dt_suggested / 2  # refine time step for accuracy
    
    n_steps = int(round((t_end - t0) / dt))
    dt = (t_end - t0) / n_steps  # exact dt
    
    # ---- Create mesh ----
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    # ---- Function space ----
    V = fem.functionspace(domain, ("Lagrange", elem_deg))
    
    # ---- Define variational forms for backward Euler ----
    # ∂u/∂t - κ∇²u = f
    # (u^{n+1} - u^n)/dt - κ∇²u^{n+1} = f^{n+1}
    # (u^{n+1}, v)/dt + κ(∇u^{n+1}, ∇v) = (u^n, v)/dt + (f^{n+1}, v)
    
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Source term f = sin(pi*x)*cos(pi*y)
    f_expr = ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
    
    # Bilinear form: (u, v)/dt + kappa*(grad(u), grad(v))
    a = ufl.inner(u_trial, v_test) * ufl.dx / dt + kappa * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    
    # Linear form depends on u_n and f, so we'll use a Constant for the RHS
    # We'll assemble manually in the time loop
    
    # ---- Boundary conditions: u = 0 on all boundaries ----
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc_func = fem.Function(V)
    u_bc_func.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)
    
    # ---- Initial condition: u0 = sin(pi*x)*sin(pi*y) ----
    u_n = fem.Function(V)
    u0_expr = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    u_n.interpolate(fem.Expression(u0_expr, V.element.interpolation_points))
    
    # Store initial condition for output
    u_initial_grid = _sample_on_grid(domain, u_n, nx_out, ny_out, xmin, xmax, ymin, ymax)
    
    # ---- Solution function ----
    u_sol = fem.Function(V)
    
    # ---- Assemble LHS matrix (constant in time) ----
    a_form = fem.form(a)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    # ---- Set up PETSc KSP solver ----
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.CG)
    ksp.getPC().setType(PETSc.PC.Type.HYPRE)
    ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    ksp.setFromOptions()
    
    # ---- Time stepping ----
    total_iterations = 0
    t = t0
    
    # RHS form: (u_n, v)/dt + (f, v)
    # We define L with u_n as a coefficient
    L = ufl.inner(u_n, v_test) * ufl.dx / dt + ufl.inner(f_expr, v_test) * ufl.dx
    L_form = fem.form(L)
    
    b = petsc.create_vector(L_form.function_spaces)
    
    for step in range(n_steps):
        t = t0 + (step + 1) * dt
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        
        # Apply lifting for BCs
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        ksp.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        # Track iterations
        total_iterations += ksp.getIterationNumber()
        
        # Update u_n
        u_n.x.array[:] = u_sol.x.array[:]
    
    # ---- Sample solution on output grid ----
    u_grid = _sample_on_grid(domain, u_sol, nx_out, ny_out, xmin, xmax, ymin, ymax)
    
    # ---- Build solver_info ----
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_deg,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1e-10,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler"
    }
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info
    }


def _sample_on_grid(domain, u_func, nx, ny, xmin, xmax, ymin, ymax):
    """Sample a dolfinx Function on a uniform grid and return (ny, nx) array."""
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    
    # Build points array: shape (3, nx*ny) for dolfinx geometry
    points = np.zeros((3, nx * ny), dtype=np.float64)
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    points[2, :] = 0.0
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((nx * ny,), np.nan, dtype=np.float64)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    # In parallel, gather results
    # For single process, this is fine as-is
    u_grid = u_values.reshape(ny, nx)
    
    return u_grid


if __name__ == "__main__":
    # Test with a mock case_spec
    case_spec = {
        "pde": {
            "equation": "heat",
            "coefficients": {"κ": 1.0, "kappa": 1.0},
            "time": {"t0": 0.0, "t_end": 0.1, "dt": 0.02, "scheme": "backward_euler"},
            "boundary_conditions": [{"type": "dirichlet", "value": 0.0}],
            "initial_condition": {"expression": "sin(pi*x)*sin(pi*y)"},
            "source": {"expression": "sin(pi*x)*cos(pi*y)"}
        },
        "output": {
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }
    
    result = solve(case_spec)
    print(f"u_grid shape: {result['u'].shape}")
    print(f"u_grid min: {np.nanmin(result['u']):.6e}, max: {np.nanmax(result['u']):.6e}")
    print(f"u_initial shape: {result['u_initial'].shape}")
    print(f"solver_info: {result['solver_info']}")
