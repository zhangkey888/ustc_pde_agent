import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parse output grid spec
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]  # [xmin, xmax, ymin, ymax]
    
    # Mesh resolution and element degree
    N = 300
    deg = 2
    rtol = 1e-10
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", deg))
    
    # Define coefficient kappa as UFL expression
    x = ufl.SpatialCoordinate(domain)
    kappa = 1.0 + 1000.0 * ufl.exp(-100.0 * (x[0] - 0.5)**2)
    
    # Variational form: -div(kappa * grad(u)) = f
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(domain, PETSc.ScalarType(1.0))
    
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # Boundary conditions: u = 0 on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    
    # Solve with CG + BoomerAMG
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
            "ksp_max_it": 500,
        },
        petsc_options_prefix="poisson_"
    )
    
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    iterations = problem.solver.getIterationNumber()
    
    # ---- Accuracy verification: compute residual norm ----
    residual = petsc.create_vector(fem.form(L).function_spaces)
    with residual.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(residual, fem.form(L))
    a_form = fem.form(a)
    petsc.apply_lifting(residual, [a_form], bcs=[[bc]])
    residual.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(residual, [bc])
    
    # Compute ||Au - b||
    Au = petsc.create_vector(fem.form(L).function_spaces)
    problem.A.mult(u_sol.x.petsc_vec, Au)
    residual.axpy(-1.0, Au)
    res_norm = residual.norm(PETSc.NormType.NORM_2)
    
    sol_norm = u_sol.x.petsc_vec.norm(PETSc.NormType.NORM_2)
    rel_res = res_norm / sol_norm if sol_norm > 0 else res_norm
    
    if comm.rank == 0:
        print(f"[Verification] Residual norm ||Au-b||: {res_norm:.6e}")
        print(f"[Verification] Relative residual: {rel_res:.6e}")
        print(f"[Verification] CG iterations: {iterations}")
    
    # Sample solution on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.zeros((nx_out * ny_out, 3))
    points[:, 0] = XX.ravel()
    points[:, 1] = YY.ravel()
    points[:, 2] = 0.0
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.zeros(points.shape[0])
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_values_global = np.zeros_like(u_values)
    comm.Allreduce(u_values, u_values_global, op=MPI.SUM)
    
    u_grid = u_values_global.reshape(ny_out, nx_out)
    
    # Build solver_info
    solver_info = {
        "mesh_resolution": N,
        "element_degree": deg,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": rtol,
        "iterations": int(iterations),
    }
    
    # Check if time-dependent info needed
    pde_spec = case_spec.get("pde", {})
    if "time" in pde_spec:
        solver_info["dt"] = 0.0
        solver_info["n_steps"] = 0
        solver_info["time_scheme"] = "none"
    
    result = {
        "u": u_grid,
        "solver_info": solver_info,
    }
    
    return result
