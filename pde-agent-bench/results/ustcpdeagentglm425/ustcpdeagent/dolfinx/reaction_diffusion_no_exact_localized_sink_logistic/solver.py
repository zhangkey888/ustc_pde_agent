import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract parameters from case_spec
    pde = case_spec.get("pde", {})
    time_params = pde.get("time", {})
    
    epsilon = float(pde.get("epsilon", 1.0))
    rho = float(pde.get("reaction_rho", 1.0))
    
    t0 = float(time_params.get("t0", 0.0))
    t_end = float(time_params.get("t_end", 0.35))
    
    output = case_spec.get("output", {})
    grid = output.get("grid", {})
    nx_out = int(grid.get("nx", 64))
    ny_out = int(grid.get("ny", 64))
    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    # Solver configuration
    mesh_res = 256
    element_degree = 1
    dt_used = 0.001
    n_steps = int(round((t_end - t0) / dt_used))
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res,
                                      cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    x = ufl.SpatialCoordinate(domain)
    
    # Source term
    f_expr = (4.0 * ufl.exp(-200.0 * ((x[0] - 0.4)**2 + (x[1] - 0.6)**2))
              - 2.0 * ufl.exp(-200.0 * ((x[0] - 0.65)**2 + (x[1] - 0.35)**2)))
    f = fem.Function(V)
    f.interpolate(fem.Expression(f_expr, V.element.interpolation_points))
    
    # Initial condition
    u0_expr = 0.4 + 0.1 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    u_init = fem.Function(V)
    u_init.interpolate(fem.Expression(u0_expr, V.element.interpolation_points))
    
    u_old = fem.Function(V)
    u_old.x.array[:] = u_init.x.array[:]
    
    u_trial = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Boundary conditions - u = 0 on all boundaries
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt_used))
    
    # IMEX scheme: implicit diffusion + explicit reaction
    # (u - u_old)/dt - epsilon * Laplacian(u) = f - rho * u_old * (1 - u_old)
    a = (u_trial * v / dt_const
         + epsilon * ufl.inner(ufl.grad(u_trial), ufl.grad(v))) * ufl.dx
    a_form = fem.form(a)
    
    rhs_source = fem.Function(V)
    L = (u_old / dt_const * v + rhs_source * v) * ufl.dx
    L_form = fem.form(L)
    
    # Assemble LHS matrix (constant - done once)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    # CG solver with ILU preconditioner
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType('cg')
    ksp.getPC().setType('ilu')
    ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=300)
    
    u_sol = fem.Function(V)
    
    # Time stepping
    total_ksp_its = 0
    
    t = t0
    for step in range(n_steps):
        t += dt_used
        
        # Explicit reaction: f - rho * u_old * (1 - u_old)
        u_old_arr = u_old.x.array.copy()
        rhs_source.x.array[:] = f.x.array - rho * u_old_arr * (1.0 - u_old_arr)
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        ksp.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_ksp_its += int(ksp.getIterationNumber())
        
        # Update
        u_old.x.array[:] = u_sol.x.array[:]
        u_old.x.scatter_forward()
    
    # === Sample solution on output grid ===
    xmin, xmax, ymin, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.zeros((nx_out * ny_out, 3))
    points[:, 0] = XX.ravel()
    points[:, 1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    pts_proc = []
    cls_proc = []
    evl_map = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            pts_proc.append(points[i])
            cls_proc.append(links[0])
            evl_map.append(i)
    
    # Sample final solution
    u_vals = np.full((points.shape[0],), np.nan)
    if len(pts_proc) > 0:
        pts_arr = np.array(pts_proc)
        cls_arr = np.array(cls_proc, dtype=np.int32)
        u_vals[evl_map] = u_sol.eval(pts_arr, cls_arr).flatten()
    
    u_flat = u_vals.copy()
    u_flat[np.isnan(u_flat)] = 0.0
    u_grid = u_flat.reshape(ny_out, nx_out)
    
    if comm.size > 1:
        u_grid_g = np.zeros_like(u_grid)
        comm.Allreduce(u_grid, u_grid_g, op=MPI.SUM)
        u_grid = u_grid_g
    
    # Sample initial condition
    u0_vals = np.full((points.shape[0],), np.nan)
    if len(pts_proc) > 0:
        u0_vals[evl_map] = u_init.eval(pts_arr, cls_arr).flatten()
    
    u0_flat = u0_vals.copy()
    u0_flat[np.isnan(u0_flat)] = 0.0
    u_initial = u0_flat.reshape(ny_out, nx_out)
    
    if comm.size > 1:
        u0_g = np.zeros_like(u_initial)
        comm.Allreduce(u_initial, u0_g, op=MPI.SUM)
        u_initial = u0_g
    
    # Build solver info
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": element_degree,
        "ksp_type": "cg",
        "pc_type": "ilu",
        "rtol": 1e-10,
        "iterations": total_ksp_its,
        "dt": dt_used,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
        "nonlinear_iterations": [1] * n_steps,
    }
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": solver_info,
    }
