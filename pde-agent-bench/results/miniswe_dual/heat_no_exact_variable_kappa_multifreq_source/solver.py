import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve the transient heat equation with variable kappa."""
    
    # ---- Extract parameters from case_spec ----
    pde = case_spec.get("pde", {})
    time_params = pde.get("time", {})
    
    # Hardcoded defaults as fallback
    t_end = float(time_params.get("t_end", 0.1))
    dt_suggested = float(time_params.get("dt", 0.02))
    scheme = time_params.get("scheme", "backward_euler")
    
    # Agent-selectable parameters
    mesh_resolution = 64
    element_degree = 1
    dt = dt_suggested
    
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps  # Adjust dt to evenly divide t_end
    
    # ---- Create mesh ----
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, 
                                      cell_type=mesh.CellType.triangle)
    
    # ---- Function space ----
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # ---- Spatial coordinate ----
    x = ufl.SpatialCoordinate(domain)
    
    # ---- Coefficients ----
    kappa = 1.0 + 0.6 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # ---- Source term ----
    f = (ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1]) 
         + 0.3 * ufl.sin(10 * ufl.pi * x[0]) * ufl.sin(9 * ufl.pi * x[1]))
    
    # ---- Initial condition ----
    u_n = fem.Function(V, name="u_n")
    u_n.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    
    # ---- Boundary condition: homogeneous Dirichlet ----
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    bcs = [bc]
    
    # ---- Variational form (Backward Euler) ----
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_const = fem.Constant(domain, ScalarType(dt))
    
    a = (ufl.inner(u, v) / dt_const * ufl.dx 
         + ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx)
    L = (ufl.inner(f, v) * ufl.dx 
         + ufl.inner(u_n, v) / dt_const * ufl.dx)
    
    # ---- Compile forms ----
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # ---- Assemble LHS matrix (constant in time) ----
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    # ---- Setup KSP solver ----
    ksp_type = "gmres"
    pc_type = "hypre"
    rtol = 1e-8
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    pc = solver.getPC()
    pc.setType(pc_type)
    solver.setTolerances(rtol=rtol, atol=1e-12, max_it=1000)
    solver.setUp()
    
    # ---- Solution function ----
    u_sol = fem.Function(V, name="u")
    u_sol.x.array[:] = u_n.x.array[:]
    
    # ---- Create RHS vector (reusable) ----
    b_petsc = petsc.create_vector(V)
    
    # ---- Time stepping loop ----
    total_iterations = 0
    t = 0.0
    
    for step in range(n_steps):
        t += dt
        
        # Assemble RHS
        with b_petsc.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b_petsc, L_form)
        petsc.apply_lifting(b_petsc, [a_form], bcs=[bcs])
        b_petsc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b_petsc, bcs)
        
        # Solve
        solver.solve(b_petsc, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
    
    # ---- Evaluate on 50x50 grid ----
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_grid = np.full(points_3d.shape[0], np.nan)
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
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid[eval_map] = vals.flatten()
    
    u_grid = u_grid.reshape((nx_out, ny_out))
    
    # ---- Also compute initial condition on same grid ----
    u_n_init = fem.Function(V)
    u_n_init.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    
    u_initial = np.full(points_3d.shape[0], np.nan)
    pts_proc2 = []
    cells_proc2 = []
    emap2 = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            pts_proc2.append(points_3d[i])
            cells_proc2.append(links[0])
            emap2.append(i)
    if len(pts_proc2) > 0:
        vals2 = u_n_init.eval(np.array(pts_proc2), np.array(cells_proc2, dtype=np.int32))
        u_initial[emap2] = vals2.flatten()
    u_initial = u_initial.reshape((nx_out, ny_out))
    
    # Cleanup
    b_petsc.destroy()
    solver.destroy()
    A.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": scheme,
        }
    }
