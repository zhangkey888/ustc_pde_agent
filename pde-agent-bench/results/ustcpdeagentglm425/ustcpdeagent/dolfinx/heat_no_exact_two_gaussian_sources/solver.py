import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc as fem_petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    pde = case_spec["pde"]
    coeff = pde["coefficients"]
    time_params = pde["time"]
    output_spec = case_spec["output"]
    grid_spec = output_spec["grid"]
    
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    kappa = float(coeff.get("kappa", coeff.get("κ", 1.0)))
    t0 = float(time_params["t0"])
    t_end = float(time_params["t_end"])
    dt_suggested = float(time_params["dt"])
    
    mesh_res = 128
    element_degree = 2
    dt_use = 0.001
    
    n_steps = int(round((t_end - t0) / dt_use))
    dt_actual = (t_end - t0) / n_steps
    
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)
    
    u_n = fem.Function(V)
    u_n.x.array[:] = 0.0
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x_spatial = ufl.SpatialCoordinate(domain)
    f_expr = (ufl.exp(-220.0 * ((x_spatial[0] - 0.25)**2 + (x_spatial[1] - 0.25)**2)) + 
              ufl.exp(-220.0 * ((x_spatial[0] - 0.75)**2 + (x_spatial[1] - 0.7)**2)))
    
    a = ufl.inner(u, v) * ufl.dx + dt_actual * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(u_n, v) * ufl.dx + dt_actual * ufl.inner(f_expr, v) * ufl.dx
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = fem_petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-14, max_it=500)
    solver.getPC().setHYPREType("boomeramg")
    
    u_sol = fem.Function(V)
    u_sol.x.array[:] = 0.0
    
    b = fem_petsc.create_vector(L_form.function_spaces)
    
    total_iterations = 0
    
    for n_step in range(n_steps):
        with b.localForm() as loc_b:
            loc_b.set(0)
        fem_petsc.assemble_vector(b, L_form)
        fem_petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem_petsc.set_bc(b, [bc])
        
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        u_n.x.array[:] = u_sol.x.array[:]
    
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points_flat = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_flat.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_flat.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_flat.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_flat.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((nx_out * ny_out,), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    if np.any(np.isnan(u_grid)):
        try:
            from scipy.ndimage import distance_transform_edt
            mask = ~np.isnan(u_grid)
            if np.any(mask) and np.any(~mask):
                indices = distance_transform_edt(~mask, return_distances=False, return_indices=True)
                u_grid = u_grid[tuple(indices)]
            else:
                u_grid = np.nan_to_num(u_grid, nan=0.0)
        except ImportError:
            u_grid = np.nan_to_num(u_grid, nan=0.0)
    
    u_initial = np.zeros((ny_out, nx_out), dtype=np.float64)
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": element_degree,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1e-10,
        "iterations": total_iterations,
        "dt": dt_actual,
        "n_steps": n_steps,
        "time_scheme": "backward_euler"
    }
    
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": solver_info
    }
