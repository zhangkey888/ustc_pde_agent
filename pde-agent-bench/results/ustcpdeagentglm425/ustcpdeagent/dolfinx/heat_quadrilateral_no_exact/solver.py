import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parse case spec
    pde = case_spec["pde"]
    coeff = pde.get("coefficients", {})
    kappa = float(coeff.get("kappa", 1.0))
    f_val = float(pde.get("source", 1.0))
    u0_val = float(pde.get("initial_condition", 0.0))
    
    time_params = pde["time"]
    t0 = float(time_params["t0"])
    t_end = float(time_params["t_end"])
    dt_suggested = float(time_params["dt"])
    
    bc_spec = pde.get("boundary_conditions", {})
    if isinstance(bc_spec.get("dirichlet"), dict):
        g_val = float(bc_spec["dirichlet"].get("value", 0.0))
    else:
        g_val = 0.0
    
    out_grid = case_spec["output"]["grid"]
    nx_out = out_grid["nx"]
    ny_out = out_grid["ny"]
    bbox = out_grid["bbox"]
    
    # Parameters
    mesh_res = 100
    elem_degree = 2
    dt = 0.0005
    time_scheme = "backward_euler"
    rtol = 1e-10
    
    # Create mesh
    p0 = np.array([float(bbox[0]), float(bbox[2])])
    p1 = np.array([float(bbox[1]), float(bbox[3])])
    domain = mesh.create_rectangle(
        comm, [p0, p1], [mesh_res, mesh_res],
        cell_type=mesh.CellType.quadrilateral
    )
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    V = fem.functionspace(domain, ("Lagrange", elem_degree))
    
    # BCs
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc_func = fem.Function(V)
    u_bc_func.x.array[:] = PETSc.ScalarType(g_val)
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)
    
    # Forms
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a_sys = ufl.inner(u, v) * ufl.dx + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    a_mass = ufl.inner(u, v) * ufl.dx
    L_src = dt * f_val * v * ufl.dx
    
    a_sys_form = fem.form(a_sys)
    a_mass_form = fem.form(a_mass)
    L_src_form = fem.form(L_src)
    
    A = petsc.assemble_matrix(a_sys_form, bcs=[bc])
    A.assemble()
    M = petsc.assemble_matrix(a_mass_form)
    M.assemble()
    b_src = petsc.assemble_vector(L_src_form)
    b_src.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b_src, [bc])
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.getPC().setHYPREType("boomeramg")
    solver.setTolerances(rtol=rtol)
    
    u_sol = fem.Function(V)
    u_sol.x.array[:] = PETSc.ScalarType(u0_val)
    u_n = fem.Function(V)
    u_n.x.array[:] = PETSc.ScalarType(u0_val)
    
    n_steps = int(round((t_end - t0) / dt))
    total_iterations = 0
    b = b_src.duplicate()
    
    for step in range(n_steps):
        M.mult(u_n.x.petsc_vec, b)
        b.axpy(1.0, b_src)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iterations += solver.getIterationNumber()
        u_n.x.array[:] = u_sol.x.array[:]
        u_n.x.scatter_forward()
    
    # Sample onto grid
    xs = np.linspace(float(bbox[0]), float(bbox[1]), nx_out)
    ys = np.linspace(float(bbox[2]), float(bbox[3]), ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    bb_tree = geometry.bb_tree(domain, tdim)
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
    
    u_values = np.full((pts.shape[1],), np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(
            np.array(points_on_proc),
            np.array(cells_on_proc, dtype=np.int32)
        )
        u_values[eval_map] = vals.flatten()
    
    if comm.Get_size() > 1:
        u_global = np.zeros_like(u_values)
        comm.Allreduce(u_values, u_global, op=MPI.SUM)
        u_global[np.isnan(u_values)] = np.nan
        u_values = u_global
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    if np.any(np.isnan(u_grid)):
        try:
            from scipy.ndimage import distance_transform_edt
            valid = ~np.isnan(u_grid)
            if np.any(valid):
                filled = np.nan_to_num(u_grid, nan=0.0)
                idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
                u_grid = filled[idx]
        except ImportError:
            u_grid = np.nan_to_num(u_grid, nan=0.0)
    
    u0_grid = np.full((ny_out, nx_out), u0_val, dtype=np.float64)
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_degree,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": rtol,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": time_scheme,
    }
    
    return {
        "u": u_grid.astype(np.float64),
        "u_initial": u0_grid.astype(np.float64),
        "solver_info": solver_info,
    }
