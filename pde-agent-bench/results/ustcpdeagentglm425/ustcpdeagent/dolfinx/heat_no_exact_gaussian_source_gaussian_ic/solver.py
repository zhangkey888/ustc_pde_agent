import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    pde = case_spec["pde"]
    kappa = float(pde["coefficients"]["kappa"])
    t0 = float(pde["time"]["t0"])
    t_end = float(pde["time"]["t_end"])
    
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
    xmin, xmax, ymin, ymax = bbox
    
    mesh_res = 128
    element_degree = 2
    dt = 0.002
    n_steps = int(round((t_end - t0) / dt))
    time_scheme = "backward_euler"
    
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    
    u_n = fem.Function(V)
    u1 = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    def source_expr(x):
        return np.exp(-200.0 * ((x[0] - 0.3)**2 + (x[1] - 0.7)**2))
    
    def ic_expr(x):
        return np.exp(-120.0 * ((x[0] - 0.6)**2 + (x[1] - 0.4)**2))
    
    u_n.interpolate(ic_expr)
    
    f_func = fem.Function(V)
    f_func.interpolate(source_expr)
    
    a_form = ufl.inner(u1, v) * ufl.dx + dt * kappa * ufl.inner(ufl.grad(u1), ufl.grad(v)) * ufl.dx
    L_form = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f_func, v) * ufl.dx
    
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    
    A = petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_compiled.function_spaces)
    
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.CG)
    ksp.getPC().setType(PETSc.PC.Type.HYPRE)
    ksp.getPC().setHYPREType("boomeramg")
    rtol = 1e-10
    ksp.setTolerances(rtol=rtol, atol=1e-12, max_it=500)
    
    u_sol = fem.Function(V)
    total_iterations = 0
    
    for step in range(n_steps):
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_compiled)
        petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        ksp.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iterations += ksp.getIterationNumber()
        u_n.x.array[:] = u_sol.x.array[:]
    
    # Sample onto output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
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
    
    u_values = np.full((points.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    if comm.size > 1:
        u_values_local = u_values.copy()
        u_values_local[np.isnan(u_values_local)] = 0.0
        u_values_global = np.zeros_like(u_values)
        comm.Allreduce(u_values_local, u_values_global, op=MPI.SUM)
        u_values = u_values_global
    
    u_grid = u_values.reshape(ny_out, nx_out)
    u_initial_grid = ic_expr(points).reshape(ny_out, nx_out)
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": element_degree,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": rtol,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": time_scheme,
    }
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }
