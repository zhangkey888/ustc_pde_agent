import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    pde = case_spec["pde"]
    out = case_spec["output"]
    grid = out["grid"]
    
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
    
    kappa = pde["coefficients"].get("kappa", 1.0)
    time_info = pde["time"]
    t0 = time_info["t0"]
    t_end = time_info["t_end"]
    dt_suggested = time_info["dt"]
    
    mesh_res = 128
    element_degree = 2
    
    # Crank-Nicolson (theta=0.5) for 2nd order time accuracy
    theta = 0.5
    n_steps = int(round((t_end - t0) / dt_suggested)) * 5
    actual_dt = (t_end - t0) / n_steps
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    x_coord = ufl.SpatialCoordinate(domain)
    f_expr = (ufl.sin(5*ufl.pi*x_coord[0]) * ufl.sin(3*ufl.pi*x_coord[1]) +
              0.5 * ufl.sin(9*ufl.pi*x_coord[0]) * ufl.sin(7*ufl.pi*x_coord[1]))
    
    f_func = fem.Function(V)
    f_func.interpolate(fem.Expression(f_expr, V.element.interpolation_points))
    
    u_n = fem.Function(V)
    u_n.x.array[:] = 0.0
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    kappa_const = fem.Constant(domain, PETSc.ScalarType(kappa))
    theta_const = fem.Constant(domain, PETSc.ScalarType(theta))
    
    a_form = fem.form((u * v / actual_dt + theta_const * kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx)
    L_form = fem.form((u_n * v / actual_dt - (1.0 - theta_const) * kappa_const * ufl.inner(ufl.grad(u_n), ufl.grad(v)) + f_func * v) * ufl.dx)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=rtol)
    
    u_sol = fem.Function(V)
    u_sol.x.array[:] = 0.0
    
    u_initial_grid = _sample_on_grid(domain, u_n, nx_out, ny_out, bbox)
    
    total_iterations = 0
    b = petsc.create_vector(L_form.function_spaces)
    
    for step in range(n_steps):
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        u_n.x.array[:] = u_sol.x.array[:]
    
    u_grid = _sample_on_grid(domain, u_sol, nx_out, ny_out, bbox)
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": total_iterations,
        "dt": actual_dt,
        "n_steps": n_steps,
        "time_scheme": "crank_nicolson",
    }
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }


def _sample_on_grid(domain, u_func, nx, ny, bbox):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    
    pts = np.zeros((nx * ny, 3), dtype=np.float64)
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    u_values = np.zeros(nx * ny, dtype=np.float64)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_proc = np.array(points_on_proc)
        cells_proc = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_proc, cells_proc)
        u_values[eval_map] = vals.flatten()
    
    return u_values.reshape(ny, nx)
