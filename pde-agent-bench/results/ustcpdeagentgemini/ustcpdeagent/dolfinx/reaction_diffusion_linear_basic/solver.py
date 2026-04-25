import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    # Output grid parameters
    out_grid = case_spec.get("output", {}).get("grid", {})
    nx = out_grid.get("nx", 50)
    ny = out_grid.get("ny", 50)
    bbox = out_grid.get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    # Time parameters
    t0 = 0.0
    t_end = 0.5
    dt = 0.01
    
    # Mesh parameters
    mesh_res = 64
    deg = 2
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", deg))
    
    t = t0
    
    x = ufl.SpatialCoordinate(domain)
    # Manufactured solution: u = exp(-t) * sin(pi*x) * sin(pi*y)
    u_exact_expr = ufl.exp(-t) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points()))
    u_initial = u_n.x.array.copy()
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Assuming standard form: du/dt - nabla^2 u + u = f
    # f = 2 * pi^2 * exp(-t) * sin(pi*x) * sin(pi*y)
    t_new_const = fem.Constant(domain, PETSc.ScalarType(t + dt))
    f_new = 2.0 * ufl.pi**2 * ufl.exp(-t_new_const) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    a = u * v * ufl.dx + dt * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + dt * u * v * ufl.dx
    L = u_n * v * ufl.dx + dt * f_new * v * ufl.dx
    
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    boundary_facets = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, lambda x: np.full(x.shape[1], True))
    boundary_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, boundary_facets)
    
    bc_func = fem.Function(V)
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form)
    A.assemble()
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("ilu")
    solver.setTolerances(rtol=1e-8)
    
    b = petsc.create_vector(L_form.function_spaces)
    
    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array[:]
    
    n_steps = int(np.round((t_end - t0) / dt))
    total_iters = 0
    
    for i in range(n_steps):
        t += dt
        t_new_const.value = t
        
        u_exact_bc = ufl.exp(-t) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        bc_func.interpolate(fem.Expression(u_exact_bc, V.element.interpolation_points()))
        bc = fem.dirichletbc(bc_func, boundary_dofs)
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iters += solver.getIterationNumber()
        u_n.x.array[:] = u_sol.x.array[:]
        
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx*ny)]
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_values = np.full(nx*ny, np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    u_grid = u_values.reshape((ny, nx))
    
    u_init_func = fem.Function(V)
    u_init_func.x.array[:] = u_initial
    init_vals = np.full(nx*ny, np.nan)
    if len(points_on_proc) > 0:
        iv = u_init_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        init_vals[eval_map] = iv.flatten()
    u_init_grid = init_vals.reshape((ny, nx))
    
    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": deg,
            "ksp_type": "cg",
            "pc_type": "ilu",
            "rtol": 1e-8,
            "iterations": total_iters,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler"
        }
    }
