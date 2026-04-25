import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc
import math

def solve(case_spec: dict) -> dict:
    t0 = 0.0
    t_end = 0.06
    dt_default = 0.001
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    mesh_res = 64
    degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    u_n = fem.Function(V)
    u_n.name = "u_n"
    
    x = ufl.SpatialCoordinate(domain)
    
    t = fem.Constant(domain, PETSc.ScalarType(t0))
    u_exact_expr = ufl.exp(-t) * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
    
    u_n.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
    
    u_initial = np.copy(u_n.x.array)
    
    kappa_expr = 1.0 + 0.4 * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
    
    f_expr = ufl.exp(-t)*ufl.sin(2*ufl.pi*x[0])*ufl.sin(2*ufl.pi*x[1]) - ufl.div(kappa_expr * ufl.grad(u_exact_expr))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_default))
    
    F = ufl.inner(u - u_n, v) * ufl.dx + dt_c * ufl.inner(kappa_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx - dt_c * ufl.inner(f_expr, v) * ufl.dx
    a = ufl.lhs(F)
    L = ufl.rhs(F)
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, boundary_facets)
    
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    A = fem.petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = fem.petsc.create_vector(L_form)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("jacobi")
    solver.setTolerances(rtol=1e-8)
    
    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array[:]
    
    t_val = t0
    total_iterations = 0
    n_steps = 0
    
    while t_val < t_end - 1e-8:
        t_val += dt_default
        t.value = t_val
        n_steps += 1
        
        u_bc.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
        
        with b.localForm() as loc:
            loc.set(0)
        fem.petsc.assemble_vector(b, L_form)
        fem.petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, [bc])
        
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iterations += solver.getIterationNumber()
        
        u_n.x.array[:] = u_sol.x.array[:]
        
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts.T)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_values = np.full(pts.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    u_grid = u_values.reshape((ny_out, nx_out))
    
    u_n.x.array[:] = u_initial
    u_init_values = np.full(pts.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals_init = u_n.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals_init.flatten()
    u_initial_grid = u_init_values.reshape((ny_out, nx_out))

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "rtol": 1e-8,
            "iterations": total_iterations,
            "dt": dt_default,
            "n_steps": n_steps,
            "time_scheme": "backward_euler"
        }
    }
