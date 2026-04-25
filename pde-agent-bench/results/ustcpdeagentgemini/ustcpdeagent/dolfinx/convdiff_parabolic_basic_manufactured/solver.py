import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc
from dolfinx.fem import petsc

def solve(case_spec: dict) -> dict:
    nx = case_spec["output"]["grid"]["nx"]
    ny = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Parameters
    epsilon = 0.1
    beta = [1.0, 0.5]
    t0 = 0.0
    t_end = 0.1
    dt = 0.01  # Use 0.01 for better accuracy
    n_steps = int(round((t_end - t0) / dt))
    
    mesh_res = 64
    degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    u_n = fem.Function(V)
    
    x = ufl.SpatialCoordinate(domain)
    t_var = fem.Constant(domain, PETSc.ScalarType(t0))
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    
    # Exact solution for initial condition and boundary condition
    u_exact = ufl.exp(-t_var) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Source term f = du/dt - eps * laplace(u) + beta . grad(u)
    # du/dt = -u_exact
    # grad(u) = [exp(-t)*pi*cos(pi*x)*sin(pi*y), exp(-t)*pi*sin(pi*x)*cos(pi*y)]
    # laplace(u) = -2*pi^2 * u_exact
    f = -u_exact - epsilon * (-2 * ufl.pi**2 * u_exact) + beta[0] * ufl.exp(-t_var) * ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) + beta[1] * ufl.exp(-t_var) * ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
    
    # Initial condition
    u_exact_expr = fem.Expression(u_exact, V.element.interpolation_points())
    u_n.interpolate(u_exact_expr)
    
    # For return
    u_initial = np.zeros((ny, nx))
    
    # Probing setup
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    eval_pts = []
    eval_cells = []
    eval_map = []
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            eval_pts.append(pts[i])
            eval_cells.append(links[0])
            eval_map.append(i)
    
    if len(eval_pts) > 0:
        vals = u_n.eval(np.array(eval_pts), np.array(eval_cells, dtype=np.int32))
        u_initial.ravel()[eval_map] = vals.flatten()
        
    # Boundary condition
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(u_exact_expr)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Variational problem (Backward Euler)
    # (u - u_n)/dt - eps * div(grad(u)) + beta . grad(u) = f
    # (u, v) + dt * eps * (grad(u), grad(v)) + dt * (beta . grad(u), v) = (u_n, v) + dt * (f, v)
    
    a = ufl.inner(u, v) * ufl.dx + dt_const * epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + dt_const * ufl.inner(ufl.as_vector(beta), ufl.grad(u)) * v * ufl.dx
    L = ufl.inner(u_n, v) * ufl.dx + dt_const * ufl.inner(f, v) * ufl.dx
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    
    u_sol = fem.Function(V)
    
    t = t0
    for i in range(n_steps):
        t += dt
        t_var.value = t
        u_bc.interpolate(u_exact_expr)
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        u_n.x.array[:] = u_sol.x.array
    
    u_grid = np.zeros((ny, nx))
    if len(eval_pts) > 0:
        vals = u_sol.eval(np.array(eval_pts), np.array(eval_cells, dtype=np.int32))
        u_grid.ravel()[eval_map] = vals.flatten()
        
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-8,
            "iterations": n_steps,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler"
        }
    }
