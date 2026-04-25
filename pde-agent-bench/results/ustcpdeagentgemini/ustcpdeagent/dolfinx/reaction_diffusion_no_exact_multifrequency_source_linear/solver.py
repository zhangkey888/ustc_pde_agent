import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import math

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Grid info
    grid_spec = case_spec["output"]["grid"]
    nx = grid_spec["nx"]
    ny = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = bbox
    
    # Time params
    time_spec = case_spec.get("time", {})
    t0 = time_spec.get("t0", 0.0)
    t_end = time_spec.get("t_end", 0.5)
    dt = time_spec.get("dt", 0.01)
    
    # PDE params
    pde_spec = case_spec.get("pde", {})
    epsilon = pde_spec.get("epsilon", 1.0)
    alpha = pde_spec.get("reaction_alpha", 0.0)
    
    # Mesh and function space
    mesh_res = 128
    domain = mesh.create_rectangle(comm, [[xmin, ymin], [xmax, ymax]], [mesh_res, mesh_res], cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 2))
    
    # Boundary Conditions
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Initial Condition
    u_n = fem.Function(V)
    x_expr = ufl.SpatialCoordinate(domain)
    u0_expr = ufl.sin(ufl.pi * x_expr[0]) * ufl.sin(ufl.pi * x_expr[1])
    u_n.interpolate(fem.Expression(u0_expr, V.element.interpolation_points()))
    
    # Source Term
    f_expr = ufl.sin(5*ufl.pi*x_expr[0])*ufl.sin(3*ufl.pi*x_expr[1]) + 0.5*ufl.sin(9*ufl.pi*x_expr[0])*ufl.sin(7*ufl.pi*x_expr[1])
    
    # Variational Problem (Crank-Nicolson)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # u_mid = 0.5*(u + u_n)
    theta = 0.5
    u_mid = theta * u + (1 - theta) * u_n
    
    # Weak form: (u - u_n)/dt + eps * grad(u_mid) . grad(v) + alpha * u_mid * v = f * v
    F = (u - u_n) * v * ufl.dx \
        + dt * epsilon * ufl.inner(ufl.grad(u_mid), ufl.grad(v)) * ufl.dx \
        + dt * alpha * u_mid * v * ufl.dx \
        - dt * f_expr * v * ufl.dx
        
    a = ufl.lhs(F)
    L = ufl.rhs(F)
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-8)
    
    u_sol = fem.Function(V)
    
    # Time stepping
    t = t0
    n_steps = int(round((t_end - t0) / dt))
    total_iters = 0
    
    u_initial_eval = u_n.copy()
    
    for _ in range(n_steps):
        t += dt
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iters += solver.getIterationNumber()
        
        u_n.x.array[:] = u_sol.x.array
        
    # Evaluate on regular grid
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]
    
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    points_on_proc = []
    cells = []
    eval_map = []
    for i, pt in enumerate(pts):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pt)
            cells.append(links[0])
            eval_map.append(i)
            
    u_grid_flat = np.full(nx * ny, np.nan)
    u_init_flat = np.full(nx * ny, np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_grid_flat[eval_map] = vals.flatten()
        vals_init = u_initial_eval.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_init_flat[eval_map] = vals_init.flatten()
        
    u_grid = u_grid_flat.reshape(ny, nx)
    u_init = u_init_flat.reshape(ny, nx)
    
    # Broadcast to handle parallel gracefully if needed, but benchmark normally runs in serial
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": 2,
        "ksp_type": "cg",
        "pc_type": "ilu",
        "rtol": 1e-8,
        "iterations": total_iters,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "crank_nicolson"
    }
    
    return {
        "u": u_grid,
        "u_initial": u_init,
        "solver_info": solver_info
    }
