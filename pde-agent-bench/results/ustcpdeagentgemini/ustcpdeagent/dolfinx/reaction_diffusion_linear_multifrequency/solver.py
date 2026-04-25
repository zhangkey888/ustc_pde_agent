import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc

def exact_solution_expr(x, t):
    # u = exp(-t)*(sin(pi*x)*sin(pi*y) + 0.2*sin(6*pi*x)*sin(5*pi*y))
    return ufl.exp(-t) * (
        ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) +
        0.2 * ufl.sin(6 * ufl.pi * x[0]) * ufl.sin(5 * ufl.pi * x[1])
    )

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    epsilon = case_spec.get("epsilon", 1.0)
    mesh_res = case_spec.get("mesh_resolution", 64)
    dt = case_spec.get("dt", 0.005)
    t_end = case_spec.get("t_end", 0.4)
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Mesh and Function Space
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 2))
    
    # Time variable
    t_ufl = fem.Constant(domain, PETSc.ScalarType(0.0))
    dt_ufl = fem.Constant(domain, PETSc.ScalarType(dt))
    
    x = ufl.SpatialCoordinate(domain)
    u_exact = exact_solution_expr(x, t_ufl)
    
    # Boundary Conditions
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    
    def update_bc(t_val):
        t_ufl.value = t_val
        expr = fem.Expression(u_exact, V.element.interpolation_points())
        u_bc.interpolate(expr)
        
    update_bc(0.0)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Initial Condition
    u_n = fem.Function(V)
    u_n.interpolate(u_bc)
    u_initial = u_n.x.array.copy()
    
    # Crank-Nicolson formulation
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Exact source term f = du/dt - epsilon * Laplace(u) + R(u)
    # Let's assume R(u) = u for a generic reaction term
    def R(u_func):
        return u_func
        
    f_exact = ufl.diff(u_exact, t_ufl) - epsilon * ufl.div(ufl.grad(u_exact)) + R(u_exact)
    
    # Crank-Nicolson: (u - u_n)/dt - epsilon * Laplace(u_mid) + R(u_mid) = f_mid
    u_mid = 0.5 * (u + u_n)
    
    F = (u - u_n) / dt_ufl * v * ufl.dx \
      + epsilon * ufl.inner(ufl.grad(u_mid), ufl.grad(v)) * ufl.dx \
      + R(u_mid) * v * ufl.dx \
      - f_exact * v * ufl.dx
      
    a = ufl.lhs(F)
    L = ufl.rhs(F)
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)
    
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("ilu")
    solver.setTolerances(rtol=1e-9)
    
    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array[:]
    
    t = 0.0
    n_steps = int(np.round(t_end / dt))
    total_iters = 0
    
    for i in range(n_steps):
        t += dt
        update_bc(t) # Updates u_bc and t_ufl which affects f_exact
        
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
        
    # Interpolate onto output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
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
            
    u_out = np.full(len(pts), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_out[eval_map] = vals.flatten()
        
    u_grid = u_out.reshape((ny_out, nx_out))
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": 2,
        "ksp_type": "cg",
        "pc_type": "ilu",
        "rtol": 1e-9,
        "iterations": total_iters,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "crank_nicolson"
    }
    
    return {"u": u_grid, "solver_info": solver_info}
