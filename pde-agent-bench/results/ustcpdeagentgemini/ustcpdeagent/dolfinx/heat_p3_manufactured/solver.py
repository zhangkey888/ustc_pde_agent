import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Grid specification
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # FEM parameters (balanced accuracy and speed)
    nx_mesh = 80
    ny_mesh = 80
    degree = 2
    dt = 0.002
    t0 = 0.0
    t_end = 0.08
    
    domain = mesh.create_unit_square(comm, nx_mesh, ny_mesh, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    u_n = fem.Function(V)
    u_n.name = "u_n"
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Exact solution for initial condition and boundary
    x = ufl.SpatialCoordinate(domain)
    t = fem.Constant(domain, PETSc.ScalarType(t0))
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt))
    kappa = fem.Constant(domain, PETSc.ScalarType(1.0))
    
    u_exact = ufl.exp(-t) * ufl.sin(ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # Source term f = du/dt - nabla^2 u
    f = (5 * ufl.pi**2 - 1.0) * u_exact
    
    # Initial condition
    u_n.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    u_n.x.scatter_forward()
    u_initial = np.copy(u_n.x.array)
    
    # Boundary condition
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Variational problem (Backward Euler)
    F = ufl.inner((u - u_n) / dt_c, v) * ufl.dx + kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - ufl.inner(f, v) * ufl.dx
    a = ufl.lhs(F)
    L = ufl.rhs(F)
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector([V])
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.getPC().setFactorSolverType("mumps")
    
    u_sol = fem.Function(V)
    total_iters = 0
    t_current = t0
    n_steps = int(np.round((t_end - t0) / dt))
    
    for i in range(n_steps):
        t_current += dt
        t.value = t_current
        
        # Update BC
        u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
        
        with b.localForm() as loc_b:
            loc_b.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iters += solver.getIterationNumber()
        u_n.x.array[:] = u_sol.x.array
    
    # Sampling
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    tree = bb_tree(domain, domain.topology.dim)
    cell_candidates = compute_collisions_points(tree, pts)
    colliding = compute_colliding_cells(domain, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts)):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_values = np.full((len(pts),), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    u_grid = u_values.reshape(ny_out, nx_out)
    u_init_grid = np.zeros_like(u_grid)
    
    solver_info = {
        "mesh_resolution": nx_mesh,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 0.0,
        "iterations": total_iters,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler"
    }
    
    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": solver_info
    }

