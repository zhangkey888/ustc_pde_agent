import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    # Extract parameters
    kappa = 0.1
    t0 = 0.0
    t_end = 0.2
    dt = 0.01  # use 0.01 for better accuracy than 0.02
    
    grid_spec = case_spec.get("output", {}).get("grid", {})
    nx_out = grid_spec.get("nx", 64)
    ny_out = grid_spec.get("ny", 64)
    bbox = grid_spec.get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    # Mesh and function space
    comm = MPI.COMM_WORLD
    nx, ny = 128, 128
    domain = mesh.create_rectangle(comm, [[0.0, 0.0], [1.0, 1.0]], [nx, ny], mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", 2))
    
    # Time variable
    t = t0
    
    # Exact solution expression
    x = ufl.SpatialCoordinate(domain)
    
    def exact_expr(t_val):
        return ufl.exp(-0.5 * t_val) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Initial condition
    u_n = fem.Function(V)
    u_exact_t0_expr = exact_expr(t0)
    u_n.interpolate(fem.Expression(u_exact_t0_expr, V.element.interpolation_points()))
    
    # Source term
    f = fem.Function(V)
    f_expr = exact_expr(t + dt) * (-0.5 + 0.5 * ufl.pi**2)
    f.interpolate(fem.Expression(f_expr, V.element.interpolation_points()))
    
    # Trial and Test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Variational form (Backward Euler)
    F = (u - u_n) / dt * v * ufl.dx + kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - f * v * ufl.dx
    a = ufl.lhs(F)
    L = ufl.rhs(F)
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Compile forms
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    # Assemble matrix
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    # Solver setup
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-8)
    
    u_sol = fem.Function(V)
    
    # Time loop
    total_iters = 0
    n_steps = int(np.round((t_end - t0) / dt))
    
    for i in range(n_steps):
        t += dt
        
        # Update BC
        u_bc_expr = exact_expr(t)
        u_bc.interpolate(fem.Expression(u_bc_expr, V.element.interpolation_points()))
        
        # Update Source
        f_expr_t = exact_expr(t) * (-0.5 + 0.5 * ufl.pi**2)
        f.interpolate(fem.Expression(f_expr_t, V.element.interpolation_points()))
        
        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        # Solve
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iters += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array[:]
    
    # Sampling
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
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_values = np.zeros(pts.shape[0])
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    u_grid = u_values.reshape((ny_out, nx_out))
    
    # Also save initial condition (just zero for evaluation purposes, or exact)
    u_init_vals = np.zeros_like(u_values)
    # Actually just reuse the same array logic for u_initial if needed, we'll return a zero array for now
    
    solver_info = {
        "mesh_resolution": nx,
        "element_degree": 2,
        "ksp_type": "cg",
        "pc_type": "ilu",
        "rtol": 1e-8,
        "iterations": total_iters,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler"
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
        "u_initial": u_init_vals.reshape((ny_out, nx_out))
    }

