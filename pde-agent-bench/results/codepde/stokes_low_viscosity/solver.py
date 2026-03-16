import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc
import basix


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    nu_val = pde_config.get("viscosity", 0.1)
    
    # 2. Create mesh - use high resolution for accuracy
    N = 80
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Mixed function space (Taylor-Hood: P2/P1)
    gdim = domain.geometry.dim
    
    # Velocity space P2 vector
    P2_el = basix.ufl.element("Lagrange", domain.basix_cell(), 2, shape=(gdim,))
    # Pressure space P1 scalar
    P1_el = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    
    # Mixed element
    mixed_el = basix.ufl.mixed_element([P2_el, P1_el])
    W = fem.functionspace(domain, mixed_el)
    
    # 4. Define variational problem
    # Trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Manufactured solution:
    # u_exact = (pi*cos(pi*y)*sin(pi*x), -pi*cos(pi*x)*sin(pi*y))
    # p_exact = cos(pi*x)*cos(pi*y)
    
    u_exact = ufl.as_vector([
        pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
        -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    ])
    p_exact = ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])
    
    # Compute source term: f = -nu * laplacian(u_exact) + grad(p_exact)
    # For u1 = pi*cos(pi*y)*sin(pi*x):
    #   d^2u1/dx^2 = -pi^3*cos(pi*y)*sin(pi*x)
    #   d^2u1/dy^2 = -pi^3*cos(pi*y)*sin(pi*x)  (wait, let me compute carefully)
    # Actually let's just use UFL to compute it symbolically
    # -nu * div(grad(u_exact)) + grad(p_exact) = f
    # But div(grad(u)) for vector u is the vector Laplacian
    
    # We need to be careful: for Stokes with the standard form
    # -nu * laplacian(u) + grad(p) = f
    # The bilinear form is: nu * inner(grad(u), grad(v)) - p * div(v) - q * div(u) = inner(f, v)
    
    # Source term from manufactured solution
    f = -nu_val * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)
    
    nu_c = fem.Constant(domain, default_scalar_type(nu_val))
    
    # Bilinear form
    a = (nu_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         - q * ufl.div(u) * ufl.dx)
    
    # Linear form
    L = ufl.inner(f, v) * ufl.dx
    
    # 5. Boundary conditions
    # Apply u = u_exact on all boundaries
    
    # Get the velocity subspace
    V_sub, _ = W.sub(0).collapse()
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary facets
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facets(domain.topology)
    
    # Velocity BC
    u_bc_func = fem.Function(V_sub)
    u_bc_expr = fem.Expression(u_exact, V_sub.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    # Pin pressure at one point to fix the constant
    # Find a DOF near (0, 0) for pressure
    Q_sub, _ = W.sub(1).collapse()
    
    p_bc_func = fem.Function(Q_sub)
    p_bc_expr = fem.Expression(p_exact, Q_sub.element.interpolation_points)
    p_bc_func.interpolate(p_bc_expr)
    
    # Pin pressure at a corner point
    def corner(x):
        return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))
    
    p_corner_dofs = fem.locate_dofs_geometrical((W.sub(1), Q_sub), corner)
    bc_p = fem.dirichletbc(p_bc_func, p_corner_dofs, W.sub(1))
    
    bcs = [bc_u, bc_p]
    
    # 6. Solve
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    b = petsc.create_vector(L_form)
    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, bcs)
    
    # Create solution function
    wh = fem.Function(W)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    solver.setTolerances(rtol=1e-12, atol=1e-14, max_it=5000)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")
    
    solver.solve(b, wh.x.petsc_vec)
    wh.x.scatter_forward()
    
    iterations = solver.getIterationNumber()
    
    # 7. Extract velocity on 100x100 grid
    nx_out, ny_out = 100, 100
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.zeros((3, nx_out * ny_out))
    points_2d[0, :] = XX.ravel()
    points_2d[1, :] = YY.ravel()
    
    # Extract velocity subfunction
    uh = wh.sub(0).collapse()
    
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_2d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_2d.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_2d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    # Evaluate velocity
    vel_values = np.full((nx_out * ny_out, gdim), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        for idx_local, idx_global in enumerate(eval_map):
            vel_values[idx_global, :] = vals[idx_local, :]
    
    # Compute velocity magnitude
    vel_mag = np.sqrt(vel_values[:, 0]**2 + vel_values[:, 1]**2)
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "rtol": 1e-12,
            "iterations": iterations,
        }
    }