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
    nu_val = pde_config.get("viscosity", 1.0)
    
    # 2. Create mesh - use fine mesh for accuracy
    N = 80
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # 3. Mixed function space: Taylor-Hood P2/P1
    P2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2)
    P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
    TH = ufl.MixedElement([P2, P1])
    W = fem.functionspace(domain, TH)
    
    # Subspaces
    V_sub, _ = W.sub(0).collapse()
    Q_sub, _ = W.sub(1).collapse()
    
    # 4. Define exact solutions and source term
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Exact velocity: u = (pi*cos(pi*y)*sin(pi*x), -pi*cos(pi*x)*sin(pi*y))
    u_exact = ufl.as_vector([
        pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
        -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    ])
    
    # Exact pressure: p = cos(pi*x)*cos(pi*y)
    p_exact = ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])
    
    # Source term: f = -nu * laplacian(u_exact) + grad(p_exact)
    # For u1 = pi*cos(pi*y)*sin(pi*x):
    #   d^2 u1/dx^2 = -pi^3 * cos(pi*y)*sin(pi*x)
    #   d^2 u1/dy^2 = -pi^3 * cos(pi*y)*sin(pi*x) (wait, let me recompute)
    # Actually let UFL handle it via symbolic differentiation
    # -nu * div(grad(u_exact)) + grad(p_exact) = f
    # But we can't take div(grad()) of a UFL expression directly on SpatialCoordinate
    # Let's compute manually:
    
    # u1 = pi * sin(pi*x) * cos(pi*y)
    # u1_xx = -pi^3 * sin(pi*x) * cos(pi*y)
    # u1_yy = -pi^3 * sin(pi*x) * cos(pi*y)
    # laplacian(u1) = -2*pi^3 * sin(pi*x) * cos(pi*y)
    
    # u2 = -pi * cos(pi*x) * sin(pi*y)
    # u2_xx = pi^3 * cos(pi*x) * sin(pi*y)
    # u2_yy = pi^3 * cos(pi*x) * sin(pi*y)
    # laplacian(u2) = 2*pi^3 * cos(pi*x) * sin(pi*y)
    
    # p = cos(pi*x)*cos(pi*y)
    # dp/dx = -pi*sin(pi*x)*cos(pi*y)
    # dp/dy = -pi*cos(pi*x)*sin(pi*y)
    
    # f1 = -nu * laplacian(u1) + dp/dx = nu*2*pi^3*sin(pi*x)*cos(pi*y) - pi*sin(pi*x)*cos(pi*y)
    # f2 = -nu * laplacian(u2) + dp/dy = -nu*2*pi^3*cos(pi*x)*sin(pi*y) - pi*cos(pi*x)*sin(pi*y)
    
    nu_c = fem.Constant(domain, default_scalar_type(nu_val))
    
    f1 = (nu_val * 2.0 * pi**3 - pi) * ufl.sin(pi * x[0]) * ufl.cos(pi * x[1])
    f2 = (-nu_val * 2.0 * pi**3 - pi) * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    f = ufl.as_vector([f1, f2])
    
    # 5. Trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    # 6. Variational form: Stokes
    a = (nu_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         - q * ufl.div(u) * ufl.dx)
    
    L = ufl.inner(f, v) * ufl.dx
    
    # 7. Boundary conditions - apply exact velocity on all boundaries
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facets(domain.topology)
    
    # Velocity BC
    u_bc_func = fem.Function(V_sub)
    u_bc_func.interpolate(lambda x: np.vstack([
        np.pi * np.cos(np.pi * x[1]) * np.sin(np.pi * x[0]),
        -np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])
    ]))
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    # Pin pressure at one point to fix the constant
    # Find a DOF near (0,0) for pressure
    def corner(x):
        return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))
    
    # Pin pressure DOF
    p_bc_func = fem.Function(Q_sub)
    p_bc_func.interpolate(lambda x: np.cos(np.pi * x[0]) * np.cos(np.pi * x[1]))
    
    # Find pressure dofs on boundary to pin one
    corner_facets = mesh.locate_entities_boundary(domain, 0, corner)  # vertices
    if len(corner_facets) > 0:
        dofs_p = fem.locate_dofs_topological((W.sub(1), Q_sub), 0, corner_facets)
        bc_p = fem.dirichletbc(p_bc_func, dofs_p, W.sub(1))
        bcs = [bc_u, bc_p]
    else:
        bcs = [bc_u]
    
    # 8. Solve
    ksp_type = "gmres"
    pc_type = "lu"
    rtol = 1e-12
    
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
    
    wh = fem.Function(W)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.getPC().setFactorSolverType("mumps")
    solver.setTolerances(rtol=rtol)
    
    solver.solve(b, wh.x.petsc_vec)
    wh.x.scatter_forward()
    
    iterations = solver.getIterationNumber()
    
    # 9. Extract velocity on 100x100 grid
    nx_out, ny_out = 100, 100
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    # Extract velocity subfunction
    uh = wh.sub(0).collapse()
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, tdim)
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
    
    # Evaluate velocity (2 components)
    vel_mag = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        # vals shape: (n_points, 2) for 2D vector
        for idx, global_idx in enumerate(eval_map):
            ux = vals[idx, 0]
            uy = vals[idx, 1]
            vel_mag[global_idx] = np.sqrt(ux**2 + uy**2)
    
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    # Cleanup
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": rtol,
            "iterations": iterations if iterations > 0 else 1,
        }
    }