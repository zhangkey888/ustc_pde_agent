import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
from basix.ufl import element, mixed_element


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    nu_val = pde_config.get("viscosity", 1.0)
    
    # 2. Create mesh - use higher resolution for P3/P2 to meet accuracy
    N = 24  # With P3/P2 Taylor-Hood, this should be sufficient
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # 3. Define Taylor-Hood mixed elements P3/P2
    degree_u = 3
    degree_p = 2
    
    P_u = element("Lagrange", domain.basix_cell(), degree_u, shape=(tdim,))
    P_p = element("Lagrange", domain.basix_cell(), degree_p)
    TH = mixed_element([P_u, P_p])
    
    W = fem.functionspace(domain, TH)
    
    # Sub-spaces for BCs
    V_sub, V_sub_map = W.sub(0).collapse()
    Q_sub, Q_sub_map = W.sub(1).collapse()
    
    # 4. Define exact solution and source term
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    
    # Exact velocity: u = (pi*cos(pi*y)*sin(pi*x), -pi*cos(pi*x)*sin(pi*y))
    u_exact = ufl.as_vector([
        ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]),
        -ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    ])
    
    # Exact pressure: p = cos(pi*x)*cos(pi*y)
    p_exact = ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
    
    # Source term: f = -nu * laplacian(u_exact) + grad(p_exact)
    # For u1 = pi*cos(pi*y)*sin(pi*x):
    #   laplacian(u1) = -2*pi^3*cos(pi*y)*sin(pi*x)
    # For u2 = -pi*cos(pi*x)*sin(pi*y):
    #   laplacian(u2) = 2*pi^3*cos(pi*x)*sin(pi*y)
    # grad(p) = (-pi*sin(pi*x)*cos(pi*y), -pi*cos(pi*x)*sin(pi*y))
    
    nu_c = fem.Constant(domain, PETSc.ScalarType(nu_val))
    
    # Compute f symbolically
    # f = -nu * div(grad(u_exact)) + grad(p_exact)
    f = -nu_c * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)
    
    # 5. Define variational forms
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    a = (nu_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         - q * ufl.div(u) * ufl.dx)
    
    L_form = ufl.inner(f, v) * ufl.dx
    
    # 6. Boundary conditions - apply exact velocity on all boundaries
    # Interpolate exact velocity onto V_sub
    u_bc_func = fem.Function(V_sub)
    u_exact_expr = fem.Expression(u_exact, V_sub.element.interpolation_points)
    u_bc_func.interpolate(u_exact_expr)
    
    # All boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    # Pin pressure at one point to remove nullspace
    # Find a vertex near (0,0)
    p_bc_func = fem.Function(Q_sub)
    p_exact_expr = fem.Expression(p_exact, Q_sub.element.interpolation_points)
    p_bc_func.interpolate(p_exact_expr)
    
    # Pin pressure at corner (0,0)
    def corner(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    
    corner_facets = mesh.locate_entities_boundary(domain, fdim, corner)
    if len(corner_facets) > 0:
        dofs_p = fem.locate_dofs_topological((W.sub(1), Q_sub), fdim, corner_facets)
        bc_p = fem.dirichletbc(p_bc_func, dofs_p, W.sub(1))
        bcs = [bc_u, bc_p]
    else:
        bcs = [bc_u]
    
    # 7. Solve using direct solver for saddle-point system
    a_form = fem.form(a)
    L_compiled = fem.form(L_form)
    
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    b = petsc.create_vector(L_compiled)
    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_compiled)
    petsc.apply_lifting(b, [a_form], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, bcs)
    
    # Create solution function
    wh = fem.Function(W)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")
    solver.setUp()
    
    solver.solve(b, wh.x.petsc_vec)
    wh.x.scatter_forward()
    
    iterations = solver.getIterationNumber()
    
    # 8. Extract velocity magnitude on 100x100 grid
    nx_out, ny_out = 100, 100
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    # Collapse velocity subspace
    uh_collapsed = wh.sub(0).collapse()
    
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
    
    vel_mag = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh_collapsed.eval(pts_arr, cells_arr)
        # vals shape: (n_points, 2) for 2D velocity
        mag = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            vel_mag[global_idx] = mag[idx]
    
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": iterations if iterations else 1,
        }
    }