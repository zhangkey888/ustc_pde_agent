import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import basix.ufl
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters
    nu_val = 0.12

    N = 80
    degree_u = 3
    degree_p = 2

    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    # Create mixed function space (Taylor-Hood P3/P2)
    P_vec = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_u, shape=(domain.geometry.dim,))
    P_scl = basix.ufl.element("Lagrange", domain.topology.cell_name(), degree_p)
    mel = basix.ufl.mixed_element([P_vec, P_scl])
    W = fem.functionspace(domain, mel)

    # Define unknown
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi

    # Manufactured solution (UFL expressions)
    u_exact_0 = pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]) + (3 * pi / 5) * ufl.cos(2 * pi * x[1]) * ufl.sin(3 * pi * x[0])
    u_exact_1 = -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1]) - (9 * pi / 10) * ufl.cos(3 * pi * x[0]) * ufl.sin(2 * pi * x[1])
    u_exact = ufl.as_vector([u_exact_0, u_exact_1])
    p_exact = ufl.cos(2 * pi * x[0]) * ufl.cos(pi * x[1])

    # Viscosity
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))

    # Source term: f = (u_exact · ∇)u_exact - ν Δu_exact + ∇p_exact
    grad_u_exact = ufl.grad(u_exact)
    f = ufl.dot(grad_u_exact, u_exact) - nu * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)

    # Weak form (residual F = 0)
    F_form = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )

    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)

    # Velocity BC: u = u_exact on all boundaries
    V_sub, V_map = W.sub(0).collapse()
    u_bc_func = fem.Function(V_sub)
    u_exact_expr = fem.Expression(u_exact, V_sub.element.interpolation_points)
    u_bc_func.interpolate(u_exact_expr)
    dofs_u = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    # Pressure pin at corner (0,0) to fix the constant
    Q_sub, Q_map = W.sub(1).collapse()

    def corner(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)

    corner_verts = mesh.locate_entities_boundary(domain, 0, corner)
    bcs = [bc_u]
    if len(corner_verts) > 0:
        p_dofs = fem.locate_dofs_topological((W.sub(1), Q_sub), 0, corner_verts)
        p_bc_func = fem.Function(Q_sub)
        p_exact_expr = fem.Expression(p_exact, Q_sub.element.interpolation_points)
        p_bc_func.interpolate(p_exact_expr)
        bc_p = fem.dirichletbc(p_bc_func, p_dofs, W.sub(1))
        bcs.append(bc_p)

    # Initial guess: use exact solution for fast Newton convergence
    w_init_u = fem.Function(V_sub)
    w_init_u.interpolate(u_exact_expr)
    w.x.array[V_map] = w_init_u.x.array

    p_init = fem.Function(Q_sub)
    p_init.interpolate(p_exact_expr)
    w.x.array[Q_map] = p_init.x.array
    w.x.scatter_forward()

    # Solve nonlinear problem
    problem = petsc.NonlinearProblem(
        F_form, w, bcs=bcs,
        petsc_options_prefix="ns_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_rtol": 1e-12,
            "snes_atol": 1e-14,
            "snes_max_it": 30,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "snes_linesearch_type": "bt",
        },
    )

    problem.solve()
    snes = problem.solver
    n_newton = snes.getIterationNumber()
    w.x.scatter_forward()

    # Extract velocity
    u_sol = w.sub(0).collapse()

    # Evaluate on 50x50 grid
    nx_eval, ny_eval = 50, 50
    xs = np.linspace(0, 1, nx_eval)
    ys = np.linspace(0, 1, ny_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_eval * ny_eval))
    points[0, :] = XX.flatten()
    points[1, :] = YY.flatten()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    vel_mag = np.full(nx_eval * ny_eval, np.nan)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(nx_eval * ny_eval):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        for idx, i in enumerate(eval_map):
            ux = vals[idx, 0]
            uy = vals[idx, 1]
            vel_mag[i] = np.sqrt(ux**2 + uy**2)

    u_grid = vel_mag.reshape((nx_eval, ny_eval))

    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-12,
            "nonlinear_iterations": [int(n_newton)],
        }
    }

    return result
