import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element, mixed_element
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    nu_val = 0.18
    if 'pde' in case_spec:
        pde = case_spec['pde']
        if 'viscosity' in pde:
            nu_val = float(pde['viscosity'])
    
    nx_out = 50
    ny_out = 50
    if 'output' in case_spec:
        out = case_spec['output']
        if 'nx' in out:
            nx_out = int(out['nx'])
        if 'ny' in out:
            ny_out = int(out['ny'])
    
    N = 64
    deg_u = 3
    deg_p = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    el_u = element("Lagrange", domain.topology.cell_name(), deg_u, shape=(2,))
    el_p = element("Lagrange", domain.topology.cell_name(), deg_p)
    mel = mixed_element([el_u, el_p])
    W = fem.functionspace(domain, mel)
    
    V, V_map = W.sub(0).collapse()
    Q, Q_map = W.sub(1).collapse()
    
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    x = ufl.SpatialCoordinate(domain)
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    
    pi = ufl.pi
    u_exact_0 = 6.0 * (1.0 - ufl.tanh(6.0 * (x[1] - 0.5))**2) * ufl.sin(pi * x[0])
    u_exact_1 = -pi * ufl.tanh(6.0 * (x[1] - 0.5)) * ufl.cos(pi * x[0])
    u_exact = ufl.as_vector([u_exact_0, u_exact_1])
    p_exact = ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])
    
    # Compute source term: u·∇u - ν∇²u + ∇p = f
    grad_u_exact = ufl.grad(u_exact)
    f = ufl.dot(grad_u_exact, u_exact) - nu * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)
    
    # Weak form residual
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
    
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    # Pressure pin at corner (0,0)
    def corner(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    
    p_bc_func = fem.Function(Q)
    p_bc_expr = fem.Expression(p_exact, Q.element.interpolation_points)
    p_bc_func.interpolate(p_bc_expr)
    dofs_p = fem.locate_dofs_geometrical((W.sub(1), Q), corner)
    bc_p = fem.dirichletbc(p_bc_func, dofs_p, W.sub(1))
    
    bcs = [bc_u, bc_p]
    
    # Initial guess from exact solution (helps Newton converge in ~1-2 iterations)
    w_init_u = fem.Function(V)
    w_init_u.interpolate(u_bc_expr)
    w.sub(0).interpolate(w_init_u)
    
    w_init_p = fem.Function(Q)
    w_init_p.interpolate(p_bc_expr)
    w.sub(1).interpolate(w_init_p)
    
    # Nonlinear solve with MUMPS direct solver
    petsc_opts = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_rtol": 1e-12,
        "snes_atol": 1e-14,
        "snes_max_it": 25,
    }
    
    problem = petsc.NonlinearProblem(
        F_form, w, bcs=bcs,
        petsc_options_prefix="ns_",
        petsc_options=petsc_opts,
    )
    
    problem.solve()
    w.x.scatter_forward()
    
    snes = problem.solver
    n_newton = snes.getIterationNumber()
    
    u_sol = w.sub(0).collapse()
    
    # Evaluate on output grid
    x_coords = np.linspace(0, 1, nx_out)
    y_coords = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    points = np.zeros((nx_out * ny_out, 3))
    points[:, 0] = X.flatten()
    points[:, 1] = Y.flatten()
    
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    u_values = np.full((nx_out * ny_out, 2), np.nan)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(points)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        for idx, global_idx in enumerate(eval_map):
            u_values[global_idx, :] = vals[idx, :]
    
    u_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = u_mag.reshape((nx_out, ny_out))
    
    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": deg_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-12,
            "nonlinear_iterations": [int(n_newton)],
        }
    }
    
    return result
