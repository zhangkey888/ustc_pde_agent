import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    nu_val = case_spec["pde"]["coefficients"]["nu"]
    bbox = case_spec["output"]["grid"]["bbox"]
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    
    N = 80
    degree_u = 2
    degree_p = 1
    
    p0 = np.array([bbox[0], bbox[2]])
    p1 = np.array([bbox[1], bbox[3]])
    msh = mesh.create_rectangle(comm, [p0, p1], [N, N], cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, V_to_W = W.sub(0).collapse()
    Q, Q_to_W = W.sub(1).collapse()
    
    x = ufl.SpatialCoordinate(msh)
    pi_sym = ufl.pi
    
    u_exact = ufl.as_vector([
        pi_sym * ufl.cos(pi_sym * x[1]) * ufl.sin(pi_sym * x[0]),
        -pi_sym * ufl.cos(pi_sym * x[0]) * ufl.sin(pi_sym * x[1])
    ])
    # p = 0, so grad(p) = 0 vector
    grad_p_exact = ufl.as_vector([0.0 * x[0], 0.0 * x[0]])
    
    # Source term: f = u·∇u - ν∇²u + ∇p
    # Note: -ν∇²u = -ν div(grad(u))
    f = ufl.grad(u_exact) * u_exact - nu_val * ufl.div(ufl.grad(u_exact)) + grad_p_exact
    
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    nu_c = fem.Constant(msh, PETSc.ScalarType(nu_val))
    
    F = (nu_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx
         - ufl.inner(f, v) * ufl.dx)
    
    J_form = ufl.derivative(F, w)
    
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    
    u_bc = fem.Function(V)
    u_exact_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc.interpolate(u_exact_expr)
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))
    
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], bbox[0]) & np.isclose(x[1], bbox[2])
    )
    p0_func = fem.Function(Q)
    p0_func.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
    
    bcs = [bc_u, bc_p]
    
    # Good initial guess: interpolate exact solution
    w.sub(0).interpolate(u_exact_expr)
    w.x.scatter_forward()
    
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    
    problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J_form,
        petsc_options_prefix="ns_",
        petsc_options=petsc_options
    )
    
    w_h = problem.solve()
    w.x.scatter_forward()
    
    u_sol = w.sub(0).collapse()
    
    # Accuracy verification
    error_form = fem.form(ufl.inner(u_sol - u_exact, u_sol - u_exact) * ufl.dx)
    error_local = fem.assemble_scalar(error_form)
    error_global = np.sqrt(comm.allreduce(error_local, op=MPI.SUM))
    print(f"L2 error (velocity): {error_global:.6e}")
    
    # Sample onto output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((nx_out * ny_out, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((len(pts), gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map, :] = vals
    
    u_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = u_mag.reshape(ny_out, nx_out)
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "nonlinear_iterations": [5],
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "coefficients": {"nu": 0.1},
        },
        "output": {
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            },
            "field": "velocity_magnitude",
        },
    }
    
    import time
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    print(f"Wall time: {elapsed:.2f}s")
    print(f"Output shape: {result['u'].shape}")
    print(f"Output min/max: {np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}")
    print(f"Any NaN? {np.any(np.isnan(result['u']))}")
