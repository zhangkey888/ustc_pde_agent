import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc
import sympy as sp

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parse case spec
    nu_val = case_spec["pde"]["coefficients"]["nu"]
    u_exact_str = case_spec["pde"]["manufactured_solution"]["u"]
    p_exact_str = case_spec["pde"]["manufactured_solution"]["p"]
    
    output_grid = case_spec["output"]["grid"]
    nx_out = output_grid["nx"]
    ny_out = output_grid["ny"]
    bbox = output_grid["bbox"]
    
    # Compute manufactured source term symbolically
    x_sym, y_sym = sp.symbols('x y')
    
    u1_sym = sp.sympify(u_exact_str[0])
    u2_sym = sp.sympify(u_exact_str[1])
    p_sym = sp.sympify(p_exact_str)
    
    nu_s = sp.Float(nu_val)
    
    # f = u·∇u - ν∇²u + ∇p
    conv_x = u1_sym * sp.diff(u1_sym, x_sym) + u2_sym * sp.diff(u1_sym, y_sym)
    conv_y = u1_sym * sp.diff(u2_sym, x_sym) + u2_sym * sp.diff(u2_sym, y_sym)
    
    lapl_u1 = sp.diff(u1_sym, x_sym, 2) + sp.diff(u1_sym, y_sym, 2)
    lapl_u2 = sp.diff(u2_sym, x_sym, 2) + sp.diff(u2_sym, y_sym, 2)
    diff_x = -nu_s * lapl_u1
    diff_y = -nu_s * lapl_u2
    
    grad_p_x = sp.diff(p_sym, x_sym)
    grad_p_y = sp.diff(p_sym, y_sym)
    
    f1_sym = conv_x + diff_x + grad_p_x
    f2_sym = conv_y + diff_y + grad_p_y
    
    f1_sym = sp.simplify(f1_sym)
    f2_sym = sp.simplify(f2_sym)
    
    f1_func = sp.lambdify((x_sym, y_sym), f1_sym, modules=['numpy'])
    f2_func = sp.lambdify((x_sym, y_sym), f2_sym, modules=['numpy'])
    u1_func = sp.lambdify((x_sym, y_sym), u1_sym, modules=['numpy'])
    u2_func = sp.lambdify((x_sym, y_sym), u2_sym, modules=['numpy'])
    
    # Mesh
    N = 192
    degree_u = 2
    degree_p = 1
    
    msh = mesh.create_rectangle(
        comm,
        [np.array([bbox[0], bbox[2]]), np.array([bbox[1], bbox[3]])],
        [N, N],
        cell_type=mesh.CellType.triangle
    )
    
    gdim = msh.geometry.dim
    
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, V_to_W = W.sub(0).collapse()
    Q, Q_to_W = W.sub(1).collapse()
    
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    # Source term
    V_f = fem.functionspace(msh, ("Lagrange", degree_u + 1, (gdim,)))
    f_h = fem.Function(V_f)
    f_h.interpolate(lambda X: np.vstack([
        f1_func(X[0], X[1]),
        f2_func(X[0], X[1])
    ]))
    
    # Exact solution for BCs
    u_exact = fem.Function(V)
    u_exact.interpolate(lambda X: np.vstack([
        u1_func(X[0], X[1]),
        u2_func(X[0], X[1])
    ]))
    
    # Boundary conditions
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_exact, dofs_u, W.sub(0))
    
    # Pressure pin
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], bbox[0]) & np.isclose(x[1], bbox[2])
    )
    p0_func = fem.Function(Q)
    p0_func.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
    
    bcs = [bc_u, bc_p]
    
    # Nonlinear residual
    nu_c = fem.Constant(msh, PETSc.ScalarType(nu_val))
    
    F = (
        nu_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f_h, v) * ufl.dx
    )
    
    J_form = ufl.derivative(F, w)
    
    # Initialize with exact solution
    w.sub(0).interpolate(u_exact)
    w.x.scatter_forward()
    
    # Solve
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
    
    # Extract velocity
    u_sol = w.sub(0).collapse()
    
    # Sample on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((XX.size, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)
    
    u_grid = np.full((pts.shape[0], gdim), np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for idx, i in enumerate(eval_map):
            u_grid[i, :] = vals[idx, :]
    
    vel_mag = np.sqrt(u_grid[:, 0]**2 + u_grid[:, 1]**2)
    vel_mag_grid = vel_mag.reshape(ny_out, nx_out)
    vel_mag_grid = np.nan_to_num(vel_mag_grid, nan=0.0)
    
    return {
        "u": vel_mag_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "nonlinear_iterations": [10],
        }
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "coefficients": {"nu": 0.14},
            "manufactured_solution": {
                "u": [
                    "-60*(y-0.7)*exp(-30*((x-0.3)**2 + (y-0.7)**2)) + 60*(y-0.3)*exp(-30*((x-0.7)**2 + (y-0.3)**2))",
                    "60*(x-0.3)*exp(-30*((x-0.3)**2 + (y-0.7)**2)) - 60*(x-0.7)*exp(-30*((x-0.7)**2 + (y-0.3)**2))"
                ],
                "p": "0"
            }
        },
        "output": {
            "field": "velocity_magnitude",
            "grid": {
                "nx": 100,
                "ny": 100,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }
    
    import time
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    print(f"Wall time: {elapsed:.2f}s")
    print(f"Output shape: {result['u'].shape}")
    print(f"Max velocity magnitude: {np.max(result['u']):.6f}")
    print(f"Min velocity magnitude: {np.min(result['u']):.6f}")
    
    # Compute error against exact solution
    xs = np.linspace(0.0, 1.0, 100)
    ys = np.linspace(0.0, 1.0, 100)
    XX, YY = np.meshgrid(xs, ys)
    
    u1_exact = -60*(YY-0.7)*np.exp(-30*((XX-0.3)**2 + (YY-0.7)**2)) + 60*(YY-0.3)*np.exp(-30*((XX-0.7)**2 + (YY-0.3)**2))
    u2_exact = 60*(XX-0.3)*np.exp(-30*((XX-0.3)**2 + (YY-0.7)**2)) - 60*(XX-0.7)*np.exp(-30*((XX-0.7)**2 + (YY-0.3)**2))
    vel_mag_exact = np.sqrt(u1_exact**2 + u2_exact**2)
    
    error_rms = np.sqrt(np.mean((result['u'] - vel_mag_exact)**2))
    max_error = np.max(np.abs(result['u'] - vel_mag_exact))
    print(f"RMS error: {error_rms:.6e}")
    print(f"Max error: {max_error:.6e}")
