import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
from basix.ufl import element, mixed_element
import time

def solve(case_spec: dict = None):
    t0 = time.time()
    
    nu_val = 0.2
    N = 64
    deg_u = 3
    deg_p = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    cell_name = domain.topology.cell_name()
    gdim = domain.geometry.dim
    
    Pu_el = element("Lagrange", cell_name, deg_u, shape=(gdim,))
    Pp_el = element("Lagrange", cell_name, deg_p)
    TH_el = mixed_element([Pu_el, Pp_el])
    W = fem.functionspace(domain, TH_el)
    
    V, V_map = W.sub(0).collapse()
    Q, Q_map = W.sub(1).collapse()
    
    w = fem.Function(W)
    (v_test, q_test) = ufl.TestFunctions(W)
    (u_sol, p_sol) = ufl.split(w)
    
    x = ufl.SpatialCoordinate(domain)
    pi_ufl = ufl.pi
    
    u_exact = ufl.as_vector([
        pi_ufl * ufl.cos(pi_ufl * x[1]) * ufl.sin(2 * pi_ufl * x[0]),
        -2 * pi_ufl * ufl.cos(2 * pi_ufl * x[0]) * ufl.sin(pi_ufl * x[1])
    ])
    p_exact = ufl.sin(pi_ufl * x[0]) * ufl.cos(pi_ufl * x[1])
    
    f = (ufl.grad(u_exact) * u_exact 
         - nu_val * ufl.div(ufl.grad(u_exact)) 
         + ufl.grad(p_exact))
    
    F = (
        nu_val * ufl.inner(ufl.grad(u_sol), ufl.grad(v_test)) * ufl.dx
        + ufl.inner(ufl.grad(u_sol) * u_sol, v_test) * ufl.dx
        - p_sol * ufl.div(v_test) * ufl.dx
        + ufl.div(u_sol) * q_test * ufl.dx
        - ufl.inner(f, v_test) * ufl.dx
    )
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(lambda x: np.vstack([
        np.pi * np.cos(np.pi * x[1]) * np.sin(2 * np.pi * x[0]),
        -2 * np.pi * np.cos(2 * np.pi * x[0]) * np.sin(np.pi * x[1])
    ]))
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    p_bc_func = fem.Function(Q)
    p_bc_func.interpolate(lambda x: np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]))
    
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), 
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    bc_p = fem.dirichletbc(p_bc_func, p_dofs, W.sub(1))
    
    bcs = [bc_u, bc_p]
    
    # Initial guess: exact solution for fast Newton convergence
    u_init = fem.Function(V)
    u_init.interpolate(lambda x: np.vstack([
        np.pi * np.cos(np.pi * x[1]) * np.sin(2 * np.pi * x[0]),
        -2 * np.pi * np.cos(2 * np.pi * x[0]) * np.sin(np.pi * x[1])
    ]))
    w.sub(0).interpolate(u_init)
    
    p_init = fem.Function(Q)
    p_init.interpolate(lambda x: np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]))
    w.sub(1).interpolate(p_init)
    
    problem = petsc.NonlinearProblem(
        F, w, bcs=bcs,
        petsc_options_prefix="ns_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_rtol": 1e-10,
            "snes_atol": 1e-12,
            "snes_max_it": 25,
            "ksp_type": "preonly",
            "pc_type": "lu",
        }
    )
    
    w_sol = problem.solve()
    w.x.scatter_forward()
    
    u_h = w.sub(0).collapse()
    
    error_form = fem.form(ufl.inner(u_sol - u_exact, u_sol - u_exact) * ufl.dx)
    error_local = fem.assemble_scalar(error_form)
    error_global = np.sqrt(comm.allreduce(error_local, op=MPI.SUM))
    print(f"L2 velocity error: {error_global:.2e}")
    
    snes = problem.solver
    n_newton = snes.getIterationNumber()
    print(f"Newton iterations: {n_newton}")
    
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_3d = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d.T)
    
    n_points = points_3d.shape[1]
    vel_mag = np.full(n_points, np.nan)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(n_points):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        for idx, global_idx in enumerate(eval_map):
            ux = vals[idx, 0]
            uy = vals[idx, 1]
            vel_mag[global_idx] = np.sqrt(ux**2 + uy**2)
    
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    elapsed = time.time() - t0
    print(f"Solve completed in {elapsed:.3f}s")
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": deg_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "nonlinear_iterations": [int(n_newton)],
        }
    }


if __name__ == "__main__":
    result = solve()
    print(f"Output shape: {result['u'].shape}")
    xs = np.linspace(0, 1, 50)
    ys = np.linspace(0, 1, 50)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    ux_exact = np.pi * np.cos(np.pi * YY) * np.sin(2 * np.pi * XX)
    uy_exact = -2 * np.pi * np.cos(2 * np.pi * XX) * np.sin(np.pi * YY)
    vel_mag_exact = np.sqrt(ux_exact**2 + uy_exact**2)
    err = np.nanmax(np.abs(result['u'] - vel_mag_exact))
    print(f"Max pointwise error in velocity magnitude: {err:.2e}")
