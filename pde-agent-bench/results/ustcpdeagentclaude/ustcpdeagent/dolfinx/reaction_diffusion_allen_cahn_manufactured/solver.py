import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # Parameters
    comm = MPI.COMM_WORLD

    # Time parameters
    t0 = 0.0
    t_end = 0.15
    dt_val = 0.005  # smaller than suggested for accuracy
    n_steps = int(round((t_end - t0) / dt_val))

    # Mesh
    N = 64
    degree = 2
    epsilon = 1.0  # default diffusion coefficient

    # Try to read from case_spec if provided
    pde = case_spec.get("pde", {})
    if isinstance(pde, dict):
        eps_val = pde.get("epsilon", None)
        if eps_val is not None:
            epsilon = float(eps_val)
        time_info = pde.get("time", {})
        if isinstance(time_info, dict):
            t_end = float(time_info.get("t_end", t_end))
            t0 = float(time_info.get("t0", t0))

    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    t_const = fem.Constant(msh, PETSc.ScalarType(t0))
    dt_const = fem.Constant(msh, PETSc.ScalarType(dt_val))
    eps_const = fem.Constant(msh, PETSc.ScalarType(epsilon))

    # Manufactured solution: u_exact = exp(-t) * 0.3 * sin(pi*x) * sin(pi*y)
    def u_exact_ufl(tt):
        return ufl.exp(-tt) * 0.3 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # Reaction R(u): Allen-Cahn form R(u) = u^3 - u
    def R(u):
        return u**3 - u

    # Source f = du/dt - eps*lap(u) + R(u) evaluated at exact
    def f_ufl(tt):
        ue = u_exact_ufl(tt)
        # du/dt = -ue
        dudt = -ue
        # lap(ue) = -2*pi^2 * ue
        lap_ue = -2.0 * ufl.pi**2 * ue
        return dudt - epsilon * lap_ue + R(ue)

    # Previous solution
    u_n = fem.Function(V)
    # Initial condition: exact at t=t0
    u_init_expr = fem.Expression(u_exact_ufl(fem.Constant(msh, PETSc.ScalarType(t0))),
                                  V.element.interpolation_points)
    u_n.interpolate(u_init_expr)

    # Current solution
    u = fem.Function(V)
    u.x.array[:] = u_n.x.array[:]

    v = ufl.TestFunction(V)

    # Backward Euler: (u - u_n)/dt - eps*lap(u) + R(u) = f(t^{n+1})
    # Weak form:
    F = (ufl.inner((u - u_n)/dt_const, v)*ufl.dx
         + eps_const * ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
         + ufl.inner(R(u), v)*ufl.dx
         - ufl.inner(f_ufl(t_const), v)*ufl.dx)

    # Boundary condition
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact_ufl(t_const), V.element.interpolation_points)
    u_bc.interpolate(u_bc_expr)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    J = ufl.derivative(F, u)

    petsc_options = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-9,
        "snes_atol": 1e-11,
        "snes_max_it": 30,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }

    problem = petsc.NonlinearProblem(F, u, bcs=[bc], J=J,
                                      petsc_options_prefix="rd_",
                                      petsc_options=petsc_options)

    nonlinear_iterations = []
    # Time stepping
    t_current = t0
    for step in range(n_steps):
        t_current += dt_val
        t_const.value = t_current
        # update BC
        u_bc.interpolate(u_bc_expr)

        u_sol = problem.solve()
        u.x.scatter_forward()
        # copy to u_n
        u_n.x.array[:] = u.x.array[:]
        try:
            its = problem.solver.getIterationNumber()
        except Exception:
            its = 0
        nonlinear_iterations.append(int(its))

    # Sample on output grid
    out = case_spec["output"]["grid"]
    nx_out = out["nx"]
    ny_out = out["ny"]
    bbox = out["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out*ny_out)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    u_grid = np.full(nx_out*ny_out, np.nan)
    pts_on = []
    cells_on = []
    idx_on = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            pts_on.append(pts[i])
            cells_on.append(links[0])
            idx_on.append(i)
    if len(pts_on) > 0:
        vals = u.eval(np.array(pts_on), np.array(cells_on, dtype=np.int32))
        u_grid[idx_on] = vals.flatten()
    u_grid = u_grid.reshape(ny_out, nx_out)

    # Initial condition sampling for u_initial
    u_init_grid = np.full(nx_out*ny_out, np.nan)
    # Compute exact at t0 (which equals initial condition)
    t0_arr = t0
    for i, idx in enumerate(idx_on):
        px, py = pts_on[i][0], pts_on[i][1]
        u_init_grid[idx] = np.exp(-t0_arr) * 0.3 * np.sin(np.pi*px) * np.sin(np.pi*py)
    u_init_grid = u_init_grid.reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": 0,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
            "nonlinear_iterations": nonlinear_iterations,
        }
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.15}},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}}
    }
    t0_wall = time.time()
    res = solve(case_spec)
    t1_wall = time.time()
    print(f"Wall time: {t1_wall - t0_wall:.2f}s")
    print(f"u shape: {res['u'].shape}")
    # Compute error against exact solution at t_end
    nx, ny = 64, 64
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys)
    u_exact = np.exp(-0.15) * 0.3 * np.sin(np.pi*XX) * np.sin(np.pi*YY)
    err = np.sqrt(np.mean((res['u'] - u_exact)**2))
    max_err = np.max(np.abs(res['u'] - u_exact))
    print(f"L2 error (RMS): {err:.6e}")
    print(f"Max error: {max_err:.6e}")
    print(f"Nonlinear iterations: {res['solver_info']['nonlinear_iterations'][:5]}...")
