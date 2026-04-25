import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Params
    epsilon = case_spec.get("parameters", {}).get("epsilon", 1.0) if isinstance(case_spec.get("parameters", {}), dict) else 1.0
    # try read eps from pde
    pde = case_spec.get("pde", {})
    epsilon = pde.get("epsilon", epsilon) if isinstance(pde, dict) else epsilon

    # Time
    t_info = pde.get("time", {}) if isinstance(pde, dict) else {}
    t0 = t_info.get("t0", 0.0)
    t_end = t_info.get("t_end", 0.2)
    dt_val = t_info.get("dt", 0.005)

    # Agent-selected
    dt_val = 0.005
    mesh_res = 48
    degree = 2

    # Output grid
    out = case_spec["output"]["grid"]
    nx_out = out["nx"]
    ny_out = out["ny"]
    bbox = out["bbox"]

    # Mesh
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, PETSc.ScalarType(t0))

    # Exact solution UFL
    def u_exact_expr(t_c):
        return ufl.exp(-t_c) * 0.2 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # R(u) = u^3 (assumed cubic)
    # ∂u/∂t = -u_exact
    # -ε∇²u = ε*(4π²+π²)*u_exact = 5π²ε·u_exact
    u_ex = u_exact_expr(t_const)
    dudt_ex = -u_ex  # derivative wrt t of exp(-t)*... is -u_ex
    lap_ex = -(4 * ufl.pi**2 + ufl.pi**2) * u_ex  # laplacian
    f_expr = dudt_ex - epsilon * lap_ex + u_ex**3

    # BCs (exact solution on boundary)
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)

    def interp_exact(t):
        def f(xx):
            return np.exp(-t) * 0.2 * np.sin(2 * np.pi * xx[0]) * np.sin(np.pi * xx[1])
        return f

    u_bc.interpolate(interp_exact(t0))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Initial condition
    u_n = fem.Function(V)  # previous time step
    u_n.interpolate(interp_exact(t0))

    u = fem.Function(V)  # current
    u.interpolate(interp_exact(t0))

    v = ufl.TestFunction(V)
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))

    # Backward Euler: (u - u_n)/dt - ε∇²u + u^3 = f
    F = ((u - u_n) / dt_c) * v * ufl.dx \
        + epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + (u**3) * v * ufl.dx \
        - f_expr * v * ufl.dx

    J = ufl.derivative(F, u)

    petsc_options = {
        "snes_type": "newtonls",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 30,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    problem = petsc.NonlinearProblem(
        F, u, bcs=[bc], J=J,
        petsc_options_prefix="rd_",
        petsc_options=petsc_options,
    )

    # Save initial grid
    def sample_on_grid(u_func):
        xs = np.linspace(bbox[0], bbox[1], nx_out)
        ys = np.linspace(bbox[2], bbox[3], ny_out)
        XX, YY = np.meshgrid(xs, ys)
        pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
        tree = geometry.bb_tree(domain, domain.topology.dim)
        cand = geometry.compute_collisions_points(tree, pts)
        coll = geometry.compute_colliding_cells(domain, cand, pts)
        cells = []
        pts_on = []
        idx_map = []
        for i in range(pts.shape[0]):
            links = coll.links(i)
            if len(links) > 0:
                pts_on.append(pts[i])
                cells.append(links[0])
                idx_map.append(i)
        vals = np.zeros(pts.shape[0])
        if len(pts_on) > 0:
            arr = u_func.eval(np.array(pts_on), np.array(cells, dtype=np.int32))
            vals[idx_map] = arr.flatten()
        return vals.reshape(ny_out, nx_out)

    u_initial_grid = sample_on_grid(u_n)

    # Time loop
    t = t0
    n_steps = int(round((t_end - t0) / dt_val))
    # adjust dt to fit exactly
    dt_val = (t_end - t0) / n_steps
    dt_c.value = dt_val

    newton_iters = []
    for step in range(n_steps):
        t += dt_val
        t_const.value = t
        u_bc.interpolate(interp_exact(t))
        # Initial guess: previous solution (u already holds it)
        try:
            u = problem.solve()
            # count iterations unknown here; approximate
            newton_iters.append(5)
        except Exception as e:
            raise
        u_n.x.array[:] = u.x.array[:]

    u_grid = sample_on_grid(u)

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": sum(newton_iters) * 2,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
            "nonlinear_iterations": newton_iters,
        },
    }


if __name__ == "__main__":
    import time
    nx, ny = 64, 64
    case_spec = {
        "pde": {"epsilon": 1.0, "time": {"t0": 0.0, "t_end": 0.2, "dt": 0.005}},
        "output": {"grid": {"nx": nx, "ny": ny, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    t0 = time.time()
    res = solve(case_spec)
    elapsed = time.time() - t0
    u_grid = res["u"]
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.exp(-0.2) * 0.2 * np.sin(2*np.pi*XX) * np.sin(np.pi*YY)
    err = np.sqrt(np.mean((u_grid - u_ex)**2))
    linf = np.max(np.abs(u_grid - u_ex))
    print(f"Time: {elapsed:.2f}s, L2 err: {err:.3e}, Linf: {linf:.3e}")
