import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import basix.ufl
import ufl
from petsc4py import PETSc
import time


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    nu_val = case_spec.get("pde", {}).get("viscosity", 5.0)
    nx_out = case_spec.get("output", {}).get("nx", 100)
    ny_out = case_spec.get("output", {}).get("ny", 100)

    # Use P3/P2 Taylor-Hood on moderate mesh for high accuracy
    N = 48
    deg_u = 3
    deg_p = 2
    result = _solve_stokes(comm, N, nu_val, nx_out, ny_out, deg_u, deg_p)

    return {
        "u": result["u_grid"],
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": deg_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": result.get("iterations", 1),
        }
    }


def _solve_stokes(comm, N, nu_val, nx_out, ny_out, deg_u=2, deg_p=1):
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    gdim = domain.geometry.dim

    cell_name = domain.topology.cell_name()
    Pu = basix.ufl.element("Lagrange", cell_name, deg_u, shape=(gdim,))
    Pp = basix.ufl.element("Lagrange", cell_name, deg_p)
    TH = basix.ufl.mixed_element([Pu, Pp])
    W = fem.functionspace(domain, TH)

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))

    u_exact = ufl.as_vector([
        pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
        -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    ])

    f = ufl.as_vector([
        2.0 * nu_val * pi**3 * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0])
        - pi * ufl.sin(pi * x[0]) * ufl.cos(pi * x[1]),
        -2.0 * nu_val * pi**3 * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
        - pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    ])

    a = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )
    L = ufl.inner(f, v) * ufl.dx

    V_sub, V_sub_map = W.sub(0).collapse()
    u_bc_func = fem.Function(V_sub)
    u_bc_func.interpolate(fem.Expression(u_exact, V_sub.element.interpolation_points))

    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs_u = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    Q_sub, Q_sub_map = W.sub(1).collapse()
    def corner(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    corner_facets = mesh.locate_entities_boundary(domain, fdim, corner)
    if len(corner_facets) > 0:
        dofs_p = fem.locate_dofs_topological((W.sub(1), Q_sub), fdim, corner_facets)
        p_bc_func = fem.Function(Q_sub)
        p_bc_func.interpolate(lambda x: np.full(x.shape[1], 1.0))
        bc_p = fem.dirichletbc(p_bc_func, dofs_p, W.sub(1))
        bcs = [bc_u, bc_p]
    else:
        bcs = [bc_u]

    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options_prefix="stokes_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )
    wh = problem.solve()

    u_sol = wh.sub(0).collapse()

    u_norm_sq = fem.assemble_scalar(fem.form(ufl.inner(u_sol, u_sol) * ufl.dx))
    u_norm = np.sqrt(comm.allreduce(u_norm_sq, op=MPI.SUM))

    u_grid = _evaluate_on_grid(domain, u_sol, nx_out, ny_out)

    return {"u_grid": u_grid, "norm": u_norm, "iterations": 1}


def _evaluate_on_grid(domain, u_func, nx, ny):
    xv = np.linspace(0, 1, nx)
    yv = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xv, yv, indexing='ij')

    points = np.zeros((3, nx * ny))
    points[0, :] = XX.flatten()
    points[1, :] = YY.flatten()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    u_values = np.full((nx * ny,), np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_arr, cells_arr)
        vel_mag = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, map_idx in enumerate(eval_map):
            u_values[map_idx] = vel_mag[idx]

    return u_values.reshape((nx, ny))


if __name__ == "__main__":
    case_spec = {
        "pde": {"viscosity": 5.0, "type": "stokes"},
        "domain": {"type": "unit_square"},
        "output": {"nx": 100, "ny": 100},
    }

    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0

    print(f"Solve time: {elapsed:.3f}s")
    print(f"solver_info: {result['solver_info']}")

    xv = np.linspace(0, 1, 100)
    yv = np.linspace(0, 1, 100)
    XX, YY = np.meshgrid(xv, yv, indexing='ij')
    u1_exact = np.pi * np.cos(np.pi * YY) * np.sin(np.pi * XX)
    u2_exact = -np.pi * np.cos(np.pi * XX) * np.sin(np.pi * YY)
    vel_mag_exact = np.sqrt(u1_exact**2 + u2_exact**2)

    error = np.sqrt(np.nanmean((result['u'] - vel_mag_exact)**2))
    print(f"RMS error: {error:.2e}")
    max_err = np.nanmax(np.abs(result['u'] - vel_mag_exact))
    print(f"Max error: {max_err:.2e}")
    
    # Check NaN count
    nan_count = np.sum(np.isnan(result['u']))
    print(f"NaN count: {nan_count}")
