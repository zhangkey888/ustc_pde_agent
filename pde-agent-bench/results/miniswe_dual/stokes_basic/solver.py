import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import basix.ufl
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parse parameters
    nu_val = 1.0
    if 'pde' in case_spec and 'viscosity' in case_spec['pde']:
        nu_val = float(case_spec['pde']['viscosity'])

    # Grid for output
    nx_out = case_spec.get('output', {}).get('nx', 100)
    ny_out = case_spec.get('output', {}).get('ny', 100)

    # Mesh resolution - P3/P2 Taylor-Hood converges faster
    N = 40
    deg_u = 3
    deg_p = 2

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim
    tdim = domain.topology.dim
    fdim = tdim - 1

    # Taylor-Hood P3/P2
    Pu_el = basix.ufl.element('Lagrange', domain.topology.cell_name(), deg_u, shape=(gdim,))
    Pp_el = basix.ufl.element('Lagrange', domain.topology.cell_name(), deg_p)
    ME = basix.ufl.mixed_element([Pu_el, Pp_el])
    W = fem.functionspace(domain, ME)

    # Trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi_ = ufl.pi

    # Manufactured solution
    u_exact = ufl.as_vector([
        pi_ * ufl.cos(pi_ * x[1]) * ufl.sin(pi_ * x[0]),
        -pi_ * ufl.cos(pi_ * x[0]) * ufl.sin(pi_ * x[1])
    ])
    p_exact = ufl.cos(pi_ * x[0]) * ufl.cos(pi_ * x[1])

    # Source term: f = -nu * laplacian(u_exact) + grad(p_exact)
    f = -nu_val * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)

    # Viscosity
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))

    # Bilinear form: Stokes
    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + q * ufl.div(u) * ufl.dx)

    # Linear form
    L = ufl.inner(f, v) * ufl.dx

    # Boundary conditions: u = u_exact on all boundary
    V_sub, _ = W.sub(0).collapse()

    u_bc_func = fem.Function(V_sub)
    u_exact_expr = fem.Expression(u_exact, V_sub.element.interpolation_points)
    u_bc_func.interpolate(u_exact_expr)

    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    dofs_bc = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_bc, W.sub(0))

    bcs = [bc_u]

    # Solve with direct solver (LU/MUMPS)
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-12

    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="stokes_"
    )
    wh = problem.solve()

    # Extract velocity
    uh = wh.sub(0).collapse()

    # Evaluate on output grid
    xg = np.linspace(0, 1, nx_out)
    yg = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xg, yg, indexing='ij')

    points = np.zeros((nx_out * ny_out, 3))
    points[:, 0] = XX.flatten()
    points[:, 1] = YY.flatten()

    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    vel_mag = np.full(nx_out * ny_out, np.nan)

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
        vals = uh.eval(pts_arr, cells_arr)
        vel_magnitude = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            vel_mag[global_idx] = vel_magnitude[idx]

    u_grid = vel_mag.reshape((nx_out, ny_out))

    solver_info = {
        "mesh_resolution": N,
        "element_degree": deg_u,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": 1,
    }

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    import time

    case_spec = {
        "pde": {
            "viscosity": 1.0,
        },
        "output": {
            "nx": 100,
            "ny": 100,
        }
    }

    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0

    print(f"Solve time: {elapsed:.3f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{np.nanmin(result['u']):.6f}, {np.nanmax(result['u']):.6f}]")
    print(f"Any NaN: {np.any(np.isnan(result['u']))}")

    # Compute error against exact solution
    nx_out, ny_out = 100, 100
    xg = np.linspace(0, 1, nx_out)
    yg = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xg, yg, indexing='ij')

    u1_exact = np.pi * np.cos(np.pi * YY) * np.sin(np.pi * XX)
    u2_exact = -np.pi * np.cos(np.pi * XX) * np.sin(np.pi * YY)
    vel_mag_exact = np.sqrt(u1_exact**2 + u2_exact**2)

    rms_error = np.sqrt(np.mean((result['u'] - vel_mag_exact)**2))
    max_error = np.max(np.abs(result['u'] - vel_mag_exact))
    rel_l2 = np.sqrt(np.sum((result['u'] - vel_mag_exact)**2) / np.sum(vel_mag_exact**2))

    print(f"RMS error: {rms_error:.2e}")
    print(f"Max error: {max_error:.2e}")
    print(f"Rel L2 error: {rel_l2:.2e}")
    print(f"Solver info: {result['solver_info']}")
