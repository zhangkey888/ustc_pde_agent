import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Extract parameters from case_spec
    k_val = case_spec["pde"]["helmholtz"]["k"]
    
    # Output grid
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    xmin, xmax, ymin, ymax = bbox

    # Mesh resolution and element degree
    mesh_res = 80
    degree = 3

    # Create mesh with quadrilaterals
    p0 = np.array([xmin, ymin])
    p1 = np.array([xmax, ymax])
    domain = mesh.create_rectangle(
        comm, [p0, p1], [mesh_res, mesh_res],
        cell_type=mesh.CellType.quadrilateral
    )

    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))

    # Manufactured solution: u = sin(2*pi*x)*cos(3*pi*y)
    x = ufl.SpatialCoordinate(domain)
    pi_val = ufl.pi
    k = fem.Constant(domain, ScalarType(k_val))

    u_exact = ufl.sin(2 * pi_val * x[0]) * ufl.cos(3 * pi_val * x[1])

    # Source term: f = -div(grad(u)) - k^2 * u
    f = -ufl.div(ufl.grad(u_exact)) - k**2 * u_exact

    # Variational formulation:
    # -nabla^2 u - k^2 u = f
    # Integration by parts: int(grad(u).grad(v)) dx - k^2 int(u*v) dx = int(f*v) dx
    u_trial = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx - k**2 * ufl.inner(u_trial, v) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    # Dirichlet BC
    tdim = domain.topology.dim
    fdim = tdim - 1

    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc.interpolate(u_bc_expr)

    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Solve with direct LU (robust for indefinite Helmholtz)
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-12

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
        },
        petsc_options_prefix="helmholtz_"
    )
    u_sol = problem.solve()

    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()

    # Compute L2 error for verification
    error_form = fem.form(ufl.inner(u_sol - u_exact, u_sol - u_exact) * ufl.dx)
    error_local = fem.assemble_scalar(error_form)
    error_global = np.sqrt(comm.allreduce(error_local, op=MPI.SUM))
    print(f"L2 error: {error_global:.6e}")

    # Sample solution onto output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((XX.size, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_grid = np.full(XX.size, np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(
            np.array(points_on_proc),
            np.array(cells_on_proc, dtype=np.int32)
        )
        u_grid[eval_map] = vals.flatten()

    u_grid = u_grid.reshape(ny_out, nx_out)

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": int(iterations),
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    import time
    case_spec = {
        "pde": {
            "helmholtz": {"k": 10.0}
        },
        "output": {
            "grid": {
                "nx": 100,
                "ny": 100,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }

    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Wall time: {t1 - t0:.2f}s")
    print(f"u shape: {result['u'].shape}")
    print(f"NaN count: {np.isnan(result['u']).sum()}")
    print(f"Solver info: {result['solver_info']}")

    # Verify against exact solution on grid
    xs = np.linspace(0, 1, 100)
    ys = np.linspace(0, 1, 100)
    XX, YY = np.meshgrid(xs, ys)
    u_exact_grid = np.sin(2 * np.pi * XX) * np.cos(3 * np.pi * YY)
    grid_error = np.sqrt(np.nanmean((result['u'] - u_exact_grid)**2))
    print(f"Grid RMS error: {grid_error:.6e}")
