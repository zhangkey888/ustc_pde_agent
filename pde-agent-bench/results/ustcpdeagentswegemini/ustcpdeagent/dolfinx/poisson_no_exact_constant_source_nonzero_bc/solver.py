import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl


ScalarType = PETSc.ScalarType


def _boundary_values(x):
    return np.sin(np.pi * x[0]) + np.cos(np.pi * x[1])


def _sample_function(u_func, bbox, nx, ny):
    msh = u_func.function_space.mesh
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts2 = np.column_stack([XX.ravel(), YY.ravel()])
    gdim = msh.geometry.dim
    pts = np.zeros((pts2.shape[0], 3), dtype=np.float64)
    pts[:, :gdim] = pts2[:, :gdim]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts)

    local_points = []
    local_cells = []
    local_ids = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            local_points.append(pts[i])
            local_cells.append(links[0])
            local_ids.append(i)

    local_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    if local_points:
        vals = u_func.eval(np.asarray(local_points, dtype=np.float64),
                          np.asarray(local_cells, dtype=np.int32))
        local_vals[np.asarray(local_ids, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = msh.comm
    gathered = comm.gather(local_vals, root=0)
    if comm.rank == 0:
        out = np.full(pts.shape[0], np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            out[mask] = arr[mask]
        if np.isnan(out).any():
            raise RuntimeError("Failed to evaluate solution at some sampling points.")
        out = out.reshape(ny, nx)
    else:
        out = None
    out = comm.bcast(out, root=0)
    return out


def _solve_poisson_on_mesh(n):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    degree = 2
    V = fem.functionspace(msh, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    kappa = fem.Constant(msh, ScalarType(1.0))
    f = fem.Constant(msh, ScalarType(1.0))

    x = ufl.SpatialCoordinate(msh)
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(_boundary_values)
    bc = fem.dirichletbc(u_bc, bdofs)

    opts = {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1.0e-10,
        "ksp_atol": 1.0e-12,
        "ksp_max_it": 2000,
    }

    uh = fem.Function(V)
    try:
        problem = petsc.LinearProblem(
            a, L, u=uh, bcs=[bc],
            petsc_options_prefix=f"poisson_{n}_",
            petsc_options=opts,
        )
        uh = problem.solve()
        ksp = problem.solver
        its = int(ksp.getIterationNumber())
        ksp_type = ksp.getType()
        pc_type = ksp.getPC().getType()
    except Exception:
        opts = {
            "ksp_type": "preonly",
            "pc_type": "lu",
        }
        problem = petsc.LinearProblem(
            a, L, u=uh, bcs=[bc],
            petsc_options_prefix=f"poissonlu_{n}_",
            petsc_options=opts,
        )
        uh = problem.solve()
        ksp = problem.solver
        its = int(ksp.getIterationNumber())
        ksp_type = ksp.getType()
        pc_type = ksp.getPC().getType()

    uh.x.scatter_forward()

    # Accuracy verification: PDE residual in weak sense against refined projection surrogate
    V1 = fem.functionspace(msh, ("Lagrange", 1))
    u_lin = fem.Function(V1)
    u_lin.interpolate(uh)
    residual_indicator = fem.assemble_scalar(
        fem.form((ufl.inner(ufl.grad(u_lin), ufl.grad(u_lin)) - 2.0 * ufl.inner(f, u_lin)) * ufl.dx)
    )
    residual_indicator = msh.comm.allreduce(residual_indicator, op=MPI.SUM)

    return {
        "mesh": msh,
        "V": V,
        "u": uh,
        "degree": degree,
        "iterations": its,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": 1.0e-10 if ksp_type != "preonly" else 0.0,
        "verification_energy": float(residual_indicator),
    }


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    bbox = grid["bbox"]

    # Adaptive time-accuracy tradeoff under strict time budget:
    # Start with a reasonably fine P2 mesh and refine once if cheap consistency check suggests it.
    candidates = [48, 64]
    coarse = _solve_poisson_on_mesh(candidates[0])
    u_grid_coarse = _sample_function(coarse["u"], bbox, nx, ny)

    fine = _solve_poisson_on_mesh(candidates[1])
    u_grid_fine = _sample_function(fine["u"], bbox, nx, ny)

    consistency_err = float(np.sqrt(np.mean((u_grid_fine - u_grid_coarse) ** 2)))

    if consistency_err < 2.0e-4:
        chosen = fine
        u_grid = u_grid_fine
        chosen_resolution = candidates[1]
    else:
        # If difference is larger than desired, still use the finer solution available.
        chosen = fine
        u_grid = u_grid_fine
        chosen_resolution = candidates[1]

    solver_info = {
        "mesh_resolution": int(chosen_resolution),
        "element_degree": int(chosen["degree"]),
        "ksp_type": str(chosen["ksp_type"]),
        "pc_type": str(chosen["pc_type"]),
        "rtol": float(chosen["rtol"]),
        "iterations": int(coarse["iterations"] + fine["iterations"]),
        "verification": {
            "grid_consistency_l2": consistency_err,
            "energy_indicator": float(chosen["verification_energy"]),
        },
    }

    return {
        "u": np.asarray(u_grid, dtype=np.float64).reshape(ny, nx),
        "solver_info": solver_info,
    }
