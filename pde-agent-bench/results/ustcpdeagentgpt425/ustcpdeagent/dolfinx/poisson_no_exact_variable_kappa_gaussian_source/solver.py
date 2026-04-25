import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

DIAGNOSIS = """DIAGNOSIS
equation_type: poisson
spatial_dim: 2
domain_geometry: rectangle
unknowns: scalar
coupling: none
linearity: linear
time_dependence: steady
stiffness: N/A
dominant_physics: diffusion
peclet_or_reynolds: N/A
solution_regularity: smooth
bc_type: all_dirichlet
special_notes: variable_coeff
"""

METHOD = """METHOD
spatial_method: fem
element_or_basis: Lagrange_P2
stabilization: none
time_method: none
nonlinear_solver: none
linear_solver: direct_lu
preconditioner: none
special_treatment: none
pde_skill: poisson
"""


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack(
        [xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)]
    )

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    local_values = np.full(pts.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_ids = []

    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_ids.append(i)

    if points_on_proc:
        vals = uh.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        vals = np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0]
        local_values[np.array(eval_ids, dtype=np.int32)] = vals

    gathered = np.empty_like(local_values)
    domain.comm.Allreduce(np.nan_to_num(local_values, nan=0.0), gathered, op=MPI.SUM)
    return gathered.reshape(ny, nx)


def _solve_poisson(mesh_resolution, degree, ksp_type, pc_type, rtol):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    kappa = 1.0 + 50.0 * ufl.exp(-150.0 * ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2))
    f = ufl.exp(-250.0 * ((x[0] - 0.4) ** 2 + (x[1] - 0.6) ** 2))

    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), bdofs, V)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="poisson_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    solver = problem.solver
    try:
        iterations = int(solver.getIterationNumber())
    except Exception:
        iterations = 0

    return domain, uh, {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(degree),
        "ksp_type": str(solver.getType()),
        "pc_type": str(solver.getPC().getType()),
        "rtol": float(rtol),
        "iterations": iterations,
    }


def _accuracy_verification():
    probe_grid = {"nx": 65, "ny": 65, "bbox": [0.0, 1.0, 0.0, 1.0]}
    domain_c, uh_c, _ = _solve_poisson(28, 2, "preonly", "lu", 1e-10)
    domain_f, uh_f, _ = _solve_poisson(40, 2, "preonly", "lu", 1e-10)
    uc = _sample_function_on_grid(domain_c, uh_c, probe_grid)
    uf = _sample_function_on_grid(domain_f, uh_f, probe_grid)
    return float(np.sqrt(np.mean((uf - uc) ** 2)))


def solve(case_spec: dict) -> dict:
    indicator = _accuracy_verification()

    if indicator < 1.0e-3:
        mesh_resolution = 72
    else:
        mesh_resolution = 88

    try:
        domain, uh, solver_info = _solve_poisson(
            mesh_resolution, 2, "preonly", "lu", 1e-10
        )
    except Exception:
        domain, uh, solver_info = _solve_poisson(
            64, 2, "cg", "hypre", 1e-9
        )

    u_grid = _sample_function_on_grid(domain, uh, case_spec["output"]["grid"])
    solver_info["accuracy_verification"] = {
        "type": "coarse_fine_grid_rms_difference",
        "value": indicator,
    }
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case = {"output": {"grid": {"nx": 32, "ny": 24, "bbox": [0.0, 1.0, 0.0, 1.0]}}}
    out = solve(case)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
