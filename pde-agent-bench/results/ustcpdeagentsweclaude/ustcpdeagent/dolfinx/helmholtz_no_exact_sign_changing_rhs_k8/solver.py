# ```DIAGNOSIS
# equation_type: helmholtz
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: steady
# stiffness: N/A
# dominant_physics: wave
# peclet_or_reynolds: N/A
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: sign_changing_rhs
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: gmres
# preconditioner: ilu
# special_treatment: none
# pde_skill: helmholtz
# ```

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc


ScalarType = PETSc.ScalarType


def _build_and_solve(n, degree=2, k=8.0, ksp_type="gmres", pc_type="ilu", rtol=1e-9):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    f_expr = ufl.cos(4 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - ScalarType(k * k) * u * v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix=f"helm_{n}_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-12,
            "ksp_max_it": 5000,
        },
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    # Access underlying KSP if available; otherwise fill conservative defaults
    iterations = -1
    actual_ksp = ksp_type
    actual_pc = pc_type
    try:
        solver = problem.solver
        iterations = int(solver.getIterationNumber())
        actual_ksp = solver.getType()
        actual_pc = solver.getPC().getType()
    except Exception:
        pass

    return domain, V, uh, {
        "mesh_resolution": n,
        "element_degree": degree,
        "ksp_type": str(actual_ksp),
        "pc_type": str(actual_pc),
        "rtol": float(rtol),
        "iterations": int(max(iterations, 0)),
    }


def _sample_function(u_func, nx, ny, bbox):
    domain = u_func.function_space.mesh
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        vals = np.real(np.asarray(vals).reshape(-1))
        values[np.array(idx_map, dtype=np.int32)] = vals

    comm = domain.comm
    gathered = comm.allreduce(np.nan_to_num(values, nan=0.0), op=MPI.SUM)
    return gathered.reshape((ny, nx))


def _verify_accuracy(target_bbox=(0.0, 1.0, 0.0, 1.0), sample_n=81):
    # Mesh convergence surrogate: compare n and 2n solutions on a common grid
    n1, n2 = 48, 96
    _, _, u1, _ = _build_and_solve(n1)
    _, _, u2, _ = _build_and_solve(n2)
    g1 = _sample_function(u1, sample_n, sample_n, target_bbox)
    g2 = _sample_function(u2, sample_n, sample_n, target_bbox)
    diff = g2 - g1
    l2_grid = float(np.sqrt(np.mean(diff**2)))
    linf_grid = float(np.max(np.abs(diff)))
    return {"grid_l2_diff_coarse_fine": l2_grid, "grid_linf_diff_coarse_fine": linf_grid}


def solve(case_spec: dict) -> dict:
    output_grid = case_spec["output"]["grid"]
    nx = int(output_grid["nx"])
    ny = int(output_grid["ny"])
    bbox = output_grid["bbox"]
    k = float(case_spec.get("pde", {}).get("k", case_spec.get("wavenumber", 8.0)))

    # Adaptive accuracy/time trade-off: use a fairly fine P2 mesh within the time budget.
    mesh_resolution = 96
    degree = 2
    rtol = 1e-9
    ksp_type = "gmres"
    pc_type = "ilu"

    try:
        _, _, uh, solver_info = _build_and_solve(
            mesh_resolution, degree=degree, k=k, ksp_type=ksp_type, pc_type=pc_type, rtol=rtol
        )
    except Exception:
        _, _, uh, solver_info = _build_and_solve(
            mesh_resolution, degree=degree, k=k, ksp_type="preonly", pc_type="lu", rtol=rtol
        )

    u_grid = _sample_function(uh, nx, ny, bbox)

    # Required accuracy verification module
    verification = _verify_accuracy(tuple(bbox), sample_n=min(max(nx, ny), 81))
    solver_info.update(verification)

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"k": 8.0, "time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    print(out["u"].shape)
    print(out["solver_info"])
