from __future__ import annotations

import math
import time
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, geometry, mesh
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType

DIAGNOSIS = "```DIAGNOSIS\nequation_type: poisson\nspatial_dim: 2\ndomain_geometry: rectangle\nunknowns: scalar\ncoupling: none\nlinearity: linear\ntime_dependence: steady\nstiffness: N/A\ndominant_physics: diffusion\npeclet_or_reynolds: N/A\nsolution_regularity: smooth\nbc_type: all_dirichlet\nspecial_notes: manufactured_solution, variable_coeff\n```"

METHOD = "```METHOD\nspatial_method: fem\nelement_or_basis: Lagrange_P2\nstabilization: none\ntime_method: none\nnonlinear_solver: none\nlinear_solver: cg\npreconditioner: amg\nspecial_treatment: none\npde_skill: poisson\n```"


def _sample_function(u_fun: fem.Function, bbox, nx: int, ny: int) -> np.ndarray:
    msh = u_fun.function_space.mesh
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    xx, yy = np.meshgrid(xs, ys, indexing='xy')
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny, dtype=np.float64)])

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

    vals = np.zeros(pts.shape[0], dtype=np.float64)
    mask = np.zeros(pts.shape[0], dtype=np.int32)
    if local_points:
        eval_vals = u_fun.eval(np.asarray(local_points, dtype=np.float64),
                               np.asarray(local_cells, dtype=np.int32))
        vals[np.asarray(local_ids, dtype=np.int32)] = np.asarray(eval_vals, dtype=np.float64).reshape(-1)
        mask[np.asarray(local_ids, dtype=np.int32)] = 1

    gvals = np.zeros_like(vals)
    gmask = np.zeros_like(mask)
    msh.comm.Allreduce(vals, gvals, op=MPI.SUM)
    msh.comm.Allreduce(mask, gmask, op=MPI.SUM)
    if np.any(gmask == 0):
        raise RuntimeError('Some sampling points were not found in any local cell.')
    return gvals.reshape(ny, nx)


def _u_exact_numpy(X):
    return np.sin(2.0 * math.pi * X[0]) * np.sin(math.pi * X[1])


def _solve_once(n: int, degree: int, ksp_type: str, pc_type: str, rtol: float):
    comm = MPI.COMM_WORLD
    msh = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0], dtype=np.float64), np.array([1.0, 1.0], dtype=np.float64)],
        [n, n],
        cell_type=mesh.CellType.quadrilateral,
    )
    V = fem.functionspace(msh, ('Lagrange', degree))
    x = ufl.SpatialCoordinate(msh)

    u_exact_ufl = ufl.sin(2.0 * math.pi * x[0]) * ufl.sin(math.pi * x[1])
    kappa = 1.0 + 0.5 * ufl.cos(2.0 * math.pi * x[0]) * ufl.cos(2.0 * math.pi * x[1])
    f_ufl = -ufl.div(kappa * ufl.grad(u_exact_ufl))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(_u_exact_numpy)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    b = petsc.create_vector(L_form.function_spaces)
    with b.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    uh = fem.Function(V)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol, atol=1e-14, max_it=1000)

    try:
        solver.solve(b, uh.x.petsc_vec)
        if solver.getConvergedReason() <= 0:
            raise RuntimeError(f'KSP failed with reason {solver.getConvergedReason()}')
    except Exception:
        solver.destroy()
        solver = PETSc.KSP().create(comm)
        solver.setOperators(A)
        solver.setType('preonly')
        solver.getPC().setType('lu')
        solver.solve(b, uh.x.petsc_vec)

    uh.x.scatter_forward()

    u_exact = fem.Function(V)
    u_exact.interpolate(_u_exact_numpy)
    e = fem.Function(V)
    e.x.array[:] = uh.x.array - u_exact.x.array
    e.x.scatter_forward()

    err_sq = fem.assemble_scalar(fem.form(e * e * ufl.dx))
    exact_sq = fem.assemble_scalar(fem.form(u_exact * u_exact * ufl.dx))
    err_sq = comm.allreduce(err_sq, op=MPI.SUM)
    exact_sq = comm.allreduce(exact_sq, op=MPI.SUM)
    err_L2 = math.sqrt(err_sq)
    rel_L2 = err_L2 / max(math.sqrt(exact_sq), 1e-16)

    info = {
        'mesh_resolution': int(n),
        'element_degree': int(degree),
        'ksp_type': str(solver.getType()),
        'pc_type': str(solver.getPC().getType()),
        'rtol': float(rtol),
        'iterations': int(solver.getIterationNumber()),
        'L2_error': float(err_L2),
        'relative_L2_error': float(rel_L2),
    }
    return uh, info


def solve(case_spec: dict) -> dict:
    grid = case_spec['output']['grid']
    nx = int(grid['nx'])
    ny = int(grid['ny'])
    bbox = grid['bbox']

    candidates = [(16, 2), (24, 2), (32, 2)]
    soft_budget = 0.7
    best_u = None
    best_info = None
    start = time.perf_counter()

    for i, (n, degree) in enumerate(candidates):
        iter_start = time.perf_counter()
        uh, info = _solve_once(n, degree, ksp_type='cg', pc_type='hypre', rtol=1e-10)
        best_u, best_info = uh, info
        elapsed = time.perf_counter() - start
        iter_elapsed = time.perf_counter() - iter_start
        if info['L2_error'] < 1.0e-3 and elapsed > 0.20:
            break
        if elapsed > soft_budget:
            break
        if i == 0 and iter_elapsed > 0.45:
            break

    u_grid = _sample_function(best_u, bbox, nx, ny)
    return {
        'u': u_grid,
        'solver_info': {
            'mesh_resolution': best_info['mesh_resolution'],
            'element_degree': best_info['element_degree'],
            'ksp_type': best_info['ksp_type'],
            'pc_type': best_info['pc_type'],
            'rtol': best_info['rtol'],
            'iterations': best_info['iterations'],
            'accuracy_verification': {
                'manufactured_solution': 'u = sin(2*pi*x)*sin(pi*y)',
                'L2_error': best_info['L2_error'],
                'relative_L2_error': best_info['relative_L2_error'],
            },
        },
    }


if __name__ == '__main__':
    case_spec = {
        'pde': {'time': None},
        'output': {'grid': {'nx': 64, 'ny': 64, 'bbox': [0.0, 1.0, 0.0, 1.0]}},
    }
    t0 = time.perf_counter()
    result = solve(case_spec)
    wall = time.perf_counter() - t0
    if MPI.COMM_WORLD.rank == 0:
        print(DIAGNOSIS)
        print(METHOD)
        print(f"L2_ERROR: {result['solver_info']['accuracy_verification']['L2_error']:.12e}")
        print(f"WALL_TIME: {wall:.12e}")
        print(f"OUTPUT_SHAPE: {result['u'].shape}")
