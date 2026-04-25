import time
import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

# ```DIAGNOSIS
# equation_type: heat
# spatial_dim: 2
# domain_geometry: rectangle
# unknowns: scalar
# coupling: none
# linearity: linear
# time_dependence: transient
# stiffness: stiff
# dominant_physics: diffusion
# peclet_or_reynolds: N/A
# solution_regularity: smooth
# bc_type: all_dirichlet
# special_notes: manufactured_solution
# ```
#
# ```METHOD
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: backward_euler
# nonlinear_solver: none
# linear_solver: cg
# preconditioner: hypre
# special_treatment: none
# pde_skill: heat
# ```

ScalarType = PETSc.ScalarType


def _make_case_defaults(case_spec: dict) -> dict:
    cs = dict(case_spec) if case_spec is not None else {}
    cs.setdefault("output", {})
    cs["output"].setdefault("grid", {})
    cs["output"]["grid"].setdefault("nx", 64)
    cs["output"]["grid"].setdefault("ny", 64)
    cs["output"]["grid"].setdefault("bbox", [0.0, 1.0, 0.0, 1.0])
    cs.setdefault("pde", {})
    cs["pde"].setdefault("time", {})
    cs["pde"]["time"].setdefault("t0", 0.0)
    cs["pde"]["time"].setdefault("t_end", 0.06)
    cs["pde"]["time"].setdefault("dt", 0.003)
    return cs


def _manufactured_exact(x, t):
    return np.exp(-t) * np.sin(4.0 * np.pi * x[0]) * np.sin(4.0 * np.pi * x[1])


def _probe_function(u_func, points_array):
    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    ptsT = points_array.T
    cell_candidates = geometry.compute_collisions_points(tree, ptsT)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, ptsT)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_array.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(ptsT[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    values = np.full((points_array.shape[1],), np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64),
                           np.array(cells_on_proc, dtype=np.int32))
        vals = np.real(np.asarray(vals).reshape(-1))
        values[np.array(eval_map, dtype=np.int32)] = vals
    return values


def solve(case_spec: dict) -> dict:
    case_spec = _make_case_defaults(case_spec)
    comm = MPI.COMM_WORLD

    t0 = float(case_spec["pde"]["time"].get("t0", 0.0))
    t_end = float(case_spec["pde"]["time"].get("t_end", 0.06))
    dt_in = float(case_spec["pde"]["time"].get("dt", 0.003))

    # Use a more accurate dt if still modest cost.
    dt = min(dt_in, 0.002)
    n_steps = int(round((t_end - t0) / dt))
    dt = (t_end - t0) / n_steps if n_steps > 0 else dt_in

    mesh_resolution = 56
    degree = 2

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution,
                                     cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, ScalarType(t0))
    kappa = fem.Constant(domain, ScalarType(1.0))

    u_exact_ufl = ufl.exp(-t_const) * ufl.sin(4.0 * ufl.pi * x[0]) * ufl.sin(4.0 * ufl.pi * x[1])
    f_ufl = (-1.0 + 32.0 * ufl.pi * ufl.pi) * u_exact_ufl

    u_n = fem.Function(V)
    u_n.interpolate(lambda X: _manufactured_exact(X, t0))
    u_n.x.scatter_forward()

    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: _manufactured_exact(X, t0))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = (u * v + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt * f_ufl * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1.0e-10, atol=1.0e-12, max_it=5000)
    solver.setFromOptions()

    uh = fem.Function(V)
    total_iterations = 0
    t = t0

    grid = case_spec["output"]["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = grid["bbox"]
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    points = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    u_initial_grid = _probe_function(u_n, points).reshape(ny, nx)

    for _ in range(n_steps):
        t += dt
        t_const.value = ScalarType(t)
        u_bc.interpolate(lambda X, tt=t: _manufactured_exact(X, tt))
        u_bc.x.scatter_forward()

        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        its = solver.getIterationNumber()
        total_iterations += int(its)

        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    u_grid = _probe_function(uh, points).reshape(ny, nx)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree,
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "rtol": 1.0e-10,
        "iterations": int(total_iterations),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "time_scheme": "backward_euler",
    }

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {"time": {"t0": 0.0, "t_end": 0.06, "dt": 0.003}},
        "output": {"grid": {"nx": 81, "ny": 81, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }

    t_start = time.perf_counter()
    result = solve(case_spec)
    wall = time.perf_counter() - t_start

    nx = case_spec["output"]["grid"]["nx"]
    ny = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    u_exact = np.exp(-0.06) * np.sin(4.0 * np.pi * XX) * np.sin(4.0 * np.pi * YY)
    err = np.sqrt(np.mean((result["u"] - u_exact) ** 2))

    if MPI.COMM_WORLD.rank == 0:
        print(f"L2_ERROR: {err:.12e}")
        print(f"WALL_TIME: {wall:.12e}")
        print(result["solver_info"])
