"""
```DIAGNOSIS
equation_type: heat
spatial_dim: 2
domain_geometry: rectangle
unknowns: scalar
coupling: none
linearity: linear
time_dependence: transient
stiffness: stiff
dominant_physics: diffusion
peclet_or_reynolds: N/A
solution_regularity: smooth
bc_type: all_dirichlet
special_notes: manufactured_solution
```

```METHOD
spatial_method: fem
element_or_basis: Lagrange_P2
stabilization: none
time_method: backward_euler
nonlinear_solver: none
linear_solver: cg
preconditioner: hypre
special_treatment: none
pde_skill: heat
```
"""

import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _interp_exact(fun: fem.Function, t: float) -> None:
    fun.interpolate(lambda x: np.exp(-2.0 * t) * np.cos(np.pi * x[0]) * np.cos(np.pi * x[1]))


def _sample_function(u_func: fem.Function, nx: int, ny: int, bbox):
    msh = u_func.function_space.mesh
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    local_vals = np.full(pts.shape[0], np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    ids = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            ids.append(i)

    if points_on_proc:
        vals = u_func.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32)).reshape(-1)
        local_vals[np.array(ids, dtype=np.int32)] = np.real(vals)

    gathered = msh.comm.allgather(local_vals)
    out = np.full_like(local_vals, np.nan)
    for arr in gathered:
        mask = np.isnan(out) & ~np.isnan(arr)
        out[mask] = arr[mask]
    out = np.nan_to_num(out)
    return out.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    pde = case_spec.get("pde", {})
    grid = case_spec["output"]["grid"]

    t0 = float(pde.get("t0", 0.0))
    t_end = float(pde.get("t_end", 0.2))
    dt_suggested = float(pde.get("dt", 0.02))
    if t_end <= t0:
        t_end = 0.2
    if dt_suggested <= 0:
        dt_suggested = 0.02

    mesh_resolution = 72
    element_degree = 2
    dt = min(dt_suggested, 0.005)
    n_steps = max(1, int(round((t_end - t0) / dt)))
    dt = (t_end - t0) / n_steps

    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", element_degree))

    kappa = 0.5
    dt_c = fem.Constant(msh, ScalarType(dt))
    t_c = fem.Constant(msh, ScalarType(t0 + dt))

    uh = fem.Function(V)
    u_n = fem.Function(V)
    u_bc = fem.Function(V)
    _interp_exact(u_n, t0)
    _interp_exact(u_bc, t0)

    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(u_bc, dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    f_expr = (-2.0 + 2.0 * kappa * ufl.pi**2) * ufl.exp(-2.0 * t_c) * ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])

    a = (u * v + dt_c * ScalarType(kappa) * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n * v + dt_c * f_expr * v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)

    u_initial = _sample_function(u_n, grid["nx"], grid["ny"], grid["bbox"])
    total_iterations = 0
    t = t0
    for _ in range(n_steps):
        t += dt
        t_c.value = ScalarType(t)
        _interp_exact(u_bc, t)

        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()
        total_iterations += int(max(solver.getIterationNumber(), 0))
        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()

    u_grid = _sample_function(uh, grid["nx"], grid["ny"], grid["bbox"])
    return {
        "u": u_grid,
        "u_initial": u_initial,
        "solver_info": {
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": solver.getType(),
            "pc_type": solver.getPC().getType(),
            "rtol": 1e-10,
            "iterations": int(total_iterations),
            "dt": float(dt),
            "n_steps": int(n_steps),
            "time_scheme": "backward_euler",
        },
    }


def _self_test():
    case = {
        "pde": {"time": True, "t0": 0.0, "t_end": 0.2, "dt": 0.02, "scheme": "backward_euler"},
        "output": {"grid": {"nx": 81, "ny": 81, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    t_start = time.perf_counter()
    result = solve(case)
    wall = time.perf_counter() - t_start
    grid = case["output"]["grid"]
    xs = np.linspace(grid["bbox"][0], grid["bbox"][1], grid["nx"])
    ys = np.linspace(grid["bbox"][2], grid["bbox"][3], grid["ny"])
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    u_exact = np.exp(-2.0 * case["pde"]["t_end"]) * np.cos(np.pi * XX) * np.cos(np.pi * YY)
    err = np.sqrt(np.mean((result["u"] - u_exact) ** 2))
    if MPI.COMM_WORLD.rank == 0:
        print(f"L2_ERROR: {err:.12e}")
        print(f"WALL_TIME: {wall:.12e}")


if __name__ == "__main__":
    _self_test()
