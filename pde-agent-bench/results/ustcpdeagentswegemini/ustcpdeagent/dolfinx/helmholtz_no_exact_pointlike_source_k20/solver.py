import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

# DIAGNOSIS:
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
# special_notes: none
#
# METHOD:
# spatial_method: fem
# element_or_basis: Lagrange_P2
# stabilization: none
# time_method: none
# nonlinear_solver: none
# linear_solver: direct_lu
# preconditioner: none
# special_treatment: none
# pde_skill: helmholtz

ScalarType = PETSc.ScalarType


def _build_zero_dirichlet_bc(domain, V):
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    return fem.dirichletbc(ScalarType(0.0), dofs, V)


def _probe_function(u_func, points_xyz):
    domain = u_func.function_space.mesh
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points_xyz)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_xyz)

    npts = points_xyz.shape[0]
    local_values = np.full(npts, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(npts):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_xyz[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if points_on_proc:
        vals = u_func.eval(
            np.array(points_on_proc, dtype=np.float64),
            np.array(cells_on_proc, dtype=np.int32),
        )
        vals = np.real(np.asarray(vals).reshape(len(points_on_proc), -1)[:, 0])
        local_values[np.array(eval_map, dtype=np.int32)] = vals

    gathered = domain.comm.allgather(local_values)
    global_values = np.full(npts, np.nan, dtype=np.float64)
    for arr in gathered:
        mask = ~np.isnan(arr)
        global_values[mask] = arr[mask]
    return global_values


def _sample_to_grid(u_func, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    xmin, xmax, ymin, ymax = map(float, grid_spec["bbox"])
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])
    vals = _probe_function(u_func, pts)
    return vals.reshape(ny, nx)


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    k = float(case_spec.get("pde", {}).get("k", case_spec.get("wavenumber", 20.0)))
    grid_spec = case_spec["output"]["grid"]

    mesh_resolution = 72
    element_degree = 2
    rtol = 1e-9

    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    bc = _build_zero_dirichlet_bc(domain, V)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    f_expr = 50.0 * ufl.exp(-200.0 * ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2))

    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - (k ** 2) * ufl.inner(u, v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)
    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    uh = fem.Function(V)
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.setTolerances(rtol=rtol, atol=1e-12, max_it=1)
    ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # Lightweight accuracy verification: discrete residual norm
    r = b.copy()
    A.mult(uh.x.petsc_vec, r)
    r.axpy(-1.0, b)
    residual_norm = float(r.norm())
    rhs_norm = float(b.norm())
    rel_residual = residual_norm / max(rhs_norm, 1e-14)

    u_grid = _sample_to_grid(uh, grid_spec)

    solver_info = {
        "mesh_resolution": int(mesh_resolution),
        "element_degree": int(element_degree),
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": float(rtol),
        "iterations": int(ksp.getIterationNumber()),
        "verification": {
            "type": "linear_system_residual",
            "absolute_residual_norm": residual_norm,
            "relative_residual_norm": rel_residual,
        },
    }
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    out = solve({
        "pde": {"k": 20.0},
        "output": {"grid": {"nx": 8, "ny": 8, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    })
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
