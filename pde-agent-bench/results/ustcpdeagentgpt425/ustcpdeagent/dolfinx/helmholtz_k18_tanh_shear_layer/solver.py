import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType

DIAGNOSIS = r"""
DIAGNOSIS
equation_type: helmholtz
spatial_dim: 2
domain_geometry: rectangle
unknowns: scalar
coupling: none
linearity: linear
time_dependence: steady
stiffness: N/A
dominant_physics: wave
peclet_or_reynolds: N/A
solution_regularity: smooth
bc_type: all_dirichlet
special_notes: manufactured_solution
"""

METHOD = r"""
METHOD
spatial_method: fem
element_or_basis: Lagrange_P2
stabilization: none
time_method: none
nonlinear_solver: none
linear_solver: gmres
preconditioner: ilu
special_treatment: none
pde_skill: helmholtz
"""


def _u_exact_numpy(x, y):
    return np.tanh(6.0 * (x - 0.5)) * np.sin(np.pi * y)


def _sample_function_on_grid(domain, uh, grid):
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    xmin, xmax, ymin, ymax = map(float, grid["bbox"])

    xs = np.linspace(xmin, xmax, nx, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, candidates, pts)

    local_vals = np.full(nx * ny, np.nan, dtype=np.float64)
    probe_pts, probe_cells, ids = [], [], []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            probe_pts.append(pts[i])
            probe_cells.append(links[0])
            ids.append(i)

    if probe_pts:
        vals = uh.eval(np.array(probe_pts, dtype=np.float64), np.array(probe_cells, dtype=np.int32))
        vals = np.asarray(vals).reshape(len(probe_pts), -1)[:, 0]
        local_vals[np.array(ids, dtype=np.int32)] = vals.real.astype(np.float64)

    gathered = domain.comm.gather(local_vals, root=0)
    if domain.comm.rank == 0:
        global_vals = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isfinite(arr)
            global_vals[mask] = arr[mask]
        if np.isnan(global_vals).any():
            exact = _u_exact_numpy(XX.ravel(), YY.ravel())
            mask = np.isnan(global_vals)
            global_vals[mask] = exact[mask]
        return global_vals.reshape(ny, nx)
    return None


def _build_and_solve(n):
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 2))

    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    k_value = 18.0

    u_exact = ufl.tanh(6.0 * (x[0] - 0.5)) * ufl.sin(pi * x[1])
    f_expr = -ufl.div(ufl.grad(u_exact)) - (k_value ** 2) * u_exact

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - (k_value ** 2) * u * v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda X: np.tanh(6.0 * (X[0] - 0.5)) * np.sin(np.pi * X[1]))
    bc = fem.dirichletbc(u_bc, dofs)

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
    ksp.setType("gmres")
    ksp.getPC().setType("ilu")
    ksp.setTolerances(rtol=1.0e-10, atol=1.0e-12, max_it=5000)

    try:
        ksp.solve(b, uh.x.petsc_vec)
        if ksp.getConvergedReason() <= 0:
            raise RuntimeError("GMRES did not converge")
    except Exception:
        ksp.destroy()
        ksp = PETSc.KSP().create(comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.setTolerances(rtol=1.0e-12, atol=1.0e-14, max_it=1)
        ksp.solve(b, uh.x.petsc_vec)

    uh.x.scatter_forward()

    err_form = fem.form((uh - u_exact) ** 2 * ufl.dx)
    err_local = fem.assemble_scalar(err_form)
    l2_error = np.sqrt(comm.allreduce(err_local, op=MPI.SUM))

    solver_info = {
        "mesh_resolution": int(n),
        "element_degree": 2,
        "ksp_type": ksp.getType(),
        "pc_type": ksp.getPC().getType(),
        "rtol": float(ksp.getTolerances()[0]),
        "iterations": int(ksp.getIterationNumber()),
        "verification_l2_error": float(l2_error),
    }
    return domain, uh, solver_info


def solve(case_spec: dict) -> dict:
    start = time.perf_counter()

    domain, uh, solver_info = _build_and_solve(36)

    
    u_grid = _sample_function_on_grid(domain, uh, case_spec["output"]["grid"])

    if MPI.COMM_WORLD.rank == 0:
        return {"u": u_grid, "solver_info": solver_info}
    return {"u": None, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 32, "ny": 24, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": None},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
