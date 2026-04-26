import numpy as np
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc as fem_petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc
import time


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    t_start = time.time()

    # Extract PDE parameters
    pde = case_spec["pde"]
    nu = float(pde["coefficients"]["viscosity"])

    # Extract output grid
    out = case_spec["output"]
    grid = out["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]  # [xmin, xmax, ymin, ymax]

    # ---- Mesh ----
    mesh_res = 384
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim
    tdim = domain.topology.dim
    fdim = tdim - 1

    # ---- Mixed function space (Taylor-Hood P2/P1) ----
    vel_el = basix_element("Lagrange", domain.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", domain.topology.cell_name(), 1)
    W = fem.functionspace(domain, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    # ---- Variational form ----
    (u_trial, p_trial) = ufl.TrialFunctions(W)
    (v_test, q_test) = ufl.TestFunctions(W)

    f_val = [float(v) for v in pde["source"]]
    f = fem.Constant(domain, PETSc.ScalarType(f_val))

    def eps(u):
        return ufl.sym(ufl.grad(u))

    # Stokes weak form: 2*nu*inner(eps(u),eps(v)) - p*div(v) + div(u)*q = inner(f,v)
    a = (2.0 * nu * ufl.inner(eps(u_trial), eps(v_test)) * ufl.dx
         - p_trial * ufl.div(v_test) * ufl.dx
         + ufl.div(u_trial) * q_test * ufl.dx)
    L = ufl.inner(f, v_test) * ufl.dx

    # ---- Boundary conditions ----
    bcs = []

    # Top boundary (y=1): u = [1.0, 0.0]
    top_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 1.0))
    top_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)
    u_top = fem.Function(V)
    u_top.interpolate(lambda x: np.vstack([np.ones(x.shape[1]), np.zeros(x.shape[1])]))
    bcs.append(fem.dirichletbc(u_top, top_dofs, W.sub(0)))

    # Right boundary (x=1): u = [0.0, -0.8]
    right_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 1.0))
    right_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, right_facets)
    u_right = fem.Function(V)
    u_right.interpolate(lambda x: np.vstack([np.zeros(x.shape[1]), -0.8 * np.ones(x.shape[1])]))
    bcs.append(fem.dirichletbc(u_right, right_dofs, W.sub(0)))

    # Left boundary (x=0): u = [0.0, 0.0]
    left_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0))
    left_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, left_facets)
    u_left = fem.Function(V)
    u_left.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(u_left, left_dofs, W.sub(0)))

    # Bottom boundary (y=0): u = [0.0, 0.0]
    bottom_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 0.0))
    bottom_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets)
    u_bottom = fem.Function(V)
    u_bottom.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(u_bottom, bottom_dofs, W.sub(0)))

    # ---- Pressure pinning at origin ----
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    if len(p_dofs) > 0:
        p0_func = fem.Function(Q)
        p0_func.x.array[:] = 0.0
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bcs.append(bc_p)

    # ---- Assemble system ----
    a_form = fem.form(a)
    L_form = fem.form(L)

    A = fem_petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()

    b = fem_petsc.create_vector(L_form.function_spaces)
    with b.localForm() as loc:
        loc.set(0)
    fem_petsc.assemble_vector(b, L_form)
    fem_petsc.apply_lifting(b, [a_form], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem_petsc.set_bc(b, bcs)

    # ---- Solve with LU (MUMPS) ----
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-12

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.getPC().setFactorSolverType("mumps")
    solver.setFromOptions()

    w_vec = A.createVecRight()
    solver.solve(b, w_vec)

    iterations = int(solver.getIterationNumber())

    # ---- Extract solution ----
    w_h = fem.Function(W)
    w_h.x.array[:] = w_vec.array
    w_h.x.scatter_forward()

    u_h = w_h.sub(0).collapse()
    u_h.x.scatter_forward()

    # ---- Accuracy verification ----
    # Residual norm
    res = w_vec.duplicate()
    A.mult(w_vec, res)
    res.axpy(-1.0, b)
    residual_norm = res.norm()

    # ---- Sample velocity magnitude on output grid ----
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)

    pts = np.zeros((ny_out * nx_out, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((pts.shape[0], gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals

    # Compute magnitude
    u_mag = np.linalg.norm(u_values, axis=1).reshape(ny_out, nx_out)
    u_mag = np.nan_to_num(u_mag, nan=0.0)

    t_total = time.time() - t_start

    print(f"Stokes solver: mesh_res={mesh_res}, iterations={iterations}, residual={residual_norm:.2e}, time={t_total:.2f}s")
    print(f"  Output shape: ({ny_out}, {nx_out}), max|u|={np.max(u_mag):.4f}")

    # ---- Build solver_info ----
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": 2,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }

    # Add time info if PDE contains time
    if "time" in pde and pde["time"] is not None:
        solver_info["dt"] = 0.0
        solver_info["n_steps"] = 0
        solver_info["time_scheme"] = "none"

    return {
        "u": u_mag,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "type": "stokes",
            "coefficients": {"viscosity": 0.3},
            "source": ["0.0", "0.0"],
            "time": None,
        },
        "output": {
            "field": "velocity_magnitude",
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        }
    }
    result = solve(case_spec)
    print(f"\nResult shape: {result['u'].shape}")
    print(f"Result min/max: {result['u'].min():.6f} / {result['u'].max():.6f}")
    print(f"Solver info: {result['solver_info']}")
