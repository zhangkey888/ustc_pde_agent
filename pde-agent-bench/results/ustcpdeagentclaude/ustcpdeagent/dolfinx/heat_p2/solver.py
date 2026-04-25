import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters
    t0 = 0.0
    t_end = 0.06
    dt_val = 0.0005
    n_steps = int(round((t_end - t0) / dt_val))
    kappa_val = 1.0

    # Mesh
    N = 64
    degree = 2
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, PETSc.ScalarType(t0))
    kappa = fem.Constant(domain, PETSc.ScalarType(kappa_val))

    # Exact: u = exp(-t)*(x^2+y^2)
    u_exact = ufl.exp(-t_const) * (x[0]**2 + x[1]**2)
    # du/dt = -exp(-t)*(x^2+y^2)
    # -kappa*Laplacian = -1*(2+2)*exp(-t) = -4*exp(-t)
    # f = du/dt - kappa*Lap(u) = -exp(-t)*(x^2+y^2) - 1*4*exp(-t)
    f_expr = -ufl.exp(-t_const) * (x[0]**2 + x[1]**2) - 4.0 * ufl.exp(-t_const)

    # Initial condition
    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(ufl.exp(-fem.Constant(domain, PETSc.ScalarType(0.0)))*(x[0]**2+x[1]**2), V.element.interpolation_points))
    # simpler: interpolate lambda
    u_n.interpolate(lambda xx: xx[0]**2 + xx[1]**2)

    u_initial_arr = None

    # BC - Dirichlet from exact
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool))
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, bc_dofs)

    # Backward Euler weak form: (u - u_n)/dt - kappa*Lap u = f
    # (u, v)/dt + kappa*(grad u, grad v) = (u_n, v)/dt + (f, v)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))

    a = (u * v / dt_c) * ufl.dx + kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n * v / dt_c) * ufl.dx + f_expr * v * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-12, atol=1e-14, max_it=500)

    u_h = fem.Function(V)
    total_iters = 0

    for step in range(n_steps):
        t_new = t0 + (step + 1) * dt_val
        t_const.value = t_new

        # Update BC from exact at t_new
        u_bc.interpolate(lambda xx, tv=t_new: np.exp(-tv) * (xx[0]**2 + xx[1]**2))

        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        total_iters += solver.getIterationNumber()

        u_n.x.array[:] = u_h.x.array

    # Sample on grid
    grid = case_spec["output"]["grid"]
    nx = grid["nx"]; ny = grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx*ny)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(domain, cand, pts)

    points_on = []
    cells_on = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            points_on.append(pts[i])
            cells_on.append(links[0])
            eval_map.append(i)

    vals = np.full(pts.shape[0], np.nan)
    if len(points_on) > 0:
        res = u_h.eval(np.array(points_on), np.array(cells_on, dtype=np.int32))
        vals[eval_map] = res.flatten()

    u_grid = vals.reshape(ny, nx)

    # Initial
    u_init_grid = (xs[None, :]**2 + ys[:, None]**2)

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-12,
            "iterations": total_iters,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": {"t0": 0.0, "t_end": 0.06}},
    }
    t0 = time.time()
    out = solve(case_spec)
    elapsed = time.time() - t0
    u = out["u"]
    # Compare to exact
    nx, ny = 64, 64
    xs = np.linspace(0, 1, nx); ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys)
    exact = np.exp(-0.06) * (XX**2 + YY**2)
    err = np.sqrt(np.mean((u - exact)**2))
    maxerr = np.max(np.abs(u - exact))
    print(f"Time: {elapsed:.3f}s  L2-ish: {err:.3e}  Max: {maxerr:.3e}")
    print("Info:", out["solver_info"])
