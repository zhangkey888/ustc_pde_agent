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
    t_end = 0.1
    dt_val = 0.0025  # refine from suggested 0.005
    n_steps = int(round((t_end - t0) / dt_val))
    dt_val = (t_end - t0) / n_steps

    N = 96
    degree = 2

    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    kappa = 1.0 + 0.3 * ufl.sin(6 * ufl.pi * x[0]) * ufl.sin(6 * ufl.pi * x[1])

    # Manufactured solution: u = exp(-t)*sin(2*pi*x)*sin(2*pi*y)
    # Time derivative: du/dt = -exp(-t)*sin(2*pi*x)*sin(2*pi*y) = -u
    # Source term: f = du/dt - div(kappa grad u) = -u - div(kappa grad u)

    t_c = fem.Constant(msh, PETSc.ScalarType(0.0))
    u_exact = ufl.exp(-t_c) * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    # f = -u - div(kappa grad u)
    f_expr = -u_exact - ufl.div(kappa * ufl.grad(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u_n = fem.Function(V)  # previous time step

    # Initial condition
    def u0_func(xx):
        return np.sin(2 * np.pi * xx[0]) * np.sin(2 * np.pi * xx[1])
    u_n.interpolate(u0_func)

    dt_c = fem.Constant(msh, PETSc.ScalarType(dt_val))

    # Backward Euler: (u - u_n)/dt - div(k grad u) = f  at t^{n+1}
    # Weak: (u,v) + dt*(k grad u, grad v) = (u_n, v) + dt*(f, v)
    a = ufl.inner(u, v) * ufl.dx + dt_c * ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(u_n, v) * ufl.dx + dt_c * ufl.inner(f_expr, v) * ufl.dx

    # Boundary condition (Dirichlet from exact solution)
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)

    def set_bc(t_val):
        def g(xx):
            return np.exp(-t_val) * np.sin(2 * np.pi * xx[0]) * np.sin(2 * np.pi * xx[1])
        u_bc.interpolate(g)

    set_bc(t0)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    pc = solver.getPC()
    pc.setType("hypre")
    pc.setHYPREType("boomeramg")
    solver.setTolerances(rtol=1e-10, atol=1e-14, max_it=200)

    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array

    total_iters = 0
    t = t0
    for step in range(n_steps):
        t_new = t + dt_val
        t_c.value = t_new
        set_bc(t_new)

        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iters += solver.getIterationNumber()

        u_n.x.array[:] = u_sol.x.array
        t = t_new

    # Sample on output grid
    grid = case_spec["output"]["grid"]
    nx = grid["nx"]
    ny = grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_grid_flat = np.full(nx * ny, np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid_flat[eval_map] = vals.flatten()
    u_grid = u_grid_flat.reshape(ny, nx)

    # Initial condition on grid
    u_init_grid = np.sin(2 * np.pi * XX) * np.sin(2 * np.pi * YY)

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iters,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {"grid": {"nx": 128, "ny": 128, "bbox": [0, 1, 0, 1]}}
    }
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    u_grid = result["u"]
    grid = case_spec["output"]["grid"]
    xs = np.linspace(0, 1, grid["nx"])
    ys = np.linspace(0, 1, grid["ny"])
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.exp(-0.1) * np.sin(2 * np.pi * XX) * np.sin(2 * np.pi * YY)
    err = np.sqrt(np.mean((u_grid - u_ex)**2))
    maxerr = np.max(np.abs(u_grid - u_ex))
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"L2 error (grid RMS): {err:.3e}")
    print(f"Max error: {maxerr:.3e}")
    print(f"Iterations: {result['solver_info']['iterations']}")
