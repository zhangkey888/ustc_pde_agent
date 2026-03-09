import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    coeffs = pde.get("coefficients", {})
    kappa_val = float(coeffs.get("kappa", 1.0))

    time_params = pde.get("time", {})
    t_end = float(time_params.get("t_end", 0.1))
    dt_val = float(time_params.get("dt", 0.01))
    scheme = time_params.get("scheme", "backward_euler")

    output_spec = case_spec.get("output", {})
    nx_out = int(output_spec.get("nx", 50))
    ny_out = int(output_spec.get("ny", 50))

    N = 64
    degree = 2

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    t_c = fem.Constant(domain, ScalarType(0.0))
    pi_v = ufl.pi
    kappa_c = fem.Constant(domain, ScalarType(kappa_val))
    dt_c = fem.Constant(domain, ScalarType(dt_val))

    u_exact_ufl = ufl.exp(-t_c) * ufl.sin(3*pi_v*(x[0]+x[1])) * ufl.sin(pi_v*(x[0]-x[1]))

    # Source term f = du/dt - kappa * laplacian(u)
    # du/dt = -u_exact, laplacian(u) = -20*pi^2*u_exact
    # f = -u_exact + kappa*20*pi^2*u_exact = u_exact*(20*kappa*pi^2 - 1)
    f_ufl = u_exact_ufl * (20.0 * kappa_val * pi_v**2 - 1.0)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V)
    u_h = fem.Function(V)

    # Initial condition
    u_n.interpolate(lambda xa: np.sin(3*np.pi*(xa[0]+xa[1])) * np.sin(np.pi*(xa[0]-xa[1])))
    u_initial = fem.Function(V)
    u_initial.x.array[:] = u_n.x.array[:]

    # Backward Euler
    a_form = (u*v/dt_c + kappa_c*ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L_form = (u_n*v/dt_c + f_ufl*v) * ufl.dx

    # BCs
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda xa: np.ones(xa.shape[1], dtype=bool))
    u_bc = fem.Function(V)
    bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc.interpolate(bc_expr)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)
    bcs = [bc]

    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)

    n_steps = int(np.ceil(t_end / dt_val))
    actual_dt = t_end / n_steps
    dt_c.value = actual_dt

    A = petsc.assemble_matrix(a_compiled, bcs=bcs)
    A.assemble()

    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.CG)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    ksp.setUp()

    total_iters = 0
    current_time = 0.0

    for step in range(n_steps):
        current_time += actual_dt
        t_c.value = current_time
        u_bc.interpolate(bc_expr)

        b_vec = petsc.create_vector(V)
        with b_vec.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b_vec, L_compiled)
        petsc.apply_lifting(b_vec, [a_compiled], bcs=[bcs])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b_vec, bcs)

        ksp.solve(b_vec, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        total_iters += ksp.getIterationNumber()

        u_n.x.array[:] = u_h.x.array[:]
        b_vec.destroy()

    u_grid = _evaluate_on_grid(domain, u_h, nx_out, ny_out)
    u_init_grid = _evaluate_on_grid(domain, u_initial, nx_out, ny_out)

    ksp.destroy()
    A.destroy()

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
            "dt": actual_dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }


def _evaluate_on_grid(domain, u_func, nx, ny):
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')

    points = np.zeros((3, nx*ny))
    points[0, :] = XX.flatten()
    points[1, :] = YY.flatten()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    values = np.full(nx*ny, np.nan)
    pts_list = []
    cells_list = []
    idx_list = []

    for i in range(nx*ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            pts_list.append(points.T[i])
            cells_list.append(links[0])
            idx_list.append(i)

    if len(pts_list) > 0:
        vals = u_func.eval(np.array(pts_list), np.array(cells_list, dtype=np.int32))
        values[idx_list] = vals.flatten()

    return values.reshape((nx, ny))


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "coefficients": {"kappa": 1.0},
            "time": {"t_end": 0.1, "dt": 0.01, "scheme": "backward_euler"},
        },
        "domain": {"type": "unit_square"},
        "output": {"nx": 50, "ny": 50},
    }

    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0

    print(f"Wall time: {elapsed:.2f}s")
    print(f"Solution shape: {result['u'].shape}")

    xs = np.linspace(0, 1, 50)
    ys = np.linspace(0, 1, 50)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    u_exact = np.exp(-0.1) * np.sin(3*np.pi*(XX+YY)) * np.sin(np.pi*(XX-YY))

    err = np.sqrt(np.mean((result['u'] - u_exact)**2))
    print(f"RMS error: {err:.6e}")
    print(f"Max error: {np.nanmax(np.abs(result['u'] - u_exact)):.6e}")
