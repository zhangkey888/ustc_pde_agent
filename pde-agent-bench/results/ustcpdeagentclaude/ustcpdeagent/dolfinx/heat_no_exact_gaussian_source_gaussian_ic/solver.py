import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    t0 = 0.0
    t_end = 0.1
    kappa_val = 1.0

    try:
        pde = case_spec.get("pde", {})
        tinfo = pde.get("time", {})
        t0 = float(tinfo.get("t0", t0))
        t_end = float(tinfo.get("t_end", t_end))
    except Exception:
        pass

    grid = case_spec["output"]["grid"]
    nx_out = int(grid["nx"])
    ny_out = int(grid["ny"])
    bbox = grid["bbox"]

    # Higher accuracy: finer mesh + P2 + smaller dt + Crank-Nicolson
    N = 160
    degree = 2
    dt_val = 0.001
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)

    u_n = fem.Function(V)
    u_n.interpolate(lambda xx: np.exp(-120.0 * ((xx[0] - 0.6) ** 2 + (xx[1] - 0.4) ** 2)))

    def sample_on_grid(func):
        xs = np.linspace(bbox[0], bbox[1], nx_out)
        ys = np.linspace(bbox[2], bbox[3], ny_out)
        XX, YY = np.meshgrid(xs, ys)
        pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
        tree = geometry.bb_tree(domain, domain.topology.dim)
        cand = geometry.compute_collisions_points(tree, pts)
        coll = geometry.compute_colliding_cells(domain, cand, pts)
        vals = np.zeros(nx_out * ny_out)
        pts_on, cells_on, idx_map = [], [], []
        for i in range(pts.shape[0]):
            links = coll.links(i)
            if len(links) > 0:
                pts_on.append(pts[i])
                cells_on.append(links[0])
                idx_map.append(i)
        if pts_on:
            v = func.eval(np.array(pts_on), np.array(cells_on, dtype=np.int32))
            vals[idx_map] = v.flatten()
        return vals.reshape(ny_out, nx_out)

    u_initial_grid = sample_on_grid(u_n)

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, bdofs)

    n_steps = int(np.ceil((t_end - t0) / dt_val))
    dt_val = (t_end - t0) / n_steps

    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))
    kappa = fem.Constant(domain, PETSc.ScalarType(kappa_val))
    theta = 0.5  # Crank-Nicolson

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f_expr = ufl.exp(-200.0 * ((x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2))

    # CN: (u-u_n)/dt + kappa*(theta*lap(u) + (1-theta)*lap(u_n)) -- sign: -lap, so:
    # (u - u_n)/dt = kappa*(theta*Lap u + (1-theta)*Lap u_n) + f
    # Weak: (u,v)/dt + theta*kappa*(grad u, grad v) = (u_n,v)/dt - (1-theta)*kappa*(grad u_n, grad v) + (f,v)
    a = u * v * ufl.dx + theta * dt_c * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n * v * ufl.dx
         - (1.0 - theta) * dt_c * kappa * ufl.inner(ufl.grad(u_n), ufl.grad(v)) * ufl.dx
         + dt_c * f_expr * v * ufl.dx)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    try:
        pc.setHYPREType("boomeramg")
    except Exception:
        pass
    solver.setTolerances(rtol=1e-10, atol=1e-14, max_it=500)

    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array

    total_iters = 0
    for step in range(n_steps):
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iters += solver.getIterationNumber()

        u_n.x.array[:] = u_sol.x.array

    u_grid = sample_on_grid(u_sol)

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": int(total_iters),
            "dt": float(dt_val),
            "n_steps": int(n_steps),
            "time_scheme": "crank_nicolson",
        },
    }
