import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    t_end = 0.005
    dt_val = 0.00005
    n_steps = int(round(t_end / dt_val))

    N = 128
    degree = 2

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    # Initial condition
    u_n = fem.Function(V)
    u_n.interpolate(lambda x: np.sin(6*np.pi*x[0]) * np.sin(6*np.pi*x[1]))

    u_h = fem.Function(V)
    u_h.x.array[:] = u_n.x.array[:]

    # BC: zero on all boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)

    kappa = 1.0
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = u*v*ufl.dx + dt_c*kappa*ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
    L = u_n*v*ufl.dx  # f=0

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    # FIX: pass function_spaces list, not form
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10)

    total_iters = 0
    for step in range(n_steps):
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        total_iters += solver.getIterationNumber()
        u_n.x.array[:] = u_h.x.array[:]

    # Sample on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(XX.size)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(domain, cand, pts)

    cells = []
    pts_on = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            cells.append(links[0])
            pts_on.append(pts[i])
            idx_map.append(i)

    u_vals = np.zeros(pts.shape[0])
    if len(pts_on) > 0:
        vals = u_h.eval(np.array(pts_on), np.array(cells, dtype=np.int32))
        u_vals[idx_map] = vals.flatten()

    u_grid = u_vals.reshape(ny_out, nx_out)

    # Initial condition grid
    u0_grid = (np.sin(6*np.pi*XX) * np.sin(6*np.pi*YY)).reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "u_initial": u0_grid,
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
        },
    }


if __name__ == "__main__":
    import time
    spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0,1,0,1]}}
    }
    t0 = time.time()
    res = solve(spec)
    print("time:", time.time()-t0)
    print("u shape:", res["u"].shape, "max:", np.max(np.abs(res["u"])))
    print("info:", res["solver_info"])
