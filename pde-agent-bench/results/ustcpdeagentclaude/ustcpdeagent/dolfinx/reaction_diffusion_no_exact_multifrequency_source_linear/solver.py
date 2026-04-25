import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters
    pde = case_spec.get("pde", {})
    params = pde.get("parameters", {}) if isinstance(pde, dict) else {}
    epsilon = float(params.get("epsilon", 0.01))
    reaction_alpha = float(params.get("reaction_alpha", 1.0))

    time_info = pde.get("time", {}) if isinstance(pde, dict) else {}
    is_transient = bool(time_info) or True
    t0 = float(time_info.get("t0", 0.0))
    t_end = float(time_info.get("t_end", 0.5))
    dt_val = min(float(time_info.get("dt", 0.01)), 0.002)
    scheme = time_info.get("scheme", "crank_nicolson")

    # Output grid
    out = case_spec["output"]["grid"]
    nx_out, ny_out = out["nx"], out["ny"]
    bbox = out["bbox"]

    # Mesh
    N = 128
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))

    # Boundary condition: u = 0
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Initial condition
    u_n = fem.Function(V)
    u_n.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

    # Source term
    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.sin(5*ufl.pi*x[0])*ufl.sin(3*ufl.pi*x[1]) + 0.5*ufl.sin(9*ufl.pi*x[0])*ufl.sin(7*ufl.pi*x[1])

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    eps_c = fem.Constant(domain, PETSc.ScalarType(epsilon))
    alpha_c = fem.Constant(domain, PETSc.ScalarType(reaction_alpha))
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))

    # Crank-Nicolson for: du/dt - eps*lap(u) + alpha*u = f
    # (u - u_n)/dt + 0.5*(eps*grad(u)*grad(v) + eps*grad(u_n)*grad(v)) + 0.5*alpha*(u+u_n) = f
    a = (u*v*ufl.dx
         + 0.5*dt_c*eps_c*ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
         + 0.5*dt_c*alpha_c*u*v*ufl.dx)
    L = (u_n*v*ufl.dx
         - 0.5*dt_c*eps_c*ufl.inner(ufl.grad(u_n), ufl.grad(v))*ufl.dx
         - 0.5*dt_c*alpha_c*u_n*v*ufl.dx
         + dt_c*f_expr*v*ufl.dx)

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
    pc.setHYPREType("boomeramg")
    solver.setTolerances(rtol=1e-10, atol=1e-14, max_it=1000)

    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array

    # Save initial grid
    def sample_on_grid(u_func):
        xs = np.linspace(bbox[0], bbox[1], nx_out)
        ys = np.linspace(bbox[2], bbox[3], ny_out)
        XX, YY = np.meshgrid(xs, ys)
        pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out*ny_out)]
        tree = geometry.bb_tree(domain, domain.topology.dim)
        cell_cand = geometry.compute_collisions_points(tree, pts)
        colliding = geometry.compute_colliding_cells(domain, cell_cand, pts)
        points_on = []
        cells_on = []
        idx_map = []
        for i in range(pts.shape[0]):
            links = colliding.links(i)
            if len(links) > 0:
                points_on.append(pts[i])
                cells_on.append(links[0])
                idx_map.append(i)
        vals = np.zeros(pts.shape[0])
        if points_on:
            v_arr = u_func.eval(np.array(points_on), np.array(cells_on, dtype=np.int32))
            vals[idx_map] = v_arr.flatten()
        return vals.reshape(ny_out, nx_out)

    u_initial_grid = sample_on_grid(u_n)

    # Time stepping
    n_steps = int(round((t_end - t0) / dt_val))
    total_iters = 0

    for step in range(n_steps):
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
            "iterations": total_iters,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "crank_nicolson",
        },
    }


if __name__ == "__main__":
    import time
    spec = {
        "pde": {
            "parameters": {"epsilon": 0.01, "reaction_alpha": 1.0},
            "time": {"t0": 0.0, "t_end": 0.5, "dt": 0.01, "scheme": "crank_nicolson"},
        },
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}},
    }
    t0 = time.time()
    res = solve(spec)
    print("time:", time.time() - t0)
    print("u shape:", res["u"].shape)
    print("u range:", res["u"].min(), res["u"].max())
    print("iters:", res["solver_info"]["iterations"])
