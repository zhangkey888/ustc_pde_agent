import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec):
    comm = MPI.COMM_WORLD
    N = 64
    deg = 2
    dt_val = 0.005
    t_end = 0.06
    n_steps = int(round(t_end / dt_val))

    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", deg))

    x = ufl.SpatialCoordinate(msh)
    t_c = fem.Constant(msh, PETSc.ScalarType(0.0)); t_var = ufl.variable(t_c)
    kappa = 1 + 0.4*ufl.sin(2*ufl.pi*x[0])*ufl.sin(2*ufl.pi*x[1])
    u_exact = ufl.exp(-t_var)*ufl.sin(2*ufl.pi*x[0])*ufl.sin(2*ufl.pi*x[1])
    # f = du/dt - div(kappa grad u)
    f_expr = ufl.diff(u_exact, t_var) - ufl.div(kappa*ufl.grad(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V)
    dt_c = fem.Constant(msh, PETSc.ScalarType(dt_val))

    a = u*v*ufl.dx + dt_c*ufl.inner(kappa*ufl.grad(u), ufl.grad(v))*ufl.dx
    L = u_n*v*ufl.dx + dt_c*f_expr*v*ufl.dx

    # Initial condition
    u_init_expr = ufl.sin(2*ufl.pi*x[0])*ufl.sin(2*ufl.pi*x[1])
    u_n.interpolate(fem.Expression(u_init_expr, V.element.interpolation_points))

    # BC
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, bdofs)

    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("hypre")
    solver.setTolerances(rtol=1e-10)

    u_sol = fem.Function(V)
    total_iters = 0

    # save initial
    u_init_arr = None

    for step in range(n_steps):
        t_new = (step+1)*dt_val
        t_c.value = t_new
        # update bc
        u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

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

    # Sample on grid
    grid = case_spec["output"]["grid"]
    nx, ny = grid["nx"], grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx*ny)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)
    cells = []
    pts_ok = []
    idx = []
    for i in range(pts.shape[0]):
        l = coll.links(i)
        if len(l) > 0:
            cells.append(l[0]); pts_ok.append(pts[i]); idx.append(i)
    vals = u_sol.eval(np.array(pts_ok), np.array(cells, dtype=np.int32)).flatten()
    out = np.zeros(nx*ny)
    out[idx] = vals
    u_grid = out.reshape(ny, nx)

    # initial
    u_init_grid = np.sin(2*np.pi*XX)*np.sin(2*np.pi*YY)

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": N, "element_degree": deg,
            "ksp_type": "cg", "pc_type": "hypre", "rtol": 1e-10,
            "iterations": total_iters,
            "dt": dt_val, "n_steps": n_steps, "time_scheme": "backward_euler"
        }
    }

if __name__ == "__main__":
    import time
    spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0,1,0,1]}}}
    t0 = time.time()
    r = solve(spec)
    dt = time.time()-t0
    nx, ny = 64, 64
    xs = np.linspace(0,1,nx); ys = np.linspace(0,1,ny)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.exp(-0.06)*np.sin(2*np.pi*XX)*np.sin(2*np.pi*YY)
    err = np.sqrt(np.mean((r["u"]-u_ex)**2))
    print(f"time={dt:.3f}s, L2err={err:.3e}, iters={r['solver_info']['iterations']}")
