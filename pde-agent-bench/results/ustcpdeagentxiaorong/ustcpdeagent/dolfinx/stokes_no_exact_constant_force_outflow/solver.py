import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc as fempetsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Extract parameters
    nu_val = case_spec["pde"]["coefficients"]["nu"]
    source = case_spec["pde"]["source"]
    bcs_spec = case_spec["pde"]["bcs"]

    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    N = 64
    degree_u = 2
    degree_p = 1

    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    # Taylor-Hood mixed elements
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, V_map = W.sub(0).collapse()
    Q, Q_map = W.sub(1).collapse()

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    f_vals = [float(s) for s in source]
    f = fem.Constant(msh, np.array(f_vals, dtype=PETSc.ScalarType))
    nu_c = fem.Constant(msh, PETSc.ScalarType(nu_val))

    # Standard Stokes weak form: nu * (grad u, grad v) - (p, div v) + (div u, q) = (f, v)
    a = (nu_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx

    # Boundary conditions
    fdim = msh.topology.dim - 1
    bcs = []
    bc_locations = set()

    def make_marker(loc):
        if loc == "x0":
            return lambda x: np.isclose(x[0], 0.0)
        elif loc == "x1":
            return lambda x: np.isclose(x[0], 1.0)
        elif loc == "y0":
            return lambda x: np.isclose(x[1], 0.0)
        elif loc == "y1":
            return lambda x: np.isclose(x[1], 1.0)
        return None

    for bc_item in bcs_spec:
        if bc_item["type"] != "dirichlet":
            continue
        location = bc_item["location"]
        value = bc_item["value"]
        bc_locations.add(location)
        marker = make_marker(location)
        if marker is None:
            continue

        facets = mesh.locate_entities_boundary(msh, fdim, marker)
        dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
        u_bc = fem.Function(V)
        val = [float(v_) for v_ in value]
        u_bc.interpolate(lambda x, v0=val[0], v1=val[1]: np.array([
            np.full(x.shape[1], v0), np.full(x.shape[1], v1)
        ]))
        bcs.append(fem.dirichletbc(u_bc, dofs, W.sub(0)))

    # If all boundaries have Dirichlet velocity BCs, pin pressure
    all_walls = {"x0", "x1", "y0", "y1"}
    if bc_locations == all_walls:
        p_dofs = fem.locate_dofs_geometrical(
            (W.sub(1), Q),
            lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
        )
        if len(p_dofs[0]) > 0:
            p0 = fem.Function(Q)
            p0.x.array[:] = 0.0
            bcs.append(fem.dirichletbc(p0, p_dofs, W.sub(1)))

    # Manual assembly with MUMPS direct solver
    a_form = fem.form(a)
    L_form = fem.form(L)

    A = fempetsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()

    b = fempetsc.create_vector(W)
    with b.localForm() as loc:
        loc.set(0)
    fempetsc.assemble_vector(b, L_form)
    fempetsc.apply_lifting(b, [a_form], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fempetsc.set_bc(b, bcs)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")
    solver.setUp()

    w_h = fem.Function(W)
    solver.solve(b, w_h.x.petsc_vec)
    w_h.x.scatter_forward()

    # Extract velocity
    u_func = fem.Function(V)
    u_func.interpolate(w_h.sub(0))

    # Sample on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((nx_out * ny_out, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((nx_out * ny_out, gdim), 0.0)
    if len(points_on_proc) > 0:
        vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for idx, global_idx in enumerate(eval_map):
            u_values[global_idx, :] = vals[idx, :]

    magnitude = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = magnitude.reshape(ny_out, nx_out)
    u_grid = np.nan_to_num(u_grid, nan=0.0)

    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "iterations": 1,
    }

    return {"u": u_grid, "solver_info": solver_info}
