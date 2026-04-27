import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Extract output grid info
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]  # [xmin, xmax, ymin, ymax]
    xmin, xmax, ymin, ymax = bbox

    # Parameters
    nu_val = case_spec.get("pde", {}).get("coefficients", {}).get("nu", 1.0)
    N = 24  # mesh resolution
    degree_u = 3
    degree_p = 2

    # Create mesh
    msh = mesh.create_rectangle(
        comm,
        [np.array([xmin, ymin]), np.array([xmax, ymax])],
        [N, N],
        cell_type=mesh.CellType.triangle
    )
    gdim = msh.geometry.dim

    # Mixed element (Taylor-Hood P3/P2)
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    # Trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    # Spatial coordinates
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi

    # Manufactured solution:
    # u = [pi*cos(pi*y)*sin(pi*x), -pi*cos(pi*x)*sin(pi*y)]
    # p = cos(pi*x)*cos(pi*y)
    u_exact = ufl.as_vector([
        pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
        -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    ])
    p_exact = ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])

    # Analytically computed source term: f = -nu * laplacian(u_exact) + grad(p_exact)
    f1 = (2.0 * nu_val * pi**3 - pi) * ufl.sin(pi * x[0]) * ufl.cos(pi * x[1])
    f2 = -(2.0 * nu_val * pi**3 + pi) * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    f_vec = ufl.as_vector([f1, f2])

    # Bilinear form for Stokes
    a_form = (
        nu_val * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )
    L_form = ufl.inner(f_vec, v) * ufl.dx

    # Boundary conditions
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )

    # Velocity BC from exact solution
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    # Pressure pinning at origin
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], xmin) & np.isclose(x[1], ymin)
    )
    p_pin = fem.Function(Q)
    p_pin.interpolate(fem.Expression(p_exact, Q.element.interpolation_points))
    bc_p = fem.dirichletbc(p_pin, p_dofs, W.sub(1))
    bcs = [bc_u, bc_p]

    # Compile forms
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)

    # Assemble matrix
    A = petsc.assemble_matrix(a_compiled, bcs=bcs)
    A.assemble()

    # Assemble RHS
    b = petsc.create_vector(W)
    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_compiled)
    petsc.apply_lifting(b, [a_compiled], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, bcs)

    # Solve with MUMPS direct solver (handles saddle-point)
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")
    ksp.setUp()

    w_h = fem.Function(W)
    ksp.solve(b, w_h.x.petsc_vec)
    w_h.x.scatter_forward()

    iterations = max(ksp.getIterationNumber(), 1)

    # Extract velocity
    u_h = w_h.sub(0).collapse()

    # Sample velocity on output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((XX.size, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
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

    u_values = np.zeros((len(pts), gdim))
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for idx, global_idx in enumerate(eval_map):
            u_values[global_idx, :] = vals[idx, :]

    # Compute velocity magnitude
    vel_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = vel_mag.reshape(ny_out, nx_out)

    # Cleanup PETSc objects
    ksp.destroy()
    A.destroy()
    b.destroy()

    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-12,
        "iterations": iterations,
    }

    return {"u": u_grid, "solver_info": solver_info}
