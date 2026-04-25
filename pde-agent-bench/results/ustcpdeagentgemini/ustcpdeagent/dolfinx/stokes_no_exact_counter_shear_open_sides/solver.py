import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from dolfinx.fem.petsc import LinearProblem
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Extract output grid specifications
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Mesh resolution
    nx = 64
    ny = 64
    
    # Create mesh
    msh = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([1.0, 1.0])], [nx, ny], cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    # Mixed function space (Taylor-Hood P2-P1)
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    # Trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    nu = 1.0
    f = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))

    # Variational form
    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx

    # Boundary conditions
    bcs = []
    fdim = msh.topology.dim - 1

    # Bottom boundary (y = 0)
    bottom_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    dofs_bottom = fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets)
    u_bottom = fem.Function(V)
    u_bottom.interpolate(lambda x: np.vstack((np.full(x.shape[1], -1.0), np.zeros(x.shape[1]))))
    bc_bottom = fem.dirichletbc(u_bottom, dofs_bottom, W.sub(0))
    bcs.append(bc_bottom)

    # Top boundary (y = 1)
    top_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
    dofs_top = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)
    u_top = fem.Function(V)
    u_top.interpolate(lambda x: np.vstack((np.full(x.shape[1], 1.0), np.zeros(x.shape[1]))))
    bc_top = fem.dirichletbc(u_top, dofs_top, W.sub(0))
    bcs.append(bc_top)

    # Pressure pinning (to remove nullspace)
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    if len(p_dofs) > 0:
        p0 = fem.Function(Q)
        p0.x.array[:] = 0.0
        bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))
        bcs.append(bc_p)

    # Linear solver setup
    problem = LinearProblem(a, L, bcs=bcs,
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
                            petsc_options_prefix="stokes_")
    
    w_h = problem.solve()
    u_h = w_h.sub(0).collapse()

    # Interpolate to target grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i, pt in enumerate(pts):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pt)
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_out = np.zeros(nx_out * ny_out)
    if len(points_on_proc) > 0:
        u_vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_out[eval_map] = np.linalg.norm(u_vals, axis=1)

    u_out = u_out.reshape((ny_out, nx_out))

    solver_info = {
        "mesh_resolution": nx,
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": 1,
    }

    return {"u": u_out, "solver_info": solver_info}
