import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parse case spec
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]  # [xmin, xmax, ymin, ymax]
    
    # Problem parameters
    nu = 1.0
    
    # Mesh resolution - chosen to meet accuracy requirement
    N = 128
    
    # Create mesh
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    
    # Taylor-Hood P2/P1 mixed element
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, V_to_W = W.sub(0).collapse()
    Q, Q_to_W = W.sub(1).collapse()
    
    # Trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(msh)
    
    # Exact solution for BCs and source term
    u_ex_ufl = ufl.as_vector([
        ufl.pi * ufl.exp(x[0]) * ufl.cos(ufl.pi * x[1]),
        -ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    ])
    p_ex = ufl.exp(x[0]) * ufl.cos(ufl.pi * x[1])
    
    # Source term f = -nu*laplacian(u_ex) + grad(p_ex)
    f_expr = -nu * ufl.div(ufl.grad(u_ex_ufl)) + ufl.grad(p_ex)
    
    # Variational form for Stokes
    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + q * ufl.div(u) * ufl.dx)
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # Boundary conditions: u = g on all boundaries
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    # Interpolate exact velocity for BC
    def u_exact_py(x):
        return np.vstack([
            np.pi * np.exp(x[0]) * np.cos(np.pi * x[1]),
            -np.exp(x[0]) * np.sin(np.pi * x[1])
        ])
    
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(u_exact_py)
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    bcs = [bc_u]
    
    # Pressure pinning at origin corner
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    if len(p_dofs) > 0:
        p0_func = fem.Function(Q)
        p0_func.interpolate(lambda x: np.exp(x[0]) * np.cos(np.pi * x[1]))
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bcs.append(bc_p)
    
    # Solve with LU direct solver
    rtol = 1e-12
    ksp_type = "preonly"
    pc_type = "lu"
    
    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="stokes_"
    )
    
    w_h = problem.solve()
    w_h.x.scatter_forward()
    
    # Get solver info
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Collapse solution
    u_h = w_h.sub(0).collapse()
    p_h = w_h.sub(1).collapse()
    u_h.x.scatter_forward()
    p_h.x.scatter_forward()
    
    # Compute L2 error for verification
    err_u = fem.form(ufl.inner(u_h - u_ex_ufl, u_h - u_ex_ufl) * ufl.dx)
    l2_err_u = np.sqrt(comm.allreduce(fem.assemble_scalar(err_u), op=MPI.SUM))
    
    err_p = fem.form((p_h - p_ex)**2 * ufl.dx)
    l2_err_p = np.sqrt(comm.allreduce(fem.assemble_scalar(err_p), op=MPI.SUM))
    
    # Sample velocity magnitude on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    
    # Points for evaluation: shape (3, N)
    pts = np.zeros((3, nx_out * ny_out), dtype=np.float64)
    pts[0] = XX.ravel()
    pts[1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((pts.shape[1], gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals
    
    # Compute magnitude
    magnitude = np.linalg.norm(u_values, axis=1)
    u_grid = magnitude.reshape(ny_out, nx_out)
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
        }
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        },
        "pde": {
            "time": False
        }
    }
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Wall time: {t1-t0:.3f}s")
    print(f"Output shape: {result['u'].shape}")
    print(f"Max velocity magnitude: {np.nanmax(result['u']):.6e}")
    print(f"Has NaN: {np.any(np.isnan(result['u']))}")
    print(f"Solver info: {result['solver_info']}")
