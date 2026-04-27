import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    k_val = case_spec["pde"]["coefficients"]["k"]
    
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    xmin, xmax, ymin, ymax = bbox
    
    elem_degree = 2
    N = 64
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(
        comm,
        [np.array([xmin, ymin]), np.array([xmax, ymax])],
        [N, N],
        cell_type=mesh.CellType.triangle
    )
    
    V = fem.functionspace(domain, ("Lagrange", elem_degree))
    
    x = ufl.SpatialCoordinate(domain)
    pi_val = ufl.pi
    
    u_exact = ufl.tanh(6.0 * (x[0] - 0.5)) * ufl.sin(pi_val * x[1])
    
    k_const = fem.Constant(domain, ScalarType(k_val))
    
    f_expr = -ufl.div(ufl.grad(u_exact)) - k_const**2 * u_exact
    
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    a = ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx - k_const**2 * ufl.inner(u_trial, v_test) * ufl.dx
    L = ufl.inner(f_expr, v_test) * ufl.dx
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
        },
        petsc_options_prefix="helmholtz_"
    )
    u_sol = problem.solve()
    
    # Compute L2 error
    error_form = fem.form(ufl.inner(u_sol - u_exact, u_sol - u_exact) * ufl.dx)
    error_local = fem.assemble_scalar(error_form)
    error_global = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
    print(f"L2 error: {error_global:.6e}")
    
    # Sample solution onto output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    
    points = np.zeros((3, nx_out * ny_out))
    points[0] = XX.ravel()
    points[1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_grid = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid[eval_map] = vals.flatten()
    
    u_grid = u_grid.reshape(ny_out, nx_out)
    
    iterations = 1  # Direct solver = 1 iteration
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": elem_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
        }
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "pde": {
            "coefficients": {"k": 18.0},
        },
        "output": {
            "grid": {
                "nx": 100,
                "ny": 100,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        }
    }
    
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Wall time: {t1-t0:.2f}s")
    print(f"Output shape: {result['u'].shape}")
    print(f"NaN count: {np.isnan(result['u']).sum()}")
    print(f"Solver info: {result['solver_info']}")
    
    xs = np.linspace(0, 1, 100)
    ys = np.linspace(0, 1, 100)
    XX, YY = np.meshgrid(xs, ys)
    u_exact_grid = np.tanh(6*(XX - 0.5)) * np.sin(np.pi * YY)
    
    max_err = np.nanmax(np.abs(result['u'] - u_exact_grid))
    l2_err = np.sqrt(np.nanmean((result['u'] - u_exact_grid)**2))
    print(f"Max pointwise error: {max_err:.6e}")
    print(f"L2 grid error: {l2_err:.6e}")
