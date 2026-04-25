import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    eps_val = 0.0
    beta_val = np.array([10.0, 4.0])
    
    # Output grid
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
    
    # Mesh
    N = 128
    degree = 2
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(msh, ("Lagrange", degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(msh)
    beta = fem.Constant(msh, PETSc.ScalarType((beta_val[0], beta_val[1])))
    
    # Manufactured solution u = sin(pi*x)*sin(pi*y)
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    # f = beta . grad(u_exact) (since eps=0)
    grad_uex = ufl.grad(u_exact)
    f_expr = beta[0] * grad_uex[0] + beta[1] * grad_uex[1]
    
    # Galerkin form (pure advection)
    # a(u,v) = (beta.grad(u), v)
    # Need SUPG stabilization: add tau * (beta.grad(u) - f) * (beta.grad(v))
    
    # Cell diameter
    h = ufl.CellDiameter(msh)
    beta_norm = ufl.sqrt(beta[0]**2 + beta[1]**2)
    # SUPG tau for pure advection
    tau = h / (2.0 * beta_norm)
    
    # Residual
    res_u = beta[0] * u.dx(0) + beta[1] * u.dx(1)
    res_f = f_expr
    
    # Galerkin + SUPG
    a = res_u * v * ufl.dx + tau * res_u * (beta[0] * v.dx(0) + beta[1] * v.dx(1)) * ufl.dx
    L = f_expr * v * ufl.dx + tau * res_f * (beta[0] * v.dx(0) + beta[1] * v.dx(1)) * ufl.dx
    
    # Dirichlet BC only on inflow boundary: x=0 or y=0 (since beta>0 in both components)
    fdim = msh.topology.dim - 1
    def inflow(x):
        return np.isclose(x[0], 0.0) | np.isclose(x[1], 0.0)
    
    inflow_facets = mesh.locate_entities_boundary(msh, fdim, inflow)
    inflow_dofs = fem.locate_dofs_topological(V, fdim, inflow_facets)
    
    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(
        ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        V.element.interpolation_points
    )
    u_bc.interpolate(u_bc_expr)
    bc = fem.dirichletbc(u_bc, inflow_dofs)
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "gmres", "pc_type": "ilu", "ksp_rtol": 1e-12, "ksp_atol": 1e-14, "ksp_max_it": 2000},
        petsc_options_prefix="cd_"
    )
    u_sol = problem.solve()
    ksp = problem.solver
    iters = ksp.getIterationNumber()
    
    # Compute L2 error for verification
    u_ex_fn = fem.Function(V)
    u_ex_expr = fem.Expression(
        ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        V.element.interpolation_points
    )
    u_ex_fn.interpolate(u_ex_expr)
    
    err_form = fem.form((u_sol - u_ex_fn)**2 * ufl.dx)
    err_L2 = np.sqrt(comm.allreduce(fem.assemble_scalar(err_form), op=MPI.SUM))
    print(f"L2 error: {err_L2}")
    
    # Sample on output grid
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
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.zeros(pts.shape[0])
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-12,
            "iterations": iters,
        }
    }


if __name__ == "__main__":
    import time
    spec = {"output": {"grid": {"nx": 128, "ny": 128, "bbox": [0,1,0,1]}}}
    t0 = time.time()
    res = solve(spec)
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")
    print(f"u shape: {res['u'].shape}")
    # Check error on grid
    xs = np.linspace(0,1,128); ys = np.linspace(0,1,128)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.sin(np.pi*XX)*np.sin(np.pi*YY)
    err = np.max(np.abs(res['u'] - u_ex))
    l2 = np.sqrt(np.mean((res['u'] - u_ex)**2))
    print(f"Max error: {err}, L2 grid: {l2}")
