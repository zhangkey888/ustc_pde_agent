import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    mesh_res = 96
    elem_deg = 3
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", elem_deg))
    
    def f_analytical(x):
        exy = np.exp(x[0]*x[1])
        spx = np.sin(np.pi*x[0])
        spy = np.sin(np.pi*x[1])
        cpx = np.cos(np.pi*x[0])
        cpy = np.cos(np.pi*x[1])
        return (-(x[0]**2 + x[1]**2)*exy*spx*spy
                - 2*np.pi*x[1]*exy*cpx*spy
                - 2*np.pi*x[0]*exy*spx*cpy
                + 2*np.pi**2*exy*spx*spy)
    
    def g_analytical(x):
        return np.exp(x[0]*x[1]) * np.sin(np.pi*x[0]) * np.sin(np.pi*x[1])
    
    f_func = fem.Function(V)
    f_func.interpolate(f_analytical)
    
    g_func = fem.Function(V)
    g_func.interpolate(g_analytical)
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(g_func, boundary_dofs)
    
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    a = ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    L = f_func * v_test * ufl.dx
    
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-12
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    iterations = problem.solver.getIterationNumber()
    
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    
    pts = np.zeros((nx_out * ny_out, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    n_pts = pts.shape[0]
    adj = colliding_cells.array
    offsets = colliding_cells.offsets
    
    pts_mask = np.zeros(n_pts, dtype=bool)
    cells_for_pts = np.full(n_pts, -1, dtype=np.int32)
    
    for i in range(n_pts):
        if offsets[i+1] > offsets[i]:
            pts_mask[i] = True
            cells_for_pts[i] = adj[offsets[i]]
    
    u_values = np.full(n_pts, np.nan)
    if np.any(pts_mask):
        eval_pts = pts[pts_mask]
        eval_cells = cells_for_pts[pts_mask]
        vals = u_sol.eval(eval_pts, eval_cells)
        u_values[pts_mask] = vals.flatten()
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    # L2 error verification
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(g_analytical)
    
    error_L2_sq = domain.comm.allreduce(
        fem.assemble_scalar(fem.form((u_sol - u_exact_func)**2 * ufl.dx)),
        op=MPI.SUM
    )
    error_L2 = error_L2_sq**0.5
    
    u_norm_sq = domain.comm.allreduce(
        fem.assemble_scalar(fem.form(u_exact_func**2 * ufl.dx)),
        op=MPI.SUM
    )
    rel_L2 = error_L2 / (u_norm_sq**0.5) if u_norm_sq > 0 else error_L2
    
    if comm.rank == 0:
        print(f"L2 error: {error_L2:.6e}, relative L2: {rel_L2:.6e}")
        print(f"Iterations: {iterations}")
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_deg,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }
    
    return {"u": u_grid, "solver_info": solver_info}
