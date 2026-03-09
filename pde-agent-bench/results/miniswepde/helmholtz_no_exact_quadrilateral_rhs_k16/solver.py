import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict = None) -> dict:
    if case_spec is None:
        case_spec = {}
    
    pde_spec = case_spec.get("pde", {})
    k_val = pde_spec.get("wavenumber", 16.0)
    
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    comm = MPI.COMM_WORLD
    
    element_degree = 3
    
    resolutions = [40, 64, 96]
    prev_norm = None
    u_sol = None
    final_N = None
    final_ksp_type = None
    final_pc_type = None
    final_rtol = None
    current_domain = None
    
    for N in resolutions:
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.quadrilateral)
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        x = ufl.SpatialCoordinate(domain)
        pi_val = np.pi
        
        f_expr = ufl.sin(6 * pi_val * x[0]) * ufl.cos(5 * pi_val * x[1])
        
        k_const = fem.Constant(domain, ScalarType(k_val))
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k_const**2 * ufl.inner(u, v) * ufl.dx
        L = ufl.inner(f_expr, v) * ufl.dx
        
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
        )
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
        
        ksp_type = "gmres"
        pc_type = "ilu"
        rtol = 1e-10
        
        try:
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                    "ksp_rtol": str(rtol),
                    "ksp_max_it": "5000",
                    "ksp_gmres_restart": "100",
                },
                petsc_options_prefix=f"helmholtz_{N}_"
            )
            u_sol_current = problem.solve()
        except Exception:
            ksp_type = "preonly"
            pc_type = "lu"
            problem = petsc.LinearProblem(
                a, L, bcs=[bc],
                petsc_options={
                    "ksp_type": ksp_type,
                    "pc_type": pc_type,
                },
                petsc_options_prefix=f"helmholtz_direct_{N}_"
            )
            u_sol_current = problem.solve()
        
        norm_form = fem.form(ufl.inner(u_sol_current, u_sol_current) * ufl.dx)
        local_norm = fem.assemble_scalar(norm_form)
        global_norm = np.sqrt(comm.allreduce(local_norm, op=MPI.SUM))
        
        u_sol = u_sol_current
        final_N = N
        final_ksp_type = ksp_type
        final_pc_type = pc_type
        final_rtol = rtol
        current_domain = domain
        
        if prev_norm is not None and global_norm > 1e-15:
            rel_change = abs(global_norm - prev_norm) / abs(global_norm)
            if rel_change < 0.01:
                break
        
        prev_norm = global_norm
    
    x_out = np.linspace(0, 1, nx_out)
    y_out = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_out, y_out, indexing='ij')
    
    points_2d = np.column_stack([X.ravel(), Y.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    bb_tree = geometry.bb_tree(current_domain, current_domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(current_domain, cell_candidates, points_3d)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": final_N,
            "element_degree": element_degree,
            "ksp_type": final_ksp_type,
            "pc_type": final_pc_type,
            "rtol": final_rtol,
            "iterations": 0,
        }
    }


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve()
    elapsed = time.time() - t0
    u_grid = result["u"]
    print(f"Solve completed in {elapsed:.2f}s")
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution range: [{np.nanmin(u_grid):.6e}, {np.nanmax(u_grid):.6e}]")
    print(f"NaN count: {np.isnan(u_grid).sum()}")
    print(f"Solver info: {result['solver_info']}")
