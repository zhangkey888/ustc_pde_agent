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
    k_val = float(pde_spec.get("wavenumber", 18.0))
    
    output = case_spec.get("output", {})
    nx_out = int(output.get("nx", 50))
    ny_out = int(output.get("ny", 50))

    element_degree = 2
    
    resolutions = [48, 80, 120]
    prev_norm = None
    u_sol = None
    domain = None
    final_N = None
    ksp_type_used = "preonly"
    pc_type_used = "lu"
    rtol_used = 1e-10
    
    for N in resolutions:
        comm = MPI.COMM_WORLD
        domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        x = ufl.SpatialCoordinate(domain)
        
        f_expr = 12.0 * (
            ufl.exp(-90.0 * ((x[0] - 0.3)**2 + (x[1] - 0.7)**2))
            - ufl.exp(-90.0 * ((x[0] - 0.7)**2 + (x[1] - 0.3)**2))
        )
        
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
        
        ksp_type_used = "preonly"
        pc_type_used = "lu"
        
        problem = petsc.LinearProblem(
            a, L, bcs=[bc],
            petsc_options={
                "ksp_type": ksp_type_used,
                "pc_type": pc_type_used,
            },
            petsc_options_prefix="helmholtz_"
        )
        u_sol = problem.solve()
        
        norm_form = fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)
        norm_val = np.sqrt(fem.assemble_scalar(norm_form))
        
        final_N = N
        
        if prev_norm is not None:
            rel_err = abs(norm_val - prev_norm) / (abs(norm_val) + 1e-15)
            if rel_err < 0.01:
                break
        
        prev_norm = norm_val
    
    x_coords = np.linspace(0, 1, nx_out)
    y_coords = np.linspace(0, 1, ny_out)
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    points_flat = np.zeros((3, nx_out * ny_out))
    points_flat[0, :] = xx.flatten()
    points_flat[1, :] = yy.flatten()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_flat.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_flat.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points_flat.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_flat[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx_out * ny_out, 0.0)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": final_N,
            "element_degree": element_degree,
            "ksp_type": ksp_type_used,
            "pc_type": pc_type_used,
            "rtol": rtol_used,
            "iterations": 1,
        },
    }


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve()
    elapsed = time.time() - t0
    print(f"Solve completed in {elapsed:.2f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{result['u'].min():.6f}, {result['u'].max():.6f}]")
    print(f"Solver info: {result['solver_info']}")
