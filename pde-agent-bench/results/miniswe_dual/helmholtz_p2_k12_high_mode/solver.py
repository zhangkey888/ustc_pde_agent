import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict = None) -> dict:
    """Solve Helmholtz equation: -nabla^2 u - k^2 u = f with Dirichlet BCs."""
    
    k_val = 12.0
    nx_out = 50
    ny_out = 50
    
    if case_spec is not None:
        pde = case_spec.get('pde', {})
        params = pde.get('parameters', {})
        k_val = params.get('k', k_val)
        output = case_spec.get('output', {})
        nx_out = output.get('nx', nx_out)
        ny_out = output.get('ny', ny_out)
    
    comm = MPI.COMM_WORLD
    
    element_degree = 3
    N = 48
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    x = ufl.SpatialCoordinate(domain)
    
    u_exact_ufl = ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
    
    coeff = 2.0 * (3.0 * np.pi)**2 - k_val**2
    f_ufl = coeff * ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
    
    u_trial = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx - k_val**2 * ufl.inner(u_trial, v) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(3 * np.pi * x[0]) * np.sin(3 * np.pi * x[1]))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-12
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
        },
        petsc_options_prefix="helmholtz_"
    )
    u_sol = problem.solve()
    
    x_out = np.linspace(0, 1, nx_out)
    y_out = np.linspace(0, 1, ny_out)
    X, Y = np.meshgrid(x_out, y_out, indexing='ij')
    points_3d = np.zeros((nx_out * ny_out, 3))
    points_3d[:, 0] = X.ravel()
    points_3d[:, 1] = Y.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": 1,
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve()
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    info = result["solver_info"]
    
    nx, ny = u_grid.shape
    x_out = np.linspace(0, 1, nx)
    y_out = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_out, y_out, indexing='ij')
    u_exact = np.sin(3 * np.pi * X) * np.sin(3 * np.pi * Y)
    
    max_err = np.nanmax(np.abs(u_grid - u_exact))
    rms_err = np.sqrt(np.nanmean((u_grid - u_exact)**2))
    
    print(f"Mesh: {info['mesh_resolution']}, Degree: {info['element_degree']}")
    print(f"Max error on grid: {max_err:.6e}")
    print(f"RMS error on grid: {rms_err:.6e}")
    print(f"Time: {elapsed:.3f}s")
