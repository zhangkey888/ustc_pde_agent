import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    """Solve the Poisson equation: -div(kappa * grad(u)) = f with Dirichlet BCs."""
    
    comm = MPI.COMM_WORLD
    
    # Mesh and element parameters - P2 on 64x64 gives excellent accuracy
    N = 64
    degree = 2
    
    # Create mesh
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Coefficient kappa = 1 + 50*exp(-200*(x-0.5)^2)
    kappa = 1.0 + 50.0 * ufl.exp(-200.0 * (x[0] - 0.5)**2)
    
    # Source term f = 1 + sin(2*pi*x)*cos(2*pi*y)
    pi = ufl.pi
    f = 1.0 + ufl.sin(2 * pi * x[0]) * ufl.cos(2 * pi * x[1])
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Bilinear and linear forms
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    
    # Boundary conditions: u = g on ∂Ω
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Parse boundary condition value from case_spec
    g_value = 0.0  # default
    bc_spec = case_spec.get("pde", {}).get("boundary_conditions", [])
    if isinstance(bc_spec, list):
        for bc_item in bc_spec:
            if isinstance(bc_item, dict) and bc_item.get("type") == "dirichlet":
                val = bc_item.get("value", "0")
                if isinstance(val, (int, float)):
                    g_value = float(val)
                elif isinstance(val, str):
                    try:
                        g_value = float(val)
                    except ValueError:
                        g_value = 0.0
                break
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x_arr: np.full_like(x_arr[0], g_value))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    bcs = [bc]
    
    # Solver parameters
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
        },
        petsc_options_prefix="poisson_"
    )
    
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, 0] = points_2d[:, 0]
    points_3d[:, 1] = points_2d[:, 1]
    
    # Point evaluation using geometry utilities
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
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
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": -1,
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    import time
    
    case_spec = {
        "pde": {
            "type": "poisson",
            "source": "1 + sin(2*pi*x)*cos(2*pi*y)",
            "coefficients": {"kappa": {"type": "expr", "expr": "1 + 50*exp(-200*(x-0.5)**2)"}},
            "boundary_conditions": [{"type": "dirichlet", "value": "0"}],
        },
        "domain": {"type": "unit_square", "dim": 2},
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    print(f"Solve time: {elapsed:.3f}s")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{np.nanmin(result['u']):.8f}, {np.nanmax(result['u']):.8f}]")
    print(f"NaN count: {np.isnan(result['u']).sum()}")
    print(f"Solver info: {result['solver_info']}")
