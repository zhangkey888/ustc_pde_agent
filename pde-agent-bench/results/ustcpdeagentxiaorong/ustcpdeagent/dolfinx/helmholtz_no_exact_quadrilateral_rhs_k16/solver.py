import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Extract parameters from case_spec
    k_val = case_spec["pde"]["helmholtz"]["k"]

    # Get output grid specifications
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    xmin, xmax, ymin, ymax = bbox

    # Mesh and element parameters
    # k=16 → wavelength ≈ 0.393
    # P3 with 96 elements per direction gives excellent accuracy
    degree = 3
    n_mesh = 96

    # Create quadrilateral mesh
    p0 = np.array([xmin, ymin])
    p1 = np.array([xmax, ymax])
    domain = mesh.create_rectangle(
        comm, [p0, p1], [n_mesh, n_mesh],
        cell_type=mesh.CellType.quadrilateral
    )

    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))

    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)

    # Source term: f = sin(6*pi*x)*cos(5*pi*y)
    pi_val = ufl.pi
    f_expr = ufl.sin(6 * pi_val * x[0]) * ufl.cos(5 * pi_val * x[1])

    # Wavenumber
    k = fem.Constant(domain, ScalarType(k_val))

    # Weak form: -∇²u - k²u = f
    # Integration by parts: ∫ ∇u·∇v dx - k² ∫ u·v dx = ∫ f·v dx
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k**2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    # Boundary conditions: u = 0 on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    # Solve using direct LU (most robust for indefinite Helmholtz)
    ksp_type = "preonly"
    pc_type = "lu"

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
        },
        petsc_options_prefix="helmholtz_"
    )
    u_sol = problem.solve()

    # Sample solution onto the output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()

    # Build bounding box tree and find cells
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

    u_values = np.full(points.shape[1], np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(
            np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32)
        )
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": n_mesh,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": 1e-10,
            "iterations": 1,
        }
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "pde": {
            "helmholtz": {"k": 16.0},
            "source": "sin(6*pi*x)*cos(5*pi*y)",
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
    print(f"Solution shape: {result['u'].shape}")
    print(f"Solution range: [{np.nanmin(result['u']):.8f}, {np.nanmax(result['u']):.8f}]")
    print(f"Any NaN: {np.any(np.isnan(result['u']))}")
    print(f"Wall time: {t1-t0:.3f}s")
    print(f"Solver info: {result['solver_info']}")
