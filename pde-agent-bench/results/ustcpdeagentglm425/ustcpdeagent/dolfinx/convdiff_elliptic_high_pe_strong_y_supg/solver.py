import numpy as np
import time
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    wall_start = time.time()
    
    # --- Extract parameters ---
    pde = case_spec["pde"]
    eps = float(pde.get("diffusion", 0.01))
    beta_list = list(pde.get("velocity", [0.0, 15.0]))
    
    out_grid = case_spec["output"]["grid"]
    nx_out = out_grid["nx"]
    ny_out = out_grid["ny"]
    bbox = out_grid["bbox"]
    xmin, xmax, ymin, ymax = bbox

    # --- Mesh and function space ---
    mesh_res = 64
    elem_degree = 2

    domain = mesh.create_unit_square(MPI.COMM_WORLD, mesh_res, mesh_res,
                                      cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim
    tdim = domain.topology.dim
    fdim = tdim - 1

    V = fem.functionspace(domain, ("Lagrange", elem_degree))

    # --- Define variational problem ---
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(domain)
    beta_ufl = ufl.as_vector(beta_list)

    # Exact solution: u_ex = sin(pi*x)*sin(pi*y)
    # Source: f = 2*eps*pi^2*sin(pi*x)*sin(pi*y) + beta_y*pi*sin(pi*x)*cos(pi*y)
    f_source = (2.0 * eps * ufl.pi**2 * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
                + beta_list[1] * ufl.pi * ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]))

    # Standard Galerkin
    a = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + ufl.inner(ufl.dot(beta_ufl, ufl.grad(u)), v) * ufl.dx
    L = ufl.inner(f_source, v) * ufl.dx

    # --- SUPG stabilization ---
    h = ufl.CellDiameter(domain)
    beta_norm_val = np.sqrt(sum(b**2 for b in beta_list))
    
    # Shakib-style tau
    tau_supg = h**2 / ufl.sqrt(4.0 * eps**2 + (beta_norm_val * h)**2)

    # Full residual SUPG for P2: R(u) = -eps*div(grad(u)) + beta.grad(u) - f
    a += ufl.inner(tau_supg * ufl.dot(beta_ufl, ufl.grad(v)),
                    -eps * ufl.div(ufl.grad(u)) + ufl.dot(beta_ufl, ufl.grad(u))) * ufl.dx
    L += ufl.inner(tau_supg * ufl.dot(beta_ufl, ufl.grad(v)), f_source) * ufl.dx

    # --- Boundary conditions (u=0 on entire boundary) ---
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, 
                        lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)

    # --- Solve with direct LU ---
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-12

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
        },
        petsc_options_prefix="convdiff_"
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()

    iterations = 1  # Direct solver

    # --- Compute L2 error for verification ---
    u_ex_expr = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    u_ex_func = fem.Function(V)
    u_ex_func.interpolate(fem.Expression(u_ex_expr, V.element.interpolation_points))

    error_L2_sq = fem.assemble_scalar(fem.form((u_sol - u_ex_func)**2 * ufl.dx))
    error_L2 = np.sqrt(domain.comm.allreduce(error_L2_sq, op=MPI.SUM))

    wall_end = time.time()
    if MPI.COMM_WORLD.rank == 0:
        print(f"L2 error: {error_L2:.6e}, Wall time: {wall_end - wall_start:.3f}s")

    # --- Evaluate on output grid ---
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((pts.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()

    u_grid_global = np.zeros_like(u_values)
    domain.comm.Allreduce(u_values, u_grid_global, op=MPI.SUM)
    u_grid_2d = u_grid_global.reshape(ny_out, nx_out)

    return {
        "u": u_grid_2d,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": elem_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
        }
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "type": "convection_diffusion",
            "diffusion": 0.01,
            "velocity": [0.0, 15.0],
        },
        "output": {
            "grid": {
                "nx": 65,
                "ny": 65,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(f"Output shape: {result['u'].shape}, Max: {np.nanmax(result['u']):.6e}")
