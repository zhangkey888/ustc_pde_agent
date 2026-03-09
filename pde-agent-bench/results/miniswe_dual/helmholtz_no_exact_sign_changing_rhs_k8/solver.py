import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    k_val = float(pde.get("wavenumber", 8.0))
    domain_info = case_spec.get("domain", {})
    x_range = domain_info.get("x_range", [0.0, 1.0])
    y_range = domain_info.get("y_range", [0.0, 1.0])
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    bc_spec = pde.get("boundary_value", 0.0)
    if bc_spec is None:
        bc_spec = 0.0
    comm = MPI.COMM_WORLD
    element_degree = 2
    resolutions = [32, 64, 128]
    prev_norm = None
    u_sol = None
    current_domain = None
    final_N = None
    final_ksp_type = "preonly"
    final_pc_type = "lu"
    final_rtol = 1e-10
    final_iterations = 1
    for N in resolutions:
        p0 = np.array([x_range[0], y_range[0]])
        p1 = np.array([x_range[1], y_range[1]])
        domain = mesh.create_rectangle(comm, [p0, p1], [N, N], cell_type=mesh.CellType.triangle)
        V = fem.functionspace(domain, ("Lagrange", element_degree))
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        x = ufl.SpatialCoordinate(domain)
        f_expr = ufl.cos(4 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
        k2 = fem.Constant(domain, ScalarType(k_val**2))
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k2 * ufl.inner(u, v) * ufl.dx
        L = ufl.inner(f_expr, v) * ufl.dx
        tdim = domain.topology.dim
        fdim = tdim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        if callable(bc_spec):
            u_bc = fem.Function(V)
            u_bc.interpolate(bc_spec)
            bc = fem.dirichletbc(u_bc, dofs)
        else:
            bc = fem.dirichletbc(ScalarType(float(bc_spec)), dofs, V)
        bcs = [bc]
        ksp_type = "preonly"
        pc_type = "lu"
        try:
            problem = petsc.LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": ksp_type, "pc_type": pc_type}, petsc_options_prefix="hlm" + str(N) + "_")
            u_h = problem.solve()
        except Exception:
            ksp_type = "gmres"
            pc_type = "ilu"
            problem = petsc.LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": "1e-10", "ksp_max_it": "2000"}, petsc_options_prefix="hlmfb" + str(N) + "_")
            u_h = problem.solve()
        norm_form = fem.form(ufl.inner(u_h, u_h) * ufl.dx)
        norm_val = np.sqrt(fem.assemble_scalar(norm_form))
        final_N = N
        final_ksp_type = ksp_type
        final_pc_type = pc_type
        u_sol = u_h
        current_domain = domain
        if prev_norm is not None:
            rel_change = abs(norm_val - prev_norm) / (norm_val + 1e-15)
            if rel_change < 1e-3:
                break
        prev_norm = norm_val
    x_coords = np.linspace(x_range[0], x_range[1], nx_out)
    y_coords = np.linspace(y_range[0], y_range[1], ny_out)
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
    points_2d = np.column_stack([xx.ravel(), yy.ravel()])
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
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    u_grid = u_values.reshape((nx_out, ny_out))
    solver_info = {"mesh_resolution": final_N, "element_degree": element_degree, "ksp_type": final_ksp_type, "pc_type": final_pc_type, "rtol": final_rtol, "iterations": final_iterations}
    return {"u": u_grid, "solver_info": solver_info}
