import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    # Parse case spec
    pde = case_spec.get("pde", {})
    k_val = float(pde.get("helmholtz", {}).get("k", 18.0))

    # Get output grid
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]  # [xmin, xmax, ymin, ymax]

    # Mesh parameters - P4 with mesh_res=100 gives ~1e-9 accuracy
    elem_degree = 4
    mesh_res = 100

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(
        comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle
    )

    # Function space
    V = fem.functionspace(domain, ("Lagrange", elem_degree))

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(domain)

    # Source term
    f_expr = 12.0 * (
        ufl.exp(-90.0 * ((x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2))
        - ufl.exp(-90.0 * ((x[0] - 0.7) ** 2 + (x[1] - 0.3) ** 2))
    )

    # Helmholtz: -nabla^2 u - k^2 u = f
    # Weak form: int(grad(u).grad(v)) dx - k^2 int(u*v) dx = int(f*v) dx
    k2 = fem.Constant(domain, ScalarType(k_val ** 2))
    a = (
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - k2 * ufl.inner(u, v) * ufl.dx
    )
    L = ufl.inner(f_expr, v) * ufl.dx

    # Boundary conditions: u = 0 on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    # Solve using direct solver (LU) - best for indefinite Helmholtz
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="helmholtz_",
    )
    u_sol = problem.solve()

    # Sample solution onto output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts_2d = np.column_stack(
        [XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    )

    # Find cells for each point
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts_2d)
    colliding_cells = geometry.compute_colliding_cells(
        domain, cell_candidates, pts_2d
    )

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts_2d)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts_2d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_grid = np.zeros(nx_out * ny_out)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(
            np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32)
        )
        u_grid[eval_map] = vals.flatten()

    u_grid = u_grid.reshape(ny_out, nx_out)

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": 1,
    }

    return {"u": u_grid, "solver_info": solver_info}
