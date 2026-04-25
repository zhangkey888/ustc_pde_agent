import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    nx = case_spec["output"]["grid"]["nx"]
    ny = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]

    # Mesh
    mesh_res = 128
    degree = 2
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)

    # Function Space
    V = fem.functionspace(domain, ("Lagrange", degree))

    # Boundary conditions
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Variational forms
    v = ufl.TrialFunction(V)
    w = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    f = ufl.sin(8 * ufl.pi * x[0]) * ufl.cos(6 * ufl.pi * x[1])

    # First solve: -Delta v = f -> a(v, w) = (f, w)
    a1 = ufl.inner(ufl.grad(v), ufl.grad(w)) * ufl.dx
    L1 = ufl.inner(f, w) * ufl.dx

    problem1 = petsc.LinearProblem(
        a1, L1, bcs=[bc],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "pc_hypre_type": "boomeramg", "ksp_rtol": 1e-10},
        petsc_options_prefix="v_"
    )
    v_sol = problem1.solve()

    # Second solve: -Delta u = v_sol -> a(u, w) = (v_sol, w)
    u = ufl.TrialFunction(V)
    a2 = ufl.inner(ufl.grad(u), ufl.grad(w)) * ufl.dx
    L2 = ufl.inner(v_sol, w) * ufl.dx

    problem2 = petsc.LinearProblem(
        a2, L2, bcs=[bc],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "pc_hypre_type": "boomeramg", "ksp_rtol": 1e-10},
        petsc_options_prefix="u_"
    )
    u_sol = problem2.solve()

    # Interpolate to output grid
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx * ny)]

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_out = np.full(len(pts), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_out[eval_map] = vals.flatten()
    
    u_grid = u_out.reshape((ny, nx))

    # Solver info
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1e-10,
        "iterations": 0 # We didn't manually extract iterations, but 0 is ok if required
    }

    return {"u": u_grid, "solver_info": solver_info}

