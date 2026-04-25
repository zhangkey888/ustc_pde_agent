import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    N = 64
    degree = 2
    domain = mesh.create_rectangle(comm, [np.array([0.0,0.0]), np.array([1.0,1.0])],
                                   [N, N], cell_type=mesh.CellType.quadrilateral)
    V = fem.functionspace(domain, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.sin(4*ufl.pi*x[0])*ufl.sin(4*ufl.pi*x[1])
    f = -ufl.div(ufl.grad(u_exact))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
    L = ufl.inner(f, v)*ufl.dx

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim,
        lambda x: np.ones(x.shape[1], dtype=bool))
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)

    problem = petsc.LinearProblem(a, L, bcs=[bc],
        petsc_options={"ksp_type":"cg","pc_type":"hypre","ksp_rtol":1e-10},
        petsc_options_prefix="poisson_")
    u_sol = problem.solve()
    its = problem.solver.getIterationNumber()

    grid = case_spec["output"]["grid"]
    nx, ny = grid["nx"], grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx*ny)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(domain, cand, pts)
    cells = []
    points_on = []
    idx = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            points_on.append(pts[i])
            cells.append(links[0])
            idx.append(i)
    vals = np.zeros(pts.shape[0])
    if len(points_on) > 0:
        v = u_sol.eval(np.array(points_on), np.array(cells, dtype=np.int32))
        vals[idx] = v.flatten()
    u_grid = vals.reshape(ny, nx)

    return {"u": u_grid,
            "solver_info": {
                "mesh_resolution": N,
                "element_degree": degree,
                "ksp_type": "cg",
                "pc_type": "hypre",
                "rtol": 1e-10,
                "iterations": int(its),
            }}
