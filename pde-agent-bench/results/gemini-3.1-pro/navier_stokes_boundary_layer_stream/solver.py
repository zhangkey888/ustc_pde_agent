V, _ = W.sub(0).collapse()
u_bc_func = fem.Function(V)
u_ex_expr = fem.Expression(u_ex, V.element.interpolation_points)
u_bc_func.interpolate(u_ex_expr)

fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)

dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))