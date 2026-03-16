dofs_p = fem.locate_dofs_geometrical((W.sub(1), Q), origin_marker)
bc_p = fem.dirichletbc(PETSc.ScalarType(0.0), dofs_p[0], W.sub(1))