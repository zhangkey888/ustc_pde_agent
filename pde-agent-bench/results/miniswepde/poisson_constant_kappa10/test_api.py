import dolfinx.fem.petsc
import inspect
print("LinearProblem attributes:")
for name in dir(dolfinx.fem.petsc.LinearProblem):
    if not name.startswith('_'):
        print(f"  {name}")
