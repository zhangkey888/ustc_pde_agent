from dolfinx import geometry
import inspect
print("determine_point_ownership signature:")
print(inspect.signature(geometry.determine_point_ownership))
print("\ndetermine_point_ownership docstring:")
print(geometry.determine_point_ownership.__doc__)
