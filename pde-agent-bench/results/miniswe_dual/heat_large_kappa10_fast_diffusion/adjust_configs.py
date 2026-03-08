with open('solver.py', 'r') as f:
    lines = f.readlines()

# Find the configurations section
for i in range(len(lines)):
    if "configurations = [" in lines[i]:
        start_idx = i
        # Find the end of the list
        for j in range(i, len(lines)):
            if "]" in lines[j]:
                end_idx = j
                break
        
        # Replace with time-optimized configurations
        new_configs = '''    # Configurations to try in order of increasing accuracy (and time)
    # Optimized to use time budget efficiently
    configurations = [
        (64, 1),    # Baseline: fast, meets target
        (96, 1),    # Better accuracy
        (128, 1),   # Good accuracy
        (96, 2),    # Higher degree
        (128, 2),   # Best accuracy within time budget
    ]
'''
        lines = lines[:start_idx] + new_configs.splitlines(keepends=True) + lines[end_idx+1:]
        break

with open('solver.py', 'w') as f:
    f.writelines(lines)
