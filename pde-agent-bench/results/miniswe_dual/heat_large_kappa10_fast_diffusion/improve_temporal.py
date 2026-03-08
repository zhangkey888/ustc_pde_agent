with open('solver.py', 'r') as f:
    lines = f.readlines()

# Find the dt assignment section
for i in range(len(lines)):
    if "dt = dt_suggested" in lines[i]:
        # Replace with adaptive dt based on available time
        lines[i] = '''        # Adaptive time stepping: use smaller dt if we have time budget
        dt = dt_suggested
        # Start with suggested dt, reduce if we have time
        elapsed_so_far = time.time() - start_time
        time_remaining = time_limit - elapsed_so_far
        configs_remaining = len(configurations) - config_idx
        
        # If we have plenty of time, use smaller dt for better accuracy
        if time_remaining > 5.0 and configs_remaining <= 2:  # Last few configs with plenty of time
            dt = dt_suggested / 2.0  # Half the time step for better temporal accuracy
        
        n_steps = int(np.ceil(t_end / dt))
        dt = t_end / n_steps  # Adjust to exactly reach t_end
'''
        break

with open('solver.py', 'w') as f:
    f.writelines(lines)
