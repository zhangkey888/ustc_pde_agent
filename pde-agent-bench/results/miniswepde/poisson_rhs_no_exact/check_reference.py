import numpy as np
data = np.load('oracle_output/reference.npz')
print('Keys:', list(data.keys()))
for key in data.keys():
    print(f'{key}: shape={data[key].shape}, dtype={data[key].dtype}')
    if data[key].ndim <= 2:
        print(f'  min={data[key].min():.6f}, max={data[key].max():.6f}, mean={data[key].mean():.6f}')
