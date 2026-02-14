import numpy as np
data = np.load('oracle_output/reference.npz')
print('Keys:', list(data.keys()))
for key in data.keys():
    print(f'{key}: shape={data[key].shape}, dtype={data[key].dtype}')
    if data[key].shape[0] < 10:
        print(f'  First few values: {data[key][:5]}')
