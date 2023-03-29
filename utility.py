import numpy as np

def bioassayfun(w, df):
    # function to be optimized
    # defines the negative log posterior = log likelihood where prior is uniform
    z = w[0] + w[1]*df['x']
    return -np.sum(df['y']*(z) - df['n']*np.log1p(np.exp(z))) 

def logl(data, a, b):
    # defines the log likelihood
    x, n, y = np.array(data['x']), np.array(data['n']), np.array(data['y'])
    a = a.reshape(-1, 1)  # Reshape a to have one column
    b = b.reshape(-1, 1)  # Reshape b to have one column
    return np.sum(y * (a + b * x) - n * np.log1p(np.exp(a + b * x)), axis=1) # returns data points in a 1D np array