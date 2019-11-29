import matplotlib.pyplot as plt
import numpy as np

def plot_bvt(data):
    '''
    Method to take plot bivariate histograms of all pairs of coordinates in data.
    
    Args:
    data: Array(float, n x p).
    '''
    
    p = data.shape[1]
    for u in range(p):
        for v in range(u, p):
            H, xedges, yedges = np.histogram2d(data[:, u], data[:, v], bins=30, normed=True)
            fig = plt.figure(figsize=(6, 4))
            X, Y = np.meshgrid(xedges, yedges)
            plt.pcolormesh(X, Y, -H, cmap='gray')



def plot_unvt(data):
    '''
    Method to take plot univariate histograms of all coordinates in data.
    
    Args:
    data: Array(float, n x p).
    '''
    
    p = data.shape[1]
    for u in range(p):
        plt.hist(data[:, u])
        plt.show()
