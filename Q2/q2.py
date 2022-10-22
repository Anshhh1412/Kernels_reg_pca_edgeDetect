import matplotlib.pyplot as plt
import numpy as np

def kernel_pca(X: np.ndarray, kernel: str) -> np.ndarray:
    '''
    Returns projections of the the points along the top two PCA vectors in the high dimensional space.

        Parameters:
                X      : Dataset array of size (n,2)
                kernel : Kernel type. Can take values from ('poly', 'rbf', 'radial')

        Returns:
                X_pca : Projections of the the points along the top two PCA vectors in the high dimensional space of size (n,2)
    '''
    if kernel=='poly':
        c = 1
        d = 1
        X_k = (np.matmul(X,X.T)+c)**d

    elif kernel=='rbf':
        x1 = np.sum(X**2, axis=-1)
        y1 = np.sum(X**2, axis=-1)
        gamma = 15
        X_k =np.exp(-gamma*(x1[:,None] + y1[None,:] - 2*np.dot(X,X.T)))
   
    elif kernel=='radial':
        R = np.linalg.norm(X, axis=1)
        R = np.reshape(R, (R.shape[0], 1))
        T = np.arctan2(X[:,1], X[:,0])
        T = np.reshape(T, (T.shape[0], 1))
        R = np.matmul(R, R.T)
        T = np.matmul(T, T.T)
        X_k = R + T
       
    
    mean = np.mean(X_k, axis= -1)
    mean_data = X_k - mean
    cov = np.cov(mean_data.T)
    eig_val, eig_vec = np.linalg.eigh(cov)
    indices = np.arange(0,len(eig_val), 1)
    indices = ([x for _,x in sorted(zip(eig_val, indices))])[::-1]
    eig_val = eig_val[indices]
    eig_vec = eig_vec[:,indices]
    X_pca = np.dot(mean_data, eig_vec[:,:2]) 
    
    # print(X.shape[0])
    # print(X_pca.shape)
    return X_pca
    pass

if __name__ == "__main__":
    from sklearn.datasets import make_moons, make_circles
    from sklearn.linear_model import LogisticRegression
  
    X_c, y_c = make_circles(n_samples = 500, noise = 0.02, random_state = 517)
    X_m, y_m = make_moons(n_samples = 500, noise = 0.02, random_state = 517)

    X_c_pca = kernel_pca(X_c, 'radial')
    X_m_pca = kernel_pca(X_m, 'rbf')
    
    plt.figure()
    plt.title("Data")
    plt.subplot(1,2,1)
    plt.scatter(X_c[:, 0], X_c[:, 1], c = y_c)
    plt.subplot(1,2,2)
    plt.scatter(X_m[:, 0], X_m[:, 1], c = y_m)
    plt.show()

    plt.figure()
    plt.title("Kernel PCA")
    plt.subplot(1,2,1)
    plt.scatter(X_c_pca[:, 0], X_c_pca[:, 1], c = y_c)
    plt.subplot(1,2,2)
    plt.scatter(X_m_pca[:, 0], X_m_pca[:, 1], c = y_m)
    plt.show()
