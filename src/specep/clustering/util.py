from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import numpy as np

class ECFactory:
    _clustering = {
        "GaussianMixture": GaussianMixture,
        "KMeans": KMeans,
    }

    @classmethod
    def create(cls, clustering, **cluster_params):
        if clustering in cls._clustering:

            if clustering == "GaussianMixture":
                cluster_params.update({'n_components':cluster_params['n_clusters']})
                cluster_params.pop('n_clusters')

            clustering_class = cls._clustering[clustering]
            return clustering_class(**cluster_params)
        else:
            raise ValueError(f"Unknown clustering:{clustering}")
        

def sample_from_component(clustering: GaussianMixture|KMeans, cluster_idx: int, n_samples: int = None):

    if isinstance(clustering, GaussianMixture):

        if n_samples is None:
            raise ValueError("Specify number of samples you want to get from GaussianMixture component")
        mean = clustering.means_[cluster_idx]
        cov = clustering.covariances_[cluster_idx]
        
        if clustering.covariance_type == 'full':
            cov_k = cov
        # elif gmm_clustering.covariance_type == 'tied':
        #     cov_k = gmm.covariances_
        # elif gmm_clustering.covariance_type == 'diag':
        #     cov_k = np.diag(cov)
        # elif gmm_clustering.covariance_type == 'spherical':
        #     cov_k = np.eye(gmm.means_.shape[1]) * cov
        else:
            raise ValueError("Unsupported covariance type")
        return np.random.multivariate_normal(mean, cov_k, n_samples)
    
    elif isinstance(clustering, KMeans):
        center = clustering.cluster_centers_[cluster_idx]
        return np.array([center])
    else:
        raise NotImplementedError(f"Unsupported clustering type {clustering}")