from joblib import Parallel, delayed

def warmup_worker_pool(ncores):
    Parallel(n_jobs=ncores)(delayed(lambda: None)() for _ in range(ncores))
