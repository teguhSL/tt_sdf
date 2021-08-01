import numpy as np 
import tensorly
from tensorly.decomposition import tensor_train as TT
from tensorly.decomposition import parafac
from tensorly import tt_to_tensor


# Find the decomposition of tensor F

# F = tsdf_vol

def apply_tt(F, rank=1, verbose=False):  
    """ Applies tensor train decomposition to tensor input F. 

    Args:
        F (numpy ndarray]): n-mode tensor 
        rank: rank to use for decomposition 
    Returns:
        : [description]
    """
    K = F.shape[0]
    d = len(F.shape)
    ttF = TT(F,rank=[1,rank,rank,1])# increase rank for better approximation rank. Note: rank[0]=1 and rank[-1]=1
    factors = ttF.factors # list of tt cores
    
    if verbose: 
        print("Number of elements in the original array: ", K**d)
        print("Number of elements in tt format: ", np.sum([ttF.rank[i]*ttF.rank[i+1]*K for i in range(d)]))
        print('Factor dimensions: {}'.format([f.shape for f in factors]))
    return factors

def apply_parafac(F, rank=1):  
    """ Applies tensor train decomposition to tensor input F. 

    Args:
        F (numpy ndarray]): n-mode tensor 
        rank: rank to use for decomposition (# increase rank for better approximation rank. 
                                            Note: rank[0]=1 and rank[-1]=1)
    Returns:
        : [description]
    """
    K = F.shape[0]
    d = len(F.shape)
    weights,factors = parafac(F,rank=rank, normalize_factors=False, orthogonalise=True)
    return factors

def orthogonalize(tensors):
    n = len(tensors)
    num_bins = [core.shape[1] for core in tensors]
    bond_dims=[core.shape[-1] for core in tensors]
    for site in range(n-1):
        dl=bond_dims[site-1] # left bond dimension
        d=bond_dims[site]   # current bond dimension
        A=tensors[site].reshape(dl*num_bins[site],d) # A is a matrix unfolded from the current tensor
        Q,R=np.linalg.qr(A)
        R/=np.linalg.norm(R) # devided by norm
        tensors[site] = Q.reshape(dl,num_bins[site],-1)
        tensors[site+1] = (R@tensors[site+1].reshape(d,-1)).reshape(-1,num_bins[site+1],bond_dims[site+1])
        bond_dims[site] = Q.shape[1] # economy QR, so the right dimension could be either dl or d
    return tensors   
        
def uncompress_tt_rep(factors, F=None, verbose=True):
    """ 
    Recosnstruct original tensor from factors of the tensor train decompositon. 

    Args:
        factors: factors of the TT 
        F: original tensor 
        verbose (bool, optional): If set to True, returns the error between 
                                  the reconstructed tensor and the original tensor. 
                                  Defaults to True.

    Returns:
        F_apprx: reconstructed tensor 
    """
    # factors = tt_rep.factors # list of tt cores
    #re-contruct the full d-dimenisonal tensor from it tt-decomposition
    # F_apprx = tt_rep.to_tensor()

    F_apprx = tt_to_tensor(factors)

    if verbose and F is not None: 
        print("Error: ",np.linalg.norm(F-F_apprx))
    
    return F_apprx
    