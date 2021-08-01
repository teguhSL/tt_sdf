import os 
import numpy as np
import pickle
# import trajoptpy.math_utils as mu
import scipy.stats
class rbf():
    def __init__(self, D=39, K=60, offset=200, width=60, T=4000, reg_factor = 1e-6):
        self.D = D
        self.K = K
        self.offset = offset
        self.width = width
        self.T = T
        self.reg_factor = reg_factor

    def create_RBF(self):
        tList = np.arange(self.T)

        Mu = np.linspace(tList[0]-self.offset, tList[-1]+self.offset, self.K)
        Sigma  = np.reshape(np.matlib.repmat(self.width, 1, self.K),[1, 1, self.K])
        Sigma.shape
        Phi = np.zeros((self.T, self.K))

        for i in range(self.K):
            Phi[:,i] = scipy.stats.norm(Mu[i], Sigma[0,0,i]).pdf(tList)

        #normalize
        Phi = Phi/np.sum(Phi,axis = 1)[:,None]
        self.Phi = Phi
        return Phi

    def transform(self,trajs):
        w_trajs = []
        for traj in trajs:
            w,_,_,_ = np.linalg.lstsq(self.Phi, traj,self.reg_factor)
            w_trajs.append(w.flatten())
        return np.array(w_trajs)

    def inverse_transform(self,ws):
        trajs = []
        for w in ws:
            w = w.reshape(self.K,-1)
            traj = np.dot(self.Phi,w)
            trajs += [traj]
        return np.array(trajs)

class rbf_pca():
    def __init__(self, rbf, pca):
        self.rbf = rbf
        self.pca = pca

    def transform(self, trajs):
        w_trajs = self.rbf.transform(trajs)
        trajs_pca = self.pca.fit_transform(w_trajs)
        self.w_trajs = w_trajs
        return trajs_pca

    def inverse_transform(self, trajs_pca):
        w_trajs = self.pca.inverse_transform(trajs_pca)
        trajs = self.rbf.inverse_transform(w_trajs)
        self.w_trajs = w_trajs
        return trajs

class Regressor():
    def __init__(self, transform=None):
        self.transform = transform
        self.pca = None

    def save_to_file(self,filename):
        f = open(filename + '.pkl', 'wb')
        pickle.dump(self.__dict__,f)
        f.close()

    def load_from_file(self,filename):
        f = open(filename + '.pkl', 'rb')
        self.__dict__ = pickle.load(f)

#Nearest Neighbor Regressor
class NN_Regressor(Regressor):
    def __init__(self, transform=None, K = 1):
        self.transform = transform
        self.pca = None
        self.K = K

    def fit(self,x,y):
        self.x = x.copy()
        self.y = y.copy()

    def nearest(self,x_i):
        dists = []
        for x_j in self.x:
            dists.append(np.linalg.norm(x_i-x_j))
        dists = np.array(dists)
        sort_indexes = np.argpartition(dists, self.K)
        return sort_indexes[:self.K], dists[sort_indexes[:self.K]]

    def predict(self,x, is_transform = True):
        y_indexes,dists = self.nearest(x)
        y_curs = self.y[y_indexes,:].copy()
        y = np.mean(y_curs, axis=0)
        if is_transform:
            y_transform = self.transform.inverse_transform([y[None,:]])[0]
            return y_transform, 0
        else:
            return y, 0

#GPy GP Regressor
import GPy
class GPy_Regressor(Regressor):
    def __init__(self, dim_input, transform = None):
        self.transform = transform #whether the output should be transformed or not. Possible option: PCA, RBF, etc.
        self.dim_input = dim_input

    def fit(self,x,y, num_restarts = 10):
        self.kernel = GPy.kern.RBF(input_dim=self.dim_input, variance=0.1,lengthscale=0.3, ARD=True) + GPy.kern.White(input_dim=self.dim_input)
        self.gp = GPy.models.GPRegression(x, y, self.kernel)
        self.gp.optimize_restarts(num_restarts=num_restarts)

    def predict(self,x, is_transform = True):
        y,cov = self.gp.predict(x)
        if is_transform:
            y_transform = self.transform.inverse_transform([y[None,:]])[0]
            return y_transform, cov
        else:
            return y,cov

    def save_model(self, path, file_name): 
        np.save(os.path.join(path, file_name+'.npy'), self.gp.param_array)

    def load_model(self,x,y,path): 
        # Model creation, without initialization:
        m_load = GPy.models.GPRegression(x, y, self.kernel, initialize=False)
        m_load.update_model(False) # do not call the underlying expensive algebra on load
        m_load.initialize_parameter() # Initialize the parameters (connect the parameters up)
        m_load[:] = np.load(path) #, allow_pickle=True)[()] # Load the parameters
        m_load.update_model(True) # Call the algebra only once


# #Sparse GP Regressor
# class Sparse_GPy_Regressor(Regressor):
#     def __init__(self, num_z = 100, transform = None):
#         self.zdim = num_z
#         self.transform = transform
#     def fit(self,x,y):
#         Z = x[0:self.zdim]
#         self.sparse_gp = GPy.models.SparseGPRegression(x, y, Z=Z)
#         self.sparse_gp.optimize('bfgs')

#     def predict(self,x, is_transform = True):
#         y,cov = self.sparse_gp.predict(x)
#         if is_transform:
#             y_transform = self.transform.inverse_transform([y[None,:]])[0]
#             return y_transform, cov
#         else:
#             return y,cov


# # Straight Line Planner
# class Straight_Regressor(Regressor):
#     def __init__(self, dof, n_steps, is_transform=None):
#         self.dof = dof
#         self.n_steps = n_steps
#         self.is_transform = is_transform

#     def predict(self, init_joint, target_joint):
#         inittraj = np.empty((self.n_steps, self.dof))
#         inittraj = mu.linspace2d(init_joint, target_joint, self.n_steps)
#         return inittraj, np.array([[0]])

#     def predict_with_waypoint(self, init_joint, target_joint, waypoint=None, waypoint_step=0):
#         inittraj = np.empty((self.n_steps, self.dof))
#         inittraj[:waypoint_step + 1] = mu.linspace2d(init_joint, waypoint, waypoint_step + 1)
#         inittraj[waypoint_step:] = mu.linspace2d(waypoint, target_joint, self.n_steps - waypoint_step)
#         return inittraj, np.array([[0]])

# #DP-GLM
# import pbdlib as pbd
# class DP_GLM_Regressor(Regressor):
#     def __init__(self, n_components = 10, n_init = 20, is_transform = False):
#         self.is_transform = is_transform
#         self.n_components = n_components
# 	self.n_init = n_init
#     def fit(self,x,y):
#         self.x_joint = np.concatenate([x, y], axis=1)
#         self.n_joint = self.x_joint.shape[1]
#         self.n_in = x.shape[1]
#         self.n_out = y.shape[1]
#         self.joint_model = pbd.VBayesianGMM({'n_components':self.n_components , 'n_init':self.n_init, 'reg_covar': 0.00006 ** 2,
#          'covariance_prior': 0.00002 ** 2 * np.eye(self.n_joint),'mean_precision_prior':1e-9})
#         self.joint_model.posterior(data=self.x_joint, dp=False, cov=np.eye(self.n_joint))
#     def predict(self,x, return_gmm=True, return_more = False):
#         result = self.joint_model.condition(x, slice(0, self.n_in), slice(self.n_in, self.n_joint),return_gmm = return_gmm) #
#         if return_gmm:
#             if return_more:
#                 return self.joint_model._h, result[0], result[1]
#             else:
#                 index = np.argmax(self.joint_model._h)
#                 return result[0][index], result[1][index]
#         else:
#             return result[0], result[1]


# #BGMR
# #Same as DP-GLM but Teguh's implementation
# from ee_utils import *
# class BGMR_Regressor(Regressor):
#     def __init__(self, n_components = 10, n_init = 4, max_iter = 20, is_transform = False):
#         self.is_transform = is_transform
#         self.n_components = n_components
#         self.n_init = n_init
#         self.max_iter = max_iter


#     def fit(self,x,y,init_type='kmeans'):
#         self.x_joint = np.concatenate([x, y], axis=1)
#         self.n_joint = self.x_joint.shape[1]
#         self.n_in = x.shape[1]
#         self.n_out = y.shape[1]

#         self.gmm = GMM(D= self.n_joint , K = self.n_components)
#         self.gmm.fit(self.x_joint, max_iter = self.max_iter, n_init = self.n_init, init_type = init_type)
#         self.gmr = GMR(self.gmm, self.n_in, self.n_out)

#     def predict(self,x):
#         return self.gmr.predict(x.flatten())
