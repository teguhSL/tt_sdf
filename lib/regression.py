import os 
import numpy as np
import pickle
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
