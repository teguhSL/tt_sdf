import os
import time
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

from costs import *
from env_creator import EnvCreator, generate_sdf_rep
from ocp import *
from ocp_utils import *
from regression import GPy_Regressor, NN_Regressor, rbf
from tensor_decomp import apply_tt, uncompress_tt_rep
from visualization_utils import (plot_dataset_overview, plot_traj_and_obs_3d,
                                 plot_traj_projections)


def neg_likelihood(y_true, y_pred):
    return -y_pred.log_prob(y_true)


def plot_loss(history, y_range=[0,1]):
    fig = plt.figure(figsize=(10,8))
    plt.plot(history.epoch, history.history['loss'], label='loss')
    plt.plot(history.epoch, history.history['val_loss'], label='val_loss')
    plt.ylim(y_range)
    plt.xlabel('Epoch')
    plt.ylabel('Error [MSE]')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)


def eval_collision_geometric(data_sample, y_traj, margin=1e-3): 
    collision_bool = []
    collision_dists = []
    for x in y_traj: 
        collision = False 
        obst_dists = []
        for obs in data_sample['obstacles']:
            p, r = obs['pos'], obs['rad']
            dist = np.linalg.norm(p-x)
            obst_dists.append(dist-r)
            
            if dist < (r+margin): # add a margin to the obstacle surface (at least radius of point mass)
                collision = True 
        collision_bool.append(collision)
        collision_dists.append(np.min(obst_dists))
        
    return np.array(collision_bool), np.array(collision_dists)


def eval_collision_ddp(data_sample, y_traj):
    '''
    DDP cost formulation 
    '''
    # Create dummy linear system for DDP cost calculation 
    dt = 0.05  #duration of 1 time step
    Dx_temp, Du = 6, 3 #dimensions of x and u
    dof = 3

    #Define the matrix A and B to define a double integrator
    A = np.eye(Dx_temp)
    A[:dof,dof:] = np.eye(dof)*dt
    B = np.zeros((Dx_temp, Du))
    B[:dof,:] = 0.5*np.eye(Du)*(dt**2)
    B[dof:, :] = np.eye(Du)*dt
    lin_sys = LinearSystem(A, B)
    collision_cost = []
    for obs in data_sample['obstacles']: 
        c = CostModelCollisionSphere(lin_sys, obs['pos'], obs['rad'], w_obs=obs['w'], d_margin=obs['d_marg'])
        # sum up costs for each point in trajectory -> should sum up to 0 if no collisions 
        for x in y_traj:  
            u_dummy = np.array([0,0,0])
            collision_cost.append(c.calc(x, u_dummy))
                            
    return np.sum(collision_cost)           


class Learner(object):
    def __init__(self, exp_name, voxel_res, decomp_rank, start_goal_fixed, target_dim_red='pca', 
                 use_tt_rep=True, use_cp_rep=False, use_raw_input=False, use_pca_rep=False, 
                 filter_by_ddp_status=True, filter_by_geom_coll=False, verbose=True): 

        self.data_dir  = os.path.join(os.getcwd(), 'data/envs/%s/'%exp_name)
        self.exp_name = exp_name 
        self.raw_data = []
        self.full_data = []
        self.rank = decomp_rank
        self.voxel_res = voxel_res
        self.grid_boundaries = \
            np.asarray([(-1,1),   # x
                        (-1,1),   # y
                        (-1,1)]   # z
                        ) # 3x2, columns: (min, max) in world coordinates in meters
        self.filter_by_ddp_status = filter_by_ddp_status
        self.filter_by_geom_coll = filter_by_geom_coll
        self.use_tt_rep = use_tt_rep
        self.use_cp_rep = use_cp_rep 
        self.use_pca_rep = use_pca_rep
        self.use_raw_input = use_raw_input
        self.start_goal_fixed = start_goal_fixed 
        self.Dx = 3 #dimensions of point mass system (xyz)
        self.T = 101 #number of time steps
        self.verbose = verbose 
        self.target_dim_red = target_dim_red 

        self.load_data(exp_name)        
        self.model = None 


    def load_data(self, exp_name):
        exp_files = [f for f in os.listdir(self.data_dir) if isfile(join(self.data_dir, f)) and f.split('.')[-1] == 'npy']
        for f in exp_files: 
            self.raw_data.append(np.load(join(self.data_dir,f), allow_pickle=True)[()])
        # filter raw data 
        self.filter_trajs = self.filter_unvalid_trajs(by_ddp_status=self.filter_by_ddp_status, 
                                                      by_geom_coll=self.filter_by_geom_coll)
    

    def filter_unvalid_trajs(self, by_ddp_status=True, by_geom_coll=False): 
        if by_ddp_status and not by_geom_coll: # filter by ddp status only 
            if self.verbose:
                print('---> Filter by ddp status')
            # Trajectories that have 'True' status from ddp
            final_status  = np.array([data['status'] for data in self.raw_data]) 
        
        elif by_geom_coll: # filter for geometric collisions
            if self.verbose: 
                print('---> Filter by geometric collisions')
            ddp_status = np.array([data['status'] for data in self.raw_data])        
            geom_status = []

            for sample in self.raw_data: # iterate over all trajectories 
                x = sample['xs'][:,:3].reshape(-1,self.Dx) 
                geom_collision_bool,_ = eval_collision_geometric(sample, x)
                geom_status.append(np.any(geom_collision_bool))
            if by_ddp_status:   
                final_status = np.logical_and(ddp_status, np.asarray(geom_status))
            else:
                if self.verbose: 
                    print('---> Filter by geometric collisions and ddp status')
                final_status = geom_status
        else: # use all data instances 
            if self.verbose:
                print('\n\nNo filtering, using %d instances'%len(self.raw_data))
            return self.raw_data

        filter_trajs = np.asarray(self.raw_data)[final_status]
        print('\nFiltered data: keeping {}/{} instances'.format(len(filter_trajs), len(self.raw_data)))
        return filter_trajs 


    def build_features(self, use_tt_orth=True):
        self.xs_set = np.array([data['xs'][:,:3] for data in self.filter_trajs])
        self.init_positions = np.array([data['x0'][:3] for data in self.filter_trajs])
        self.goal_positions = np.array([data['xT'][:3] for data in self.filter_trajs])
        if self.use_tt_rep: 
            if self.verbose:
                print('--> Using Tensor Train Decomposition')
        elif self.use_cp_rep: 
            if self.verbose: 
                print('--> Using CP Decomposition')
        elif self.use_pca_rep: 
            if self.verbose: 
                print('--> Using PCA Dim Red.')
        elif self.use_raw_input: 
            if self.verbose: 
                print('--> Using raw data input')

        t0 = time.time()
        for sample_dct in self.filter_trajs: 
            obstacles = sample_dct['obstacles']
            # TT decomposition 
            if self.use_tt_rep: 
                sdf_vol, tt_sdf_vol, tt_sdf_orth, _ = \
                    generate_sdf_rep(obstacles, 
                                    grid_bounds=self.grid_boundaries, 
                                    voxel_res=self.voxel_res, 
                                    tt_rank=self.rank)    
                if use_tt_orth:
                    key = 'tt_sdf_orth' 
                    sample_dct[key] = tt_sdf_orth # factors of tt after canonicalization                 
                else: 
                    key = 'tt_sdf' 
                    sample_dct[key] = tt_sdf_vol # factors of the tensor train decomposition of the sdf tensor 
            
            # CP decomposition 
            elif self.use_cp_rep: 
                key = 'cp_sdf'
                sdf_vol, cp_sdf_vol, _ = generate_sdf_rep(obstacles, 
                                                          grid_bounds=self.grid_boundaries, 
                                                          voxel_res=self.voxel_res, 
                                                          tt_rank=self.rank, use_cp=True)    
                sample_dct[key] = cp_sdf_vol # factors of the tensor train decomposition of the sdf tensor 
            # Raw input 
            elif self.use_raw_input or self.use_pca_rep: 
                sdf_vol,_ = generate_sdf_rep(obstacles, 
                                             grid_bounds=self.grid_boundaries, 
                                             voxel_res=self.voxel_res,
                                             use_cp=False, sdf_only=True)

            sample_dct['sdf'] = sdf_vol
            self.full_data.append(sample_dct)
        t1 = time.time()
        elapsed = t1-t0
        print("Computation time: %.4f secs"%elapsed)

        self.raw_env_set = np.array([data['sdf'].flatten() for data in self.full_data])

        if self.use_pca_rep: 
            key = 'pca_sdf'
            K = 320
            pca = PCA(n_components=K)
            self.comp_env_set = pca.fit_transform(self.raw_env_set)
            cum_exp_var = pca.explained_variance_ratio_.cumsum()[-1]
            if self.verbose: 
                print('Cum. explained variance by %d principal comps. = %f'%(K, cum_exp_var))
            
        elif not self.use_raw_input: 
            self.comp_env_set = np.array([np.concatenate([np.asarray(f).flatten() for f in data[key]]) for data in self.full_data])        

        # if self.verbose: 
        print('Shape of uncompressed environment data input: {}'.format(self.raw_env_set.shape))
        if not self.use_raw_input: 
            print('Shape of compressed environment data input: {}'.format(self.comp_env_set.shape))


    def train_test_split(self, test_size=0.3):
        n_samples = len(self.x_inputs)
        indices = np.arange(n_samples)
        self.x_train, self.x_test, self.y_train, self.y_test, self.train_idx, self.test_idx = \
                train_test_split(self.x_inputs, self.x_outputs, indices, random_state=3, test_size=test_size)
        if self.verbose: 
            print('Train-Test-Split:\nx_train: ',self.x_train.shape,'\ny_train: ', self.y_train.shape,
                                    '\nx_test: ', self.x_test.shape,'\ny_test: ', self.y_test.shape)


    def construct_in_out(self, use_tt_orth=True):
        self.build_features(use_tt_orth=use_tt_orth)
        if (self.use_tt_rep or self.use_cp_rep or self.use_pca_rep) and self.start_goal_fixed: 
            self.x_inputs = self.comp_env_set
        elif (self.use_tt_rep or self.use_cp_rep or self.use_pca_rep) and not self.start_goal_fixed: 
            self.x_inputs = np.concatenate([self.comp_env_set, self.init_positions, self.goal_positions], axis=1)
        elif self.use_raw_input and self.start_goal_fixed: 
            self.x_inputs = self.raw_env_set
        elif self.use_raw_input and not self.start_goal_fixed: 
            self.x_inputs = np.concatenate([self.raw_env_set, self.init_positions, self.goal_positions], axis=1)
        else: 
            obs_set = [data['obstacles'] for data in self.filter_trajs]
            r_set = np.array([[obs['rad'] for obs in obstacles] for obstacles in obs_set])
            p_set = np.array([[obs['pos'] for obs in obstacles] for obstacles in obs_set])
            self.x_inputs = np.concatenate([r_set[:,None], p_set], axis=1) # use obstacle parametrization instead 

        N = len(self.x_inputs)
        self.D_in = self.x_inputs.shape[1]
        # if self.verbose: 
        print('Input with dimensions: {}'.format(self.x_inputs.shape))

        self.x_outputs = self.xs_set.reshape(N,-1)


    def compress_target(self, K=15):
        if self.target_dim_red == 'pca':
            self.pca = PCA(n_components=K)
            self.y_train_red = self.pca.fit_transform(self.y_train)
            self.y_test_red = self.pca.transform(self.y_test)
            cum_exp_var = self.pca.explained_variance_ratio_.cumsum()[-1]
            if self.verbose: 
                print('Using pca for target compression with params: K=%d'%(K))
                print('Cum. explained variance by %d principal comps. = %f'%(K, cum_exp_var))
        elif self.target_dim_red == 'rbf':
            self.rbf_transform = rbf(D=3, K = 8, offset = 20, width = 15, T = T)
            Phi = rbf_transform.create_RBF()
            # plt.plot(Phi)
            # plt.show()
            self.y_train_red = rbf_transform.transform(self.y_train.reshape(-1,T,Dx))
            self.y_test_red = rbf_transform.transform(self.y_test.reshape(-1,T,Dx))
        elif self.target_dim_red == None:
            self.y_train_red = self.y_train
            self.y_test_red = self.y_test
        else: 
            raise NotImplementedError

        self.D_out = self.y_train_red.shape[1]


    def save_data(self): 
        data = dict()
        if self.target_dim_red == 'pca':
            data['pca'] = self.pca
        elif self.target_dim_red == 'rbf':
            data['rbf'] = self.rbf_transform
            
        data['x_inputs'] = self.x_inputs
        data['x_outputs'] = self.x_outputs
        data['obstacles'] = [data['obstacles'] for data in self.filter_trajs]
        save_path = os.path.join(os.getcwd(), 'data/training_data/data_%s'%exp_name)
        np.save(save_path, data)


    def learn_nn(self, lr=0.0002, num_epochs=300, batch_size=8, load=False, file_name = ''):
        save_path = os.path.join(os.getcwd(),'data/models/nn_'+file_name+'.h5')  
        if load:
            nn = tf.keras.models.load_model(save_path)
            return nn
        
        if self.verbose: 
            print('learning_rate: %.6f \nnum_epochs: %d\ns'%(lr, num_epochs))
        nn = Sequential([
            Dense(256, activation='relu', input_shape=(self.D_in,)),
            Dense(256, activation='relu'),#, kernel_constraint=MaxNorm(3)),
        #     Dropout(rate=0.2),
            Dense(self.D_out)
        ])
        nn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse') # set loss and optimizer 
        t0 = time.time()
        
        nn_history = nn.fit(self.x_train, self.y_train_red,
                            batch_size=batch_size, validation_split=0.2, 
                            epochs=num_epochs, callbacks=[tf.keras.callbacks.EarlyStopping(patience=30)], 
                            verbose=0)
        t1 = time.time()
        print('Elapsed time in secs:  %.4f'%(t1 - t0))
        hist = pd.DataFrame(nn_history.history)
        hist['epoch'] = nn_history.epoch
        print(hist.tail())
        plot_loss(nn_history, y_range=[0,1])
        print('Saving NN..')
        nn.save(save_path)
        return nn 
        

    def learn_knn(self): 
        knn = KNeighborsRegressor(1)
        knn.fit(self.x_train, self.y_train_red)
        return knn 


    def learn_gpr(self, save_model=False): 
        t0 = time.time()
        gpr = GPy_Regressor(self.D_in)
        gpr.fit(self.x_train, self.y_train_red, num_restarts=3) # num_restarts should avoid local minima in optimization process 
        t1 = time.time()
        print('Elapsed time in secs:  %.4f'%(t1 - t0))
        if save_model: 
            save_path = os.path.join(os.getcwd(),'data/models')    
            gpr.save_model(save_path, self.exp_name)
        return gpr 


    def learn_mdn(self, n_comps=15, lr=0.0002, num_epochs=300, batch_size=8,verbose=0, load=False, file_name = ''): 
        t0 = time.time()
        tf.keras.backend.set_floatx('float64')
        tfd = tfp.distributions
        tfpl = tfp.layers
        n_comp_params_size = tfpl.IndependentNormal.params_size(event_shape=(self.D_out,))
        if self.verbose: 
            print('learning_rate: %.6f \nnum_epochs: %d\nnum_components: %d \n'%(lr, num_epochs, n_comps))
        params_size = tfpl.MixtureSameFamily.params_size(num_components=n_comps, component_params_size=n_comp_params_size)
        mdn = Sequential([
            Dense(256, activation='relu', input_shape=(self.D_in,), kernel_regularizer = tf.keras.regularizers.l2(1e-2)),
            Dense(256, activation='relu', kernel_regularizer = tf.keras.regularizers.l2(1e-2)),
            Dense(params_size),
            tfpl.MixtureSameFamily(n_comps, tfpl.IndependentNormal(event_shape=(self.D_out,)))
        ])
        mdn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=neg_likelihood)
        
        save_path = os.path.join(os.getcwd(),'data/models/mdn_'+file_name+'.h5')  
        if load:
            mdn.load_weights(save_path)
        else:    
            history = mdn.fit(self.x_train, self.y_train_red, 
                          batch_size=batch_size, validation_split=0.2, 
                          epochs=num_epochs, callbacks=[tf.keras.callbacks.EarlyStopping(patience=30)], 
                          verbose=verbose)
            mdn.save_weights(save_path)
            t1 = time.time()
            print('Elapsed time in secs:  %.4f'%(t1 - t0))
            hist = pd.DataFrame(history.history)
            hist['epoch'] = history.epoch
            print(hist.tail())
            plot_loss(history, y_range= [0,20])
        return mdn 


    def predict(self, model, method, x, n_sample=1, test_idx = None): 
        N = len(x)
        if method == 'gpr':
            y_pred, y_cov = model.predict(x, False)
        elif method == 'nn':
            y_pred = model.predict(x)
        elif method == 'mdn': 
            y_pred = np.array(model(x).sample(n_sample))
        elif method == 'knn':
            y_pred = model.predict(x)

        #Inverse transform, if using PCA or rbf
        if self.target_dim_red == 'pca':
            y_traj = self.pca.inverse_transform(y_pred)
        elif self.target_dim_red == 'rbf':
            y_traj = self.rbf_transform.inverse_transform(y_pred)
        elif self.target_dim_red == None:
            y_traj = np.array(y_pred)
        else: 
            raise NotImplementedError
            
        if method == 'mdn':
            if n_sample == 1:
                y_traj = y_traj[0]
            else:
                y_traj_best = []
                for i in range(N):
                    x_i = x[i]
                    full_data_idx = test_idx[i] # map idx from test set to full dataset idx   
                    obstacles = self.full_data[full_data_idx]['obstacles']
                    y_traj_i = y_traj[:,i].reshape(n_sample,-1,self.Dx)
                    print(i, y_traj_i.shape, x_i.shape)
                    y_traj_i_best = get_best_mdn_prediction(y_traj_i, x_i, obstacles)
                    y_traj_best.append(y_traj_i_best)
                y_traj = np.array(y_traj_best).reshape(N,-1)

        return y_traj, y_pred 

    
    def MSE(self, y_true, y_pred):
        N = len(y_true)
        error = (y_pred-y_true)**2
        goal_mse = np.mean(error[:,-3:])
        total_mse = np.mean(error.reshape(N, -1))
        if self.verbose: 
            print('\n MSE: \n - total: {:.4f}\n - goal positions: {:.4f}'.format(total_mse, goal_mse))
        return total_mse, goal_mse 
    

    def collision_eval(self, y_preds, test_idx): 
        num_samples = len(y_preds)
        geometric_collision_check = []
        geometric_collision_dists = []

        for i in range(num_samples): # iterate over all trajectories 
            full_data_idx = test_idx[i] # map idx from test set to full dataset idx           
            y_traj = y_preds[i].reshape(-1,3)
            data = self.full_data[full_data_idx]
            
            geom_collision_bool, geom_collision_dists = eval_collision_geometric(data, y_traj)
            collision_inds = np.where(geom_collision_bool)[0]
            if collision_inds.size != 0:
                avg_collision_dist = np.mean(geom_collision_dists[collision_inds])
                geometric_collision_dists.append(avg_collision_dist)         
            geometric_collision_check.append(np.any(geom_collision_bool))
        # geom collision eval 
        geom_no_collision_inds = np.where(~np.asarray(geometric_collision_check))[0]
        geom_collision_inds = np.where(np.asarray(geometric_collision_check))[0]
        
        geom_no_collision = 1-np.asarray(geometric_collision_check).sum()/len(geometric_collision_check)
        geom_mean_coll_dist = np.mean(geometric_collision_dists)
        geom_med_coll_dist = np.median(geometric_collision_dists)
        if self.verbose: 
            print('\nCollision Metric: ')
            print('Geometric - Share of trajectories with no collisions: %.4f'%geom_no_collision)
            print('Dist to surface when collision: Avg: %.4f, Med: %.4f'%(geom_mean_coll_dist, geom_med_coll_dist))
        return geom_no_collision, geom_mean_coll_dist, geom_med_coll_dist


    def evaluate(self, model, method, x_test, y_test, test_idx, n_sample = 1):
        y_preds, _ = self.predict(model, method, x_test, n_sample, test_idx)
        total_mse, goal_mse =  self.MSE(y_test, y_preds)
        
        coll_free, _, med_coll_free_dist = self.collision_eval(y_preds, test_idx)
        print('%.4f\n%.4f\n%.4f\n'%(total_mse,goal_mse, coll_free))
        # TODO: write to logfile 


if __name__ == '__main__': 
    datasets = {
        # 1: 'start_goal_fixed_1000samples', # Dataset 1
#         2: 'var_start_goal_3000samples',  # Dataset 2
#        3: 'var_start_goal_1000samples_no_waypoint', # Dataset 3
        # 4: 'mult_obs_fixed_start_goal_1000samples_teguh_noviapoint', # Dataset 4
#         5: 'mult_obs_var_start_goal_1000samples_teguh', # Dataset 5
        6: 'mult_obs_var_start_goal_1000samples_teguh_noviapoint' # Dataset 6
    }
    # exp_name = 'mult_obs_var_start_goal_1000samples_teguh_noviapoint'
    for key, exp_name in datasets.items(): 
        print('\n---------------------------------\nDataset %d: %s \n---------------------------------\n'%(key,exp_name))
        voxel_res = 0.05
        decomp_rank = 3
        if 'fixed' in exp_name: 
            start_goal_fixed = True
        else: 
            start_goal_fixed = False
        
        # for decomp_rank in [2,3,5,10]: 
        print('=================================')
        print('Decomposition rank: %d\nVoxel_res: %.2f \n'%(decomp_rank, voxel_res))

        ## TT-Setting 
        traj_learner = Learner(exp_name, voxel_res, decomp_rank, start_goal_fixed, use_tt_rep=True,   
                                filter_by_ddp_status=True, filter_by_geom_coll=False, verbose=False,
                                target_dim_red='pca')
        # ## CP Setting 
        # traj_learner = Learner(exp_name, voxel_res, decomp_rank, start_goal_fixed, use_tt_rep=False, use_cp_rep=True)

        # ## Raw Input 
        # traj_learner = Learner(exp_name, voxel_res, decomp_rank, start_goal_fixed, use_tt_rep=False, use_cp_rep=False, use_raw_input=True)

        traj_learner.construct_in_out(use_tt_orth=True)
        traj_learner.train_test_split()
        traj_learner.compress_target(K=15)
        traj_learner.save_data()

        print('\n----------------- \nNN')
        nn = traj_learner.learn_nn(lr=0.0002, num_epochs=300, batch_size=8, load=False, file_name = exp_name)
        traj_learner.evaluate(nn, 'nn', traj_learner.x_test, traj_learner.y_test, traj_learner.test_idx)

#         print('\n----------------- \nkNN')
#         knn = traj_learner.learn_knn()
#         traj_learner.evaluate(knn, 'knn', traj_learner.x_test, traj_learner.y_test, traj_learner.test_idx)
        
        # print('\n----------------- \nGPR')
        # gpr = traj_learner.learn_gpr()
        # traj_learner.evaluate(gpr, 'gpr')

        print('\n----------------- \nMDN')
        mdn = traj_learner.learn_mdn(n_comps=10, lr=0.0002, num_epochs=300, batch_size=8,  verbose=0, load=False, file_name = exp_name)
        traj_learner.evaluate(mdn, 'mdn', traj_learner.x_test, traj_learner.y_test, traj_learner.test_idx, n_sample = 10) 



        ### PCA Input 
        # traj_learner = Learner(exp_name, voxel_res, decomp_rank, start_goal_fixed, use_tt_rep=False, use_pca_rep=True,  
        #                 filter_by_ddp_status=True, filter_by_geom_coll=False)
        # traj_learner.construct_in_out(use_tt_orth=True)
