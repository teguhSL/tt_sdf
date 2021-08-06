import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pybullet as p
import time
import os
from os.path import isfile, join
from ocp import *
from ocp_sys import * 
from costs import *


#for data loading
def load_data(exp_name):
    data_dir = '../../data/envs/%s/'%exp_name
    exp_files = [f for f in os.listdir(data_dir) if isfile(join(data_dir, f)) and f.split('.')[-1] == 'npy']

    raw_data = []
    for f in exp_files: 
        raw_data.append(np.load(join(data_dir,f), allow_pickle=True)[()])
    print('The number of data points: ' + str(len(raw_data)))
    return raw_data

#for pybullet

def plot_traj(xs, obj_id, dt = 0.01):
    for x in xs:
        p.resetBasePositionAndOrientation(obj_id, x[:3], (0,0,0,1))
        time.sleep(dt)
        
def setup_pybullet(obj_id, target_id, x0, x_target):     
    p.resetBasePositionAndOrientation(obj_id, x0[:3], (0,0,0,1))
    p.resetBasePositionAndOrientation(target_id, x_target[:3], (0,0,0,1))
    return obj_id,target_id
 
def modify_obstacle(obstacle_ids, obstacles):
    for obs_id in obstacle_ids:
        p.removeBody(obs_id)
    obstacle_ids = create_obstacles(obstacles)
    return obstacle_ids

def create_obstacles(obstacles):
    obstacle_ids = []
    for obstacle in obstacles:
        if 'obs_type' in list(obstacle.keys()):
            obs_type = obstacle['obs_type']
        else:
            obs_type = p.GEOM_SPHERE
        
        if obs_type == p.GEOM_SPHERE:
            _,_,obs_id = create_primitives(obs_type, radius =obstacle['rad'], basePosition=np.zeros(3))
        elif obs_type == p.GEOM_BOX:
            _,_,obs_id = create_primitives(obs_type, halfExtents=obstacle['halfExtents'], basePosition=np.zeros(3))
        elif obs_type == p.GEOM_CAPSULE:
            _,_,obs_id = create_primitives(obs_type, radius=obstacle['rad'], length=obstacle['length'], basePosition=np.zeros(3))
            
        p.resetBasePositionAndOrientation(obs_id, obstacle['pos'], (0,0,0,1))    
        obstacle['obs_id'] = obs_id
        obstacle_ids.append(obs_id)
    return obstacle_ids
        
def init_pybullet(x0, x_target, obstacles):
    '''
    Return: obj_id, init_id, target_id, border_id, obstacle_ids
    '''
    p.resetSimulation()

    _,_,obj_id = create_primitives(rgbaColor=[0,0,1,1],radius = 0.04)
    _,_,init_id = create_primitives(radius = 0.04, rgbaColor=[1,0,0,0.5])
    _,_,target_id = create_primitives(radius = 0.04, rgbaColor=[0,1,0,0.5])
    _,_,border_id = create_primitives(rgbaColor=[1,0,0,0.1], shapeType=p.GEOM_BOX, halfExtents=[1.,1.,1.])
    obstacle_ids = create_obstacles(obstacles)
    
    p.resetBasePositionAndOrientation(init_id, x0[:3], (0,0,0,1))
    p.resetBasePositionAndOrientation(target_id, x_target[:3], (0,0,0,1))
    return obj_id, init_id, target_id, border_id, obstacle_ids


def plot_gaussian_2D(mu, sigma,ax=None,color=[0.7,0.7,0.],alpha=1.0, label='label'):
    if ax is None:
        fig,ax = plt.subplots()
    eig_val, eig_vec = np.linalg.eigh(sigma)
    std = np.sqrt(eig_val)*2
    angle = np.arctan2(eig_vec[1,0],eig_vec[0,0])
    ell = Ellipse(xy = (mu[0], mu[1]), width=std[0], height = std[1], angle = np.rad2deg(angle))
    ell.set_facecolor(color)
    ell.set_alpha(alpha)
    ell.set_label(label)
    ax.add_patch(ell)
    return

def Rotz(angle):
    A = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
    return A

def compute_covariance(radius, ori):
    A = Rotz(ori)
    Sigma = np.diag(radius**2)
    Sigma = A.T.dot(Sigma).dot(A) 
    return Sigma

def create_primitives(shapeType=2, rgbaColor=[1, 1, 0, 1], pos = [0, 0, 0], radius = 1, length = 2, halfExtents = [0.5, 0.5, 0.5], baseMass=1, basePosition = [0,0,0]):
    visualShapeId = p.createVisualShape(shapeType=shapeType, rgbaColor=rgbaColor, visualFramePosition=pos, radius=radius, length=length, halfExtents = halfExtents)
    collisionShapeId = p.createCollisionShape(shapeType=shapeType, collisionFramePosition=pos, radius=radius, height=length, halfExtents = halfExtents)
    bodyId = p.createMultiBody(baseMass=baseMass,
                      baseInertialFramePosition=[0, 0, 0],
                      baseVisualShapeIndex=visualShapeId,
                      baseCollisionShapeIndex=collisionShapeId,    
                      basePosition=basePosition,
                      useMaximalCoordinates=True)
    return visualShapeId, collisionShapeId, bodyId


def get_joint_limits(robot_id, dof):
    limit = np.zeros((2, dof))
    for i in range(dof):
        limit[0,i] = p.getJointInfo(robot_id, i)[8]
        limit[1,i] = p.getJointInfo(robot_id, i)[9]
    return limit

def plot_pose_matrix(ax,T,label=None,scale=1):
    base = T[:3,-1]
    orn = T[:3,:3]
    ax.plot([base[0],base[0]+scale*orn[0,0]],[base[1],base[1]+scale*orn[1,0]],[base[2],base[2]+scale*orn[2,0]],c='r')
    ax.plot([base[0],base[0]+scale*orn[0,1]],[base[1],base[1]+scale*orn[1,1]],[base[2],base[2]+scale*orn[2,1]],c='g')
    ax.plot([base[0],base[0]+scale*orn[0,2]],[base[1],base[1]+scale*orn[1,2]],[base[2],base[2]+scale*orn[2,2]],c='b')
    
    if label is not None:
        ax.text(base[0],base[1],base[2],label)

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
def quat2Mat(quat):
    return np.array(p.getMatrixFromQuaternion(quat)).reshape(3,3)


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
    


#for tensorflow

def plot_loss(history, y_range=[0,1]):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim(y_range)
    plt.xlabel('Epoch')
    plt.ylabel('Error [MSE]')
    plt.legend()
    plt.grid(True)
    
    
#for evaluating metrics

def eval_collision_geometric(obstacles, y_traj, margin=-1e-2): 
    collision_bool = []
    collision_dists = []
    for x in y_traj: 
        collision = False 
        obst_dists = []
        for obs in obstacles:
            p, r = obs['pos'], obs['rad']
            dist = np.linalg.norm(p-x)
            obst_dists.append(dist-r)
            
            if dist < (r+margin): # add a margin to the obstacle surface (at least radius of point mass)
                collision = True 
        collision_bool.append(collision)
        collision_dists.append(np.min(obst_dists))
        
    return np.array(collision_bool), np.array(collision_dists)

def check_collision_single(obstacle, x, x_rad = 0.03):
    obs_rad = obstacle['rad']
    obs_pos = obstacle['pos']
    dist = np.linalg.norm(x[:3]-obs_pos)
    return dist < (obs_rad + x_rad)

def check_collision(obstacles, x, x_rad = 0.03):
    col_status_array = []
    for obstacle in obstacles:
        col_status_array.append(check_collision_single(obstacle, x, x_rad))
    col_status = np.max(col_status_array)
    return col_status, col_status_array



#for ddp
def lin_interpolate(x0, xT, T):
    xs = [x0]
    diff = (xT-x0)/T
    for i in range(T):
        xs += [x0 + diff*(i+1)]
    return xs

def lin_interpolate_double(x0, x1, x2, T1, T2):
    xs1 = lin_interpolate(x0, x1, T1)
    xs2 = lin_interpolate(x1,x2, T2)[1:]
    xs = xs1 + xs2
    return np.array(xs)

def create_double_integrator(Dx = 6, Du = 3, dt = 0.05):
    #Define the matrix A and B to define a double integrator
    dof = Dx//2
    A = np.eye(Dx)
    A[:dof,dof:] = np.eye(dof)*dt
    B = np.zeros((Dx, Du))
    B[:dof,:] = 0.5*np.eye(Du)*(dt**2)
    B[dof:, :] = np.eye(Du)*dt
    lin_sys = LinearSystem(A, B)
    return lin_sys


def compute_v_a(qs, dp0=np.zeros(3), T=100, dt = 0.05):
    dqs = np.zeros((T+1, 3))
    us = np.zeros((T, 3))
    dp = dp0
    dqs[0] = dp
    for i in range(T):
        p = qs[i]
        pn = qs[i+1]
        a = (pn-p - dp*dt)/(0.5*dt**2)
        dp = dp + a*dt
        dqs[i+1] = dp
        us[i] = a
    xs = np.concatenate([qs, dqs],axis=1)
    return xs, us

def setup_croc_model(sys, costs, x0, T):
    '''
    Setup crocoddyl model
    ''' 
    sys.set_init_state(x0)
    rmodels = []
    for i in range(T):
        state = crocoddyl.StateVector(sys.Dx)
        rmodel = ActionModelRobot(state, sys.Du)
        rmodel.init_robot_sys(sys, nr = sys.Dx)
        rmodel.set_cost(costs[i])
        rmodels += [rmodel]

    rmodel_T = ActionModelRobot(state, sys.Du)
    rmodel_T.init_robot_sys(sys, nr = sys.Dx)
    rmodel_T.set_cost(costs[-1]) #terminalCost
    
    problem = crocoddyl.ShootingProblem(x0, rmodels, rmodel_T)
    return problem 

def setup_cost(sys, obstacles, Q, Qf, R, x_target, timesteps, w_bound, bounds):
    costs = []
    for i in range(timesteps):
        runningStateCost = CostModelQuadratic(sys, Q, x_ref = x_target)
        runningControlCost = CostModelQuadratic(sys, None, R)
        runningBoundCost = CostModelBound(sys, bounds, weight=w_bound)
        
        runningObstacleCosts = []
        for obs in obstacles: 
            runningObstacleCosts.append(CostModelCollisionSphere(sys, obs['pos'], obs['rad'], w_obs=obs['w'], d_margin=obs['d_marg']))
        
        runningCost = CostModelSum(sys, [runningStateCost, runningControlCost, runningBoundCost]+runningObstacleCosts)        
        costs += [runningCost]

    terminalStateCost = CostModelQuadratic(sys,Qf, x_ref = x_target)
    terminalControlCost = CostModelQuadratic(sys, None,R)
    terminalBoundCost = CostModelBound(sys, bounds, weight=w_bound)
    terminalObstacleCosts = []
    for obs in obstacles: 
        terminalObstacleCosts.append(CostModelCollisionSphere(sys, obs['pos'], obs['rad'], w_obs=obs['w'], d_margin=obs['d_marg']))
    terminalCost = CostModelSum(sys, [terminalStateCost, terminalControlCost, terminalBoundCost]+terminalObstacleCosts)
    costs += [terminalCost]
    return costs


from costs import *
def setup_ilqr(timesteps, x0, x_target, obstacles): 
    # setup pointmass system with 3 DOF (i.e. double integrator)
    dt = 0.05  #duration of 1 time step
    Dx, Du = 6, 3 #dimensions of x and u
    lin_sys = create_double_integrator(Dx, Du, dt)
    
    # set regularization cost coefficients Q & R 
    Q = np.eye(lin_sys.Dx)*0.01 #coefficient for running cost
    Qf = np.eye(lin_sys.Dx)*100  #coefficient for terminal cost
    R = np.eye(lin_sys.Du)*0.005  #control coefficient
    
    # Set workspace bounds
    large_num = 1e10
    bounds = np.array([[-1,-1,-1, -large_num, -large_num, -large_num],
                       [1,1,1, large_num, large_num, large_num]])
    w_bound = 100.


    # intialize costs 
    costs = setup_cost(lin_sys, obstacles, Q, Qf, R, x_target, timesteps, w_bound, bounds)
    
    # setup problem in crocoddyl 
    prob = setup_croc_model(lin_sys, costs, x0, timesteps)
    
    return prob, lin_sys

def solve_ilqr(croc_prob, xs_init, us_init, iterations=100, th_grad = 1e-9, th_stop=1e-9, x_reg = 1e-8, u_reg = 1e-8): 
    ddp = crocoddyl.SolverFDDP(croc_prob)
    ddp.setCallbacks([crocoddyl.CallbackVerbose()])
    ddp.th_grad = th_grad
    ddp.th_stop = th_stop
    ddp.x_reg = x_reg
    ddp.u_reg = u_reg
    ddp.solve(list(xs_init[:,:,None]), list(us_init[:,:,None]), iterations, False, x_reg)
    
    xs, us = np.array(ddp.xs), np.array(ddp.us)
    return xs, us, ddp 

def create_standard_init(syst, x0, T):
    '''
    compute standard initialisation (stationary)
    '''
    #set initial control output to be all zeros
    us_init = np.zeros((T,syst.Du))
    xs_init = syst.rollout(us_init)
    return xs_init, us_init 

def create_linear_init(x0, x_target, T):
    '''
    compute linear interpolation from x0 to x_target
    '''
    #linear interpolation of the position from x0 to x_target
    qs_init = np.array(lin_interpolate(x0[:3], x_target[:3], T))
    xs_init, us_init = compute_v_a(qs_init)
    return xs_init, us_init

def create_waypoint_init(x0, x_waypoint, x_target, T):
    '''
    compute linear interpolation from x0 to x_waypoint and then to x_target
    '''
    qs_init = np.array(lin_interpolate_double(x0[:3], x_waypoint[:3], x_target[:3], int(T/2), int(T/2)))
    xs_init, us_init = compute_v_a(qs_init)
    return xs_init, us_init


def create_pred_init(q_traj, x0):
    '''
    warm_start from predicition: 
    compute xs and us from the predicted qs (position traj)
    '''
    qs = q_traj.copy()
    qs[0] = x0[:3] #force to start from initial position
    xs_init, us_init = compute_v_a(qs)
    xs_init[:,3:] *= 0 #velocity prediction is often bad, so set to zero
    return xs_init, us_init

def create_lqt_init(lin_sys, q_traj, x0, x_target, T):
    '''
    compute xs and us from the predicted qs (position traj) via LQT
    '''
    if len(x0) == 3:
        x0 = np.concatenate([x0, np.zeros(3)])
    if len(x_target) == 3:
        x_target = np.concatenate([x_target, np.zeros(3)])
    
    lin_sys.set_init_state(x0)
    lqt = finiteLQT(lin_sys)
    
    #set lqt parameters
    Q = np.identity(lin_sys.Dx)*0.05
    Q[3:,3:] *= 0

    Qf = np.identity(lin_sys.Dx)*1000
#     Qf[3:,3:] *= 0

    R = np.identity(3)*0.001

    x_ref = np.concatenate([q_traj, np.zeros((q_traj.shape))], axis=1)
    x_ref[-1] = x_target
    
    #set and solve lqt
    lqt.set_ref(x_ref)
    lqt.set_timestep(T)
    lqt.set_cost(Q, R, Qf)
    xs_init, us_init = lqt.solve()
    return xs_init, us_init


#checking solution
def check_cost(ddp, target_thres, obstacles, T, obstacles_thres = 0.03):
    cost_idx_map = {'state':0, 'control':1, 'bound':2, 'obstacle':3}
    
    dist_obstacles = []
    obs_thresholds = []
    for idx, obs in enumerate(obstacles): 
        obs_thres = obs['rad'] + obstacles_thres # does it not touch the obstacle -> collision status (bool)
        obs_thresholds += [obs_thres for i in range(T+1)]
        dist_obstacles += [ddp.problem.runningModels[i].cost_model.costs[cost_idx_map['obstacle']+idx].dist for i in range(T)]
        dist_obstacles += [ddp.problem.terminalModel.cost_model.costs[cost_idx_map['obstacle']+idx].dist]
    dist_obstacles = np.array(dist_obstacles)
    collision_status_array = dist_obstacles > obs_thresholds
    collision_status = np.min(collision_status_array)    

    cost_target_res = ddp.problem.terminalModel.cost_model.costs[cost_idx_map['state']].res
    target_status = np.linalg.norm(cost_target_res) < target_thres
    
    status = np.logical_and(collision_status, target_status)
    return status, target_status, collision_status, dist_obstacles, obs_thresholds



#for quadcopter
from ocp_sys import QuadcopterCasadi
def setup_ilqr_quadcopter(timesteps, x0, x_target, obstacles): 
    # setup pointmass system with 3 DOF (i.e. double integrator)
    dt = 0.05  #duration of 1 time step
    Dx, Du = 6, 3 #dimensions of x and u
    robot_sys = QuadcopterCasadi(dt = dt)
    
    # set regularization cost coefficients Q & R 
    Q = np.eye(robot_sys.Dx)*0.01 #coefficient for running cost
    Qf = np.eye(robot_sys.Dx)*100  #coefficient for terminal cost
    R = np.eye(robot_sys.Du)*0.005  #control coefficient
    
    # Set workspace bounds
    large_num = 1e10
    bounds = np.array([[-1.,-1.,-1., -np.pi, -np.pi, -np.pi, -large_num, -large_num, -large_num, -large_num, -large_num, -large_num],
                       [1.,1.,1., np.pi, np.pi, np.pi, large_num, large_num, large_num, large_num, large_num, large_num]])
    w_bound = 100.


    # intialize costs 
    costs = setup_cost(robot_sys, obstacles, Q, Qf, R, x_target, timesteps, w_bound, bounds)
    
    # setup problem in crocoddyl 
    prob = setup_croc_model(robot_sys, costs, x0, timesteps)
    
    return prob, robot_sys

def init_pybullet_quadcopter(x0, x_target, obstacles, obj_rad = 0.15, globalScaling=0.5):
    '''
    Return: obj_id, init_id, target_id, border_id, obstacle_ids
    '''
    p.resetSimulation()

    quad_id = p.loadURDF("../data/urdf/quadrotor.urdf",[0,0,0],p.getQuaternionFromEuler([0,0,0]),globalScaling=0.5)
    _,_,obj_id = create_primitives(radius = obj_rad, rgbaColor=[1,1,1,0.5])    
    _,_,init_id = create_primitives(radius = 0.04, rgbaColor=[1,0,0,0.5])
    _,_,target_id = create_primitives(radius = 0.04, rgbaColor=[0,1,0,0.5])
    _,_,border_id = create_primitives(rgbaColor=[1,0,0,0.1], shapeType=p.GEOM_BOX, halfExtents=[1.,1.,1.])
    obstacle_ids = create_obstacles(obstacles)
    
    p.resetBasePositionAndOrientation(init_id, x0[:3], (0,0,0,1))
    p.resetBasePositionAndOrientation(target_id, x_target[:3], (0,0,0,1))
    return quad_id, obj_id, init_id, target_id, border_id, obstacle_ids


def setup_cost_general(sys, obj_id, obstacles, Q, Qf, R, x_target, timesteps, w_bound, bounds):
    '''
    Setting ddp costs for general collision objects
    '''
    costs = []
    for i in range(timesteps):
        runningStateCost = CostModelQuadratic(sys, Q, x_ref = x_target)
        runningControlCost = CostModelQuadratic(sys, None, R)
        runningBoundCost = CostModelBound(sys, bounds, weight=w_bound)
        
        runningObstacleCosts = []
        for obs in obstacles: 
            cost_col = CostModelCollisionGeneral(sys, obj_id, obs['obs_id'], w_obs=obs['w'])
            runningObstacleCosts.append(cost_col)
        runningCost = CostModelSum(sys, [runningStateCost, runningControlCost, runningBoundCost]+runningObstacleCosts)        
        costs += [runningCost]

    terminalStateCost = CostModelQuadratic(sys,Qf, x_ref = x_target)
    terminalControlCost = CostModelQuadratic(sys, None,R)
    terminalBoundCost = CostModelBound(sys, bounds, weight=w_bound)
    terminalObstacleCosts = []
    for obs in obstacles: 
        cost_col = CostModelCollisionGeneral(sys, obj_id, obs['obs_id'], w_obs=obs['w'])
        terminalObstacleCosts.append(cost_col)
    terminalCost = CostModelSum(sys, [terminalStateCost, terminalControlCost, terminalBoundCost]+terminalObstacleCosts)
    costs += [terminalCost]
    return costs

def setup_ilqr_quadcopter_general(timesteps, x0, x_target, obstacles, obj_id): 
    '''
    Setting ddp for quadcopter &  general collision objects
    '''
    # setup pointmass system with 3 DOF (i.e. double integrator)
    dt = 0.05  #duration of 1 time step
    Dx, Du = 6, 3 #dimensions of x and u
    robot_sys = QuadcopterCasadi(dt = dt)
    
    # set regularization cost coefficients Q & R 
    Q = np.eye(robot_sys.Dx)*0.01 #coefficient for running cost
    Qf = np.eye(robot_sys.Dx)*100  #coefficient for terminal cost
    R = np.eye(robot_sys.Du)*0.005  #control coefficient
    
    # Set workspace bounds
    large_num = 1e10
    bounds = np.array([[-1.,-1.,-1., -np.pi, -np.pi, -np.pi, -large_num, -large_num, -large_num, -large_num, -large_num, -large_num],
                       [1.,1.,1., np.pi, np.pi, np.pi, large_num, large_num, large_num, large_num, large_num, large_num]])
    w_bound = 100.


    # intialize costs 
    costs = setup_cost_general(robot_sys, obj_id, obstacles, Q, Qf, R, x_target, timesteps, w_bound, bounds)
    
    # setup problem in crocoddyl 
    prob = setup_croc_model(robot_sys, costs, x0, timesteps)
    
    return prob, robot_sys

def get_best_mdn_prediction(y_traj, x, obstacles, T = 100, general_obs = False, obj_id = 0, add_zeros_dim = 3):
    costs = []
    x0, x_target = x[-6:-3], x[-3:]
    x0 = np.concatenate([x0, np.zeros(add_zeros_dim)])
    x_target = np.concatenate([x_target, np.zeros(add_zeros_dim)])
    if general_obs:
        prob, lin_sys = setup_ilqr_quadcopter_general(T, x0, x_target, obstacles, obj_id)
    else:
        prob, lin_sys = setup_ilqr(T, x0, x_target, obstacles)
    ddp = crocoddyl.SolverFDDP(prob)
    for k in range(len(y_traj)):
        xs_init, us_init = create_lqt_init(lin_sys, y_traj[k], x0, x_target, T)
        costs.append(ddp.problem.calc(list(xs_init[:,:,None]), list(us_init[:,:,None])))    
    best_idx = np.argmin(costs)
    y_traj = y_traj[best_idx]
    return y_traj

def check_collision_single_general(obstacle, obj_id, x, margin = 0.0):
    col_info = p.getClosestPoints(obj_id, obstacle['obs_id'], 100)[0]
    dist = col_info[8]
    return dist < margin

def check_collision_general(obstacles, obj_id, x, margin = 0.0):
    p.resetBasePositionAndOrientation(obj_id, x[:3],(0,0,0,1))
    col_status_array = []
    for obstacle in obstacles:
        col_status_array.append(check_collision_single_general(obstacle, obj_id, x,  margin))
    col_status = np.max(col_status_array)
    return col_status, col_status_array

def check_cost_general(xs, x_target,  obstacles, obj_id, target_thres = 0.03, obstacles_margin = -1e-2):
    target_dist = np.linalg.norm(xs[-1,:3] - x_target[:3]) #only position target
    target_status = target_dist < target_thres
    
    collision_status = np.max(eval_collision_geometric_general(obstacles, obj_id, xs, obstacles_margin))
    status = np.logical_and(np.logical_not(collision_status), target_status)
    return status, target_status, collision_status

def eval_collision_geometric_general(obstacles, obj_id, y_traj, margin=-1e-2): 
    collision_bool = []
    for x in y_traj: 
        col_status, _ = check_collision_general(obstacles, obj_id, x, margin)
        collision_bool.append(col_status)

        
    return np.array(collision_bool)





def get_best_mdn_prediction_mse(y_traj, y_true):
    K = len(y_traj)
    dist = np.linalg.norm((y_traj.reshape(K,-1)-y_true), axis=-1)
    best_idx = np.argmin(dist)
    y_traj = y_traj[best_idx]
    return y_traj

def get_best_mdn_prediction_ddp(y_traj, x, obstacles, T = 100, general_obs = True, obj_id = 0, add_zeros_dim = 3, verbose = False):
    costs = []
    x0, x_target = x[-12:-6], x[-6:]
    x0 = np.concatenate([x0, np.zeros(add_zeros_dim)])
    x_target = np.concatenate([x_target, np.zeros(add_zeros_dim)])
    if general_obs:
        prob, lin_sys = setup_ilqr_quadcopter_general(T, x0, x_target, obstacles, obj_id)
    else:
        prob, lin_sys = setup_ilqr_quadcopter_general(T, x0, x_target, obstacles)
    ddp = crocoddyl.SolverFDDP(prob)
    Dx, Du = 12, 6 #dimensions of x and u
    lin_sys = create_double_integrator(Dx, Du, 0.05)
    for k in range(len(y_traj)):
        xs_init, us_init = create_lqt_init_quad(lin_sys, y_traj[k], x0[:6], x_target[:6], T)
#         xs_init = np.hstack([xs_init[:,:3], np.zeros((T+1,3)), xs_init[:,6:], np.zeros((T+1,3))])
        us_init = np.array(ddp.problem.quasiStatic(list(xs_init[:-1])))
        costs.append(ddp.problem.calc(list(xs_init[:,:,None]), list(us_init[:,:,None])))    
    best_idx = np.argmin(costs)
    print(costs[best_idx])
    y_traj = y_traj[best_idx]
    return y_traj


def create_lqt_init_quad(lin_sys, q_traj, x0, x_target, T):
    '''
    compute xs and us from the predicted qs (position traj) via LQT
    '''

    
#     if len(x0) == 6:
#         x0 = np.concatenate([x0, np.zeros(6)])
#     if len(x_target) == 6:
#         x_target = np.concatenate([x_target, np.zeros(6)])
    
    lin_sys.set_init_state(x0)
    lqt = finiteLQT(lin_sys)
    
    #set lqt parameters
    Q = np.identity(lin_sys.Dx)*0.05
#     Q[6:,6:] *= 0

    Qf = np.identity(lin_sys.Dx)*1000
#     Qf[3:,3:] *= 0

    R = np.identity(6)*0.001

#     x_ref = np.concatenate([q_traj, np.zeros((q_traj.shape))], axis=1)
    x_ref = q_traj.copy()
    
    x_ref[-1] = x_target
    
    #set and solve lqt
    lqt.set_ref(x_ref)
    lqt.set_timestep(T)
    lqt.set_cost(Q, R, Qf)
    xs_init, us_init = lqt.solve()
    return xs_init, us_init



def get_best_mdn_prediction_ddp2(y_traj, x, obstacles, T = 100, general_obs = True, obj_id = 0, add_zeros_dim = 3, verbose = False):
    costs = []
    x0, x_target = x[-24:-12], x[-12:]
#     x0 = np.concatenate([x0, np.zeros(add_zeros_dim)])
#     x_target = np.concatenate([x_target, np.zeros(add_zeros_dim)])
    if general_obs:
        prob, lin_sys = setup_ilqr_quadcopter_general(T, x0, x_target, obstacles, obj_id)
    else:
        prob, lin_sys = setup_ilqr_quadcopter_general(T, x0, x_target, obstacles)
    ddp = crocoddyl.SolverFDDP(prob)
    Dx, Du = 12, 6 #dimensions of x and u
    lin_sys = create_double_integrator(Dx, Du, 0.05)
    for k in range(len(y_traj)):
        xs_lqt = np.hstack([y_traj[k][:,:3], y_traj[k][:,6:9], y_traj[k][:,3:6], y_traj[k][:,9:]])
        xs_init, us_init = create_lqt_init_quad(lin_sys, xs_lqt, x0, x_target, T)
        xs_init = np.hstack([xs_init[:,:3], xs_init[:,6:9], xs_init[:,3:6], xs_init[:,9:]])
#         xs_init = np.hstack([xs_init[:,:6], np.zeros((T+1,3)), xs_init[:,3:], np.zeros((T+1,3))])
        us_init = np.array(ddp.problem.quasiStatic(list(xs_init[:-1])))
        costs.append(ddp.problem.calc(list(xs_init[:,:,None]), list(us_init[:,:,None])))    
    best_idx = np.argmin(costs)
    print(costs[best_idx])
    y_traj = y_traj[best_idx]
    return y_traj, xs_init, us_init