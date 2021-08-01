import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pybullet as p
import time
import crocoddyl
from casadi import mtimes, MX, sin, cos, vertcat, horzcat, sum1, cross, Function, jacobian
import casadi

import transforms3d

def rectify_quat(quat):
    #transform from transforms3d format (w,xyz) to pybullet and pinocchio (xyz, w)
    quat_new = np.concatenate([quat[1:], quat[0:1]])
    return quat_new

def euler2quat(rpy, axes='sxyz'):
    #euler sxyz: used by Manu's codes
    return rectify_quat(transforms3d.euler.euler2quat(*rpy, axes=axes))


class LinearSystem():
    def __init__(self,A,B):
        self.A = A
        self.B = B
        self.Dx = A.shape[0]
        self.Du = B.shape[1]
        
    def reset_AB(self, A,B):
        self.A = A
        self.B = B
        
    def set_init_state(self,x0):
        self.x0 = x0
    
    def compute_matrices(self,x,u):
        #compute the derivatives of the dynamics
        return self.A,self.B
    
    def compute_ee(self,x, ee_id=1):
        #The end-effector for a point mass system is simply its position
        #The ee_id is added as dummy variable, just for uniformity of notation with other systems
        return x[:int(self.Dx/2)], None 
    
    def compute_Jacobian(self,x, ee_id=1):
        #The end-effector Jacobian for a point mass system is simply an identity matrix
        #The ee_id is added as dummy variable, just for uniformity of notation with other systems
        return np.eye(int(self.Dx/2)) 
    
    
    def step(self, x, u):
        return self.A.dot(x) + self.B.dot(u)
    
    def rollout(self,us):
        x_cur = self.x0
        xs = [x_cur]
        T = len(us)
        for i in range(T):
            x_cur = self.step(x_cur, us[i])
            xs += [x_cur]
        return np.array(xs)

        
class TwoLinkRobot():
    def __init__(self, dt = 0.01):
        self.dt = dt
        self.Dx = 4
        self.Du = 2
        self.dof = 2
        self.l1 = 1.5
        self.l2 = 1
        self.p_ref = np.zeros(2)
        
    def set_init_state(self,x0):
        self.x0 = x0
        
    def set_pref(self, p_ref):
        self.p_ref = p_ref
    
    def compute_matrices(self,x,u):
        #compute the derivatives of the dynamics
        A = np.eye(self.Dx)
        B = np.zeros((self.Dx,self.Du))
        
        A[0,2] = self.dt
        A[1,3] = self.dt
        
        B[2:,:] = np.eye(self.Du)
        
        self.A, self.B = A,B
        return A,B
    
    def compute_ee(self,x, ee_id=0):
        self.p1 = np.array([self.l1*np.cos(x[0]), self.l1*np.sin(x[0])])
        self.p2 = np.array([self.p1[0] + self.l2*np.cos(x[0] + x[1]), self.p1[1] + self.l2*np.sin(x[0] + x[1])])
        return self.p2, self.p1

    
    def compute_Jacobian(self, x, ee_id=0):
        J = np.zeros((2, 2))
        s1 = np.sin(x[0])
        c1 = np.cos(x[0])
        s12 = np.sin(x[0] + x[1])
        c12 = np.cos(x[0] + x[1])
        
        J[0,0] = -self.l1*s1 - self.l2*s12
        J[0,1] = - self.l2*s12
        J[1,0] =  self.l1*c1 + self.l2*c12
        J[1,1] =  self.l2*c12
        
        self.J = J
        return self.J
        
    def step(self, x, u):
        x_next = self.A.dot(x) + self.B.dot(u)
        return x_next
    
    def rollout(self,us):
        x_cur = self.x0.copy()
        xs = [x_cur]
        T = len(us)
        for i in range(T):
            x_cur = self.step(x_cur, us[i])
            xs += [x_cur]
        return np.array(xs)
    
    
    def plot(self, x, color='k'):
        self.compute_ee(x)
        
        line1 = plt.plot(np.array([0, self.p1[0]]),np.array([0, self.p1[1]]) , marker='o', color=color, lw=10, mfc='w', solid_capstyle='round')
        line2 = plt.plot(np.array([self.p1[0], self.p2[0]]),np.array([self.p1[1], self.p2[1]]) , marker='o', color=color, lw=10, mfc='w', solid_capstyle='round')
        xlim = [-1.5*(self.l1+self.l2), 1.5*(self.l1+self.l2)]
        #plt.axes().set_aspect('equal')
        plt.axis(xlim+xlim)
        return line1,line2

    def plot_traj(self, xs, dt = 0.1):
        for x in xs:
            clear_output(wait=True)
            self.plot(x)
            plt.plot(self.p_ref[0], self.p_ref[1], '*')
            plt.show()
            time.sleep(self.dt)
            
class URDFRobot():
    def __init__(self, dof, robot_id, joint_indices = None, dt = 0.01):
        self.dt = dt
        self.Dx = dof*2
        self.Du = dof
        self.dof = dof
        self.robot_id = robot_id
        if joint_indices is None:
            self.joint_indices = np.arange(dof)
        else:
            self.joint_indices = joint_indices
        
    def set_init_state(self,x0):
        self.x0 = x0
        self.set_q(x0)

    def compute_matrices(self,x,u):
        #compute the derivatives of the dynamics
        A = np.eye(self.Dx)
        B = np.zeros((self.Dx,self.Du))
        
        A[:self.dof, self.dof:] = np.eye(self.dof)*self.dt
        
        #B[self.dof:,:] = np.eye(self.Du)
        B[:self.dof,:] = np.eye(self.Du) * self.dt * self.dt /2
        B[-self.dof:,:] = np.eye(self.Du) * self.dt    

        self.A, self.B = A,B
        return A,B
    
    def compute_ee(self,x, ee_id):
        self.set_q(x)
        ee_data = p.getLinkState(self.robot_id, ee_id)
        pos = np.array(ee_data[0])
        quat = np.array(ee_data[1])
        return pos, quat
    
    def compute_Jacobian(self, x, ee_id):
        zeros = [0.]*self.dof
        Jl, Ja = p.calculateJacobian(self.robot_id, ee_id, [0.,0.,0.], x[:self.dof].tolist(),zeros,zeros)
        Jl, Ja = np.array(Jl), np.array(Ja)
        self.J = np.concatenate([Jl, Ja], axis=0)
        return self.J
        
    def step(self, x, u):
        x_next = self.A.dot(x) + self.B.dot(u)
        return x_next
    
    def rollout(self,us):
        x_cur = self.x0.copy()
        xs = [x_cur]
        T = len(us)
        for i in range(T):
            x_cur = self.step(x_cur, us[i])
            xs += [x_cur]
        return np.array(xs)
    
    def set_q(self, x):
        q = x[:self.dof]
        for i in range(self.dof):
            p.resetJointState(self.robot_id, self.joint_indices[i], q[i])
        return 

    def vis_traj(self, xs, dt = 0.1):
        for x in xs:
            clear_output(wait=True)
            self.set_q(x)
            time.sleep(self.dt)
            
class ActionModelRobot(crocoddyl.ActionModelAbstract):
    def __init__(self, state, nu):
        crocoddyl.ActionModelAbstract.__init__(self, state, nu)
        
    def init_robot_sys(self,robot_sys, nr = 1):
        self.robot_sys = robot_sys
        self.Du = robot_sys.Du
        self.Dx = robot_sys.Dx
        self.Dr = nr
        
    def set_cost(self, cost_model):
        self.cost_model = cost_model
        
    def calc(self, data, x, u):
        #calculate the cost
        data.cost = self.cost_model.calc(x,u)
        
        #calculate the next state
        data.xnext = self.robot_sys.step(x,u)
        
    def calcDiff(self, data, x, u, recalc = False):
        if recalc:
            self.calc(data, x, u)

        #compute cost derivatives
        self.cost_model.calcDiff(x, u)
        data.Lx = self.cost_model.Lx.copy()
        data.Lxx = self.cost_model.Lxx.copy()
        data.Lu = self.cost_model.Lu.copy()
        data.Luu = self.cost_model.Luu.copy()
        
        #compute dynamic derivatives 
        A, B = self.robot_sys.compute_matrices(x,u)
        data.Fx = A.copy()
        data.Fu = B.copy()
        
    def createData(self):
        data = ActionDataRobot(self)
        return data

class ActionDataRobot(crocoddyl.ActionDataAbstract):
    def __init__(self, model):
        crocoddyl.ActionDataAbstract.__init__(self,model)
        
class QuadcopterCasadi():
    def __init__(self, dt = 0.01, I = np.diag(np.array([2,2,4])), kd = 1, 
                 k = 1, L = 0.3, b = 1, m=1, g=9.81, Dx=12, Du=4):
        self.I = I #inertia
        self.I_inv = np.linalg.inv(self.I)
        self.kd = kd #friction
        self.k = k #motor constant
        self.L = L# distance between center and motor
        self.b = b # drag coefficient
        self.m = m # mass
        self.g = g
        self.Dx = Dx
        self.Du = Du
        self.dt = dt
        self.gravity = np.array([0,0,-self.g])
        self.x = MX.sym('x', Dx)
        self.u = MX.sym('u', Du)
        
        #initialise
        self.def_step_func(self.x, self.u)
        self.def_jacobian()
        
    def compute_ee(self,x, ee_id=1):
        #The end-effector for a point mass system is simply its position
        #The ee_id is added as dummy variable, just for uniformity of notation with other systems
        return x[:3], None 
    
    def compute_Jacobian(self,x, ee_id=1):
        #The end-effector Jacobian for a point mass system is simply an identity matrix
        #The ee_id is added as dummy variable, just for uniformity of notation with other systems
        return np.eye(3) 
        
    def thrust(self, inputs):
        T = vertcat(0,0, self.k*sum1(inputs))
        return T

    def torques(self, inputs):
        tau = vertcat(self.L*self.k*(inputs[0]-inputs[2]), self.L*self.k*(inputs[1]-inputs[3]), \
                        self.b*(inputs[0]-inputs[1] + inputs[2] - inputs[3]))
        return tau

    def acceleration(self, inputs, thetas, xdot):
        R = self.Rotation(thetas)
        T = mtimes(R,self.thrust(inputs))
        Fd = -self.kd*xdot
        a = self.gravity + T/self.m + Fd
        return a

    def angular_acceleration(self, inputs, omega):
        tau = self.torques(inputs)
        omegadot = mtimes(self.I_inv, tau - cross(omega, mtimes(self.I,omega)))
        return omegadot

    def thetadot2omega(self, thetadot, theta):
        R1 = vertcat(1,0,0)
        R2 = vertcat(0, cos(theta[0]), -sin(theta[0]))
        R3 = vertcat(-sin(theta[1]), cos(theta[1])*sin(theta[0]), cos(theta[1])*cos(theta[0]))
        R = horzcat(R1, R2, R3)
        return mtimes(R,thetadot)

    def omega2thetadot(self, omega, theta):
        R1 = vertcat(1,0,0)
        R2 = vertcat(0, cos(theta[0]), -sin(theta[0]))
        R3 = vertcat(-sin(theta[1]), cos(theta[1])*sin(theta[0]), cos(theta[1])*cos(theta[0]))
        R = horzcat(R1, R2, R3)
        return mtimes(casadi.inv(R), omega)

    def Rotation(self, theta):
        c0,s0 = cos(theta[0]), sin(theta[0])
        c1,s1 = cos(theta[1]), sin(theta[1])
        c2,s2 = cos(theta[2]), sin(theta[2])
        
        R1 = vertcat(c0*c2 - c1*s0*s2, c1*c2*s0 + c0*s2, s0*s1)
        R2 = vertcat(-c2*s0 - c0*c1*s2, c0*c1*c2-s0*s2, c0*s1 )
        R3 = vertcat( s1 * s2, -c2*s1, c1)
        R = horzcat(R1,R2,R3)
        return R
        
    def set_init_state(self,x0):
        self.x0 = x0
        
    def def_step_func(self, x, u, u_offset = None):
        if u_offset is None:
            u_mag = np.sqrt(9.81/4)
            u_offset = np.array([u_mag]*self.Du)**2 
        u_act = u_offset + u
        p, pdot, theta, thetadot = x[:3], x[3:6], x[6:9], x[9:]

        #step
        omega = self.thetadot2omega(thetadot, theta)
    
        a = self.acceleration(u_act, theta, pdot)
        omegadot = self.angular_acceleration(u_act, omega)
        omega = omega + self.dt*omegadot
        thetadot= self.omega2thetadot(omega, theta)
        theta = theta + self.dt*thetadot
        pdot = pdot + self.dt*a
        p = p + self.dt*pdot
        
        self.x_next = vertcat(p, pdot, theta, thetadot)
        self.step_fun = Function('step', [x, u], [self.x_next])

    def step(self, x, u):
        return np.array(self.step_fun(x,u)).flatten()
        
    def def_jacobian(self):
        self.A = jacobian(self.x_next, self.x)
        self.B = jacobian(self.x_next, self.u)
        
        self.A_val = Function('A', [self.x,self.u], [self.A])
        self.B_val = Function('B', [self.x,self.u], [self.B])

    def compute_matrices(self,x,u):
        A = np.array(self.A_val(x,u))
        B = np.array(self.B_val(x,u))
        return A, B
        
    def rollout(self,us):
        x_cur = self.x0.copy()
        xs = [x_cur]
        T = len(us)
        for i in range(T):
            x_cur = self.step(x_cur, us[i])
            xs += [x_cur]
        return np.array(xs)
    
    def vis_traj(self, quadId, xs, dt = 0.05, camDist = 2.5, camYaw = 90, camPitch = -20, changeCamera = True):
        for i,x in enumerate(xs):
            ori = euler2quat(xs[i,6:9], 'rzyz')
            p.resetBasePositionAndOrientation(quadId, xs[i,:3], ori)
            time.sleep(dt)
            if changeCamera: p.resetDebugVisualizerCamera(cameraDistance=camDist, cameraYaw=camYaw, 
                                         cameraPitch= camPitch, cameraTargetPosition=xs[i,:3])
                
                