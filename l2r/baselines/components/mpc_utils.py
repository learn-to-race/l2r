# ========================================================================= #
# Filename:                                                                 #
#    mpc_utils.py                                                           #
#                                                                           #
# Description:                                                              # 
#    an agent used a MPC controller                                         #
# ========================================================================= #

import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter

from mpc import mpc
from mpc.mpc import QuadCost, LinDx, GradMethods

import ipdb as pdb

class BikeModel(nn.Module):
    def __init__(self, dt = 1/20, 
                       n_state = 4, # state = [x, y, v, phi]
                       n_ctrl = 2, # action = [a, delta]
                       init_params = None):
        super().__init__()
        self.dt = dt
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.u_lower = -1
        self.u_upper = 1

        ## Learnable parameters related to the properties of the car
        if init_params is None:
            self.params = torch.ones(3, requires_grad = True)
        else:
            self.params = torch.tensor(init_params, requires_grad = True)
            
    def forward(self, x_init, u):
        # k1 and k2 maps the action in [-1, 1] to actual acceleration and steering angle
        l, k1, k2 = torch.unbind(self.params)
        n = x_init.shape[0]
        x, y, v, phi = torch.unbind(x_init, dim = -1)
        a = u[:, 0]*k1
        delta = u[:, 1]*k2
        
        x_dot = torch.zeros(n, self.n_state)
        x_dot[:, 0] = v*torch.cos(phi)
        x_dot[:, 1] = v*torch.sin(phi)
        x_dot[:, 2] = a
        x_dot[:, 3] = v*torch.tan(delta)/l
        
        return x_init + self.dt * x_dot

    def grad_input(self, x, u):
        '''
        Dynamics follows https://pythonrobotics.readthedocs.io/en/latest/modules/path_tracking.html
            Reference: Back Axle
            Frame: Global
        Input: 
            x, u: (T-1, dim)
        Output: 
            Ft: (T-1, m, m+n)
            ft: (T-1, m)
        '''
        
        # k1 and k2 maps the action in [-1, 1] to actual acceleration and steering angle
        #l, k1, k2 = torch.unbind(self.params)
        l, k1, k2 = torch.unbind(self.params)
    
        T, _ = x.shape
        x, y, v, phi = torch.unbind(x, dim = -1) # T
    
        a = u[:, 0]*k1
        delta = u[:, 1]*k2
        
        A = torch.eye(self.n_state).repeat(T, 1, 1) # T x m x m
        A[:, 0, 2] = self.dt * torch.cos(phi)
        A[:, 0, 3] = - self.dt * v * torch.sin(phi)
        A[:, 1, 2] = self.dt * torch.sin(phi)
        A[:, 1, 3] = self.dt * v * torch.cos(phi)
        A[:, 3, 2] = self.dt * torch.tan(delta) / l
        
        
        B = torch.zeros(T, self.n_state, self.n_ctrl)
        B[:, 2, 0] = self.dt
        B[:, 3, 1] = self.dt * v / (l * torch.cos(delta) ** 2)

        #F = torch.cat([A, B], dim = -1) # T-1 x n_batch x m x (m+n)
        #pdb.set_trace()
        return A.squeeze(1), B.squeeze(1)

class MPC(nn.Module):
    def __init__(self, T = 6,
                       n_state = 4, # state = [x, y, v, phi]
                       n_ctrl = 2, # action = [a, delta]
                       cost_weight = None,
                       cost_type = 'target',
                       ):
        '''
        Usage: 
            Use as if it is a regular NN
        '''
        self.T = T
        self.n_state = n_state
        self.n_ctrl = n_ctrl

        if cost_weight is None:
            self.cost_weight = torch.ones(self.n_state+self.n_ctrl)
        else: 
            cost_weight = cost_weight.reshape(-1)
            assert len(cost_weight) == self.n_state+self.n_ctrl
            self.cost_weight = cost_weight
            
        assert cost_type.lower() in ['target', 'reference']
        self.cost_type = cost_type.lower()
        
        self.C = torch.diag(self.cost_weight).repeat(self.T, 1, 1) ## T x (m+n) x (m+n)
        if self.cost_type == 'target':
            self.C[:-1, :self.n_state, :self.n_state] = 0
       
    def forward(self, dx, x_init, x_target):
        '''
        Input: 
            dynamics: A model of the environment
            x_init: current state (n, n_state)
            x_target:
                target states at the end of the planning horizon (n, n_state) OR
                reference trajectories over the planning horizon (T, n, n_state)
        Output: 
            x_lqr: T x n x n_state
            u_lqr: T x n x n_ctrl
        '''
        #pdb.set_trace()
        n_batch = x_init.shape[0]
        C, c = self._cost_fn(x_target.float())

        # The MPC iteratively linearize the dynamics at each iteration
        x_lqr, u_lqr, objs_lqr = mpc.MPC(n_state=self.n_state,
                                        n_ctrl=self.n_ctrl,
                                        T=self.T,
                                        u_lower= dx.u_lower*torch.ones((self.T, n_batch, self.n_ctrl)),
                                        u_upper= dx.u_upper*torch.ones((self.T, n_batch, self.n_ctrl)),
                                        lqr_iter=20,
                                        backprop=True,
                                        verbose=0,
                                        n_batch=n_batch,
                                        grad_method = GradMethods.FINITE_DIFF, 
                                        exit_unconverged=False,
                                        )(x_init.float(),
                                        QuadCost(C, c),
                                        dx)  
        return x_lqr, u_lqr

    def _cost_fn(self, x_target):
        '''
        Input:
            x_target:  n x n_state or T x n x n_state
        Output:
            C: T x n x (n_state+n_ctrl) x  (n_state+n_ctrl)
            c: T x n x (n_state+n_ctrl)
        '''
        if self.cost_type == 'target':
            assert x_target.ndimension() == 2
            n = x_target.shape[0]
        elif self.cost_type == 'reference':
            assert x_target.ndimension() == 3
            n = x_target.shape[1]
        else:
            print("Not implemented")
        C = self.C.unsqueeze(1).repeat(1, n, 1, 1)
        
        c = torch.zeros(self.T, n, self.n_state + self.n_ctrl)
        if self.cost_type == 'target':
            c[-1, :, :self.n_state] = -x_target
        elif self.cost_type == 'reference':
            c[:, :, :self.n_state] = -x_target
        else:
            print("Not implemented")
        c *= self.cost_weight.reshape(1, 1, -1)
        return C, c