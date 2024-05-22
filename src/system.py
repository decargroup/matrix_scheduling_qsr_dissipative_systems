# Purpose: Setting up the three link robot in Figure 4
# %------------------------------------------ Packages -------------------------------------------% #
import pathlib
import numpy as np

from typing import Tuple, List

from src import paper_plot
# %------------------------------------------ Abstract Classes -------------------------------------% #
class ManipulatorRobot(object):
    def __init__(self,
                 x_sys0:np.ndarray, 
                 trajectory, r_lin:np.ndarray = None,
                 sys_type:str = "Nonlinear",
                 dissipitivity:str = "nonsquare",
                 disturbance_input=None, 
                 disturbance_output=None) -> None:
        # Initial condition
        self.x_sys0 = x_sys0

        # Desired trajectory
        self.trajectory = trajectory
        
        # Store Dissipitivity type
        self.dissipitivity = dissipitivity

        # Linearization point
        self.r_lin = r_lin
        if r_lin is None:
            self.r_lin = trajectory.r_end
            
        # Choose system type and add disturbance to input
        self.sys_type = sys_type
        match sys_type:  
            case "Nonlinear":
                if disturbance_input is not None:
                    np.random.seed(3)
                    print("Adding Disturbance to Input!")
                    self.f = lambda xs, u: self._f_nonlinear(xs, u + disturbance_input())
                else:
                    self.f = lambda xs, u: self._f_nonlinear(xs, u)
            case "Linearized":
                if disturbance_input is not None:
                    np.random.seed(3)
                    print("Adding Disturbance to Input!")
                    self._f = lambda xs, u: self._f_linearized(xs, u + disturbance_input())
                else:
                    self._f = lambda xs, u: self._f_linearized(xs, u)
            case _:
                raise ValueError
            
        # Const matrices
        O = np.zeros((self.DOF, self.DOF))
        I = np.eye(self.DOF)
        
        # Setup measurement and control based on dissipitivity type
        match dissipitivity:
            case "nonsquare":
                # Control u = [tau_2 ... tau_DOF]
                self.u_dim = self.DOF - 1
                self.bhat = np.block([[np.zeros((1, self.DOF-1))], 
                                      [np.eye(self.DOF-1)]])
                
            case _:
                raise NotImplementedError
        
        # Compute Linearized Model about r_lin
        self.A = np.block([[O, I],
                           [O, -self.M_inv(r_lin) @ self.Kd]])
        
        self.B = np.block([[O],
                           [self.M_inv(r_lin)]]) @ self.bhat
        
        self.C_prewrap = np.block([[I, O]])
        
        self.C = np.block([[O, I]])
        
        self.D = O
        
        # QSR Q, S, R matrices
        self.Q = -self.Kd
        self.S = 1/2*self.bhat
        self.R = np.zeros((self.u_dim, self.u_dim))

        # Define prewrap measurement  
        self.g_prewrap = lambda xs: self.C_prewrap @ xs

        # Add disturbance to output
        if disturbance_output is not None:
            np.random.seed(3)
            print("Adding Disturbance to Output!")
            self.g = lambda xs: self.C @ xs + disturbance_output()
        else:
            self.g = lambda xs: self.C @ xs 

    # Purpose: Compute inv(M) matrix
    def M_inv(self, x) -> np.ndarray:
        return np.linalg.inv(self._M(x))
    
    # Purpose: Non-Linear model
    def _f_nonlinear(self, xs, u) -> np.ndarray:
        # Extract states.
        q_dot = xs[self.DOF:]

        RHS = u - self._fnon(xs) - self.Kd @ q_dot
        q_ddot = np.linalg.solve(self._M(xs), RHS)
        return np.vstack((q_dot, q_ddot))  # xs_dot
    
    # Purpose: Linearized model
    def _f_linearized(self, xs, u) -> np.ndarray:
        return self.A @ xs + self.B @ u
    
    # Purpose: Compute control effort for plotting
    def compute_control_effort(self, N, x_ctrl, q, q_dot, ctrl, t) -> np.ndarray:
        # Compute error and control (for plotting purposes)
        u_ctrl   = np.zeros((self.DOF, N))
        error_th = np.zeros((self.DOF, N))
        error_th_dot = np.zeros((self.DOF, N))
        
        nStates = 2*self.DOF
        x_d    = x_ctrl[nStates:,:] # [x_d]
        x_ctrl = x_ctrl[:nStates,:] # [x_c]
        
        for i, ti in enumerate(t):
            error_th[:, [i]]     = q[:, [i]]     - self.trajectory.r_des(ti)
            error_th_dot[:, [i]] = q_dot[:, [i]] - self.trajectory.r_des_dot(ti)
            u_ctrl[:, [i]] =  ctrl.g_prewrap(error_th[:, [i]]) \
                              + self.bhat @ ctrl.g(x_ctrl[:, [i]], error_th_dot[:, [i]])
        return error_th, error_th_dot, u_ctrl
    
    # Purpose: Extract States and Compute Plot data
    def extract_states(self, sol, ctrl) -> Tuple[np.ndarray,
                                                 List[np.ndarray],
                                                 List[np.ndarray],
                                                 np.ndarray,
                                                 np.ndarray,
                                                 np.ndarray]:
        # Extract states
        t     = sol.t
        sol_x = sol.y
        
        nStates = 2*self.DOF
        x_sys  = sol_x[:nStates, :]
        x_ctrl = sol_x[nStates:, :]
        
        q = x_sys[:self.DOF, :]
        q_dot = x_sys[self.DOF:, :]
        
        # Convert angles to degrees
        th     = [q[i,:]* 180 / np.pi for i in range(self.DOF)]
        dot_th = [q_dot[i,:]* 180 / np.pi for i in range(self.DOF)]
        
        # Number of points
        N = len(t)
          
        # Compute Error for plotting
        error_th, error_th_dot, u_ctrl = self.compute_control_effort(N, x_ctrl, q, q_dot, ctrl, t)
        
        return t, th, dot_th, u_ctrl, error_th, error_th_dot
    
    # Purpose: Plot x, u, and energy
    def plot_results(self, sol, ctrl, save_fig=False) -> None:
        # Extract States and Compute Plot data
        t, th, dot_th, u_ctrl, error_th, error_th_dot = self.extract_states(sol, ctrl)
        
        # Plotting Path
        path = None
        if save_fig:
            path = pathlib.Path('figs')
            path.mkdir(exist_ok=True)
            
        # Title
        title = f"{self.sys_type} model using {type(ctrl).__name__}"
        
        # Plot theta and theta_dot
        labels = lambda indx : [rf'$\theta_{indx+1}(t)$ (deg)',
                                rf'$\dot{{\theta}}_{indx+1}(t)$ (deg/s)']

        for i in range(self.DOF):
            paper_plot.plt_subplot_vs_t(t, th[i], dot_th[i], 
                                        labels=labels(i), title=title,
                                        trajectory=self.trajectory, 
                                        entry=i,
                                        path=path)
            
        # Plot error and error_dot
        labels = lambda indx : [rf'error in $\theta_{indx+1}(t)$ (deg)',
                                rf'error in $\dot{{\theta}}_{indx+1}(t)$ (deg/s)']

        err     = lambda i : [e * 180 / np.pi for e in error_th[i]]
        err_dot = lambda i : [e * 180 / np.pi for e in error_th_dot[i]]
        
        for i in range(self.DOF):
            paper_plot.plt_subplot_vs_t(t, err(i), err_dot(i), 
                                        labels=labels(i), title=title,
                                        path=path)

        # Plot control
        labels = [r'$u_1(t)$ (N/m)', r'$u_2(t)$ (N/m)']
        paper_plot.plt_subplot_vs_t(t, u_ctrl[0, :], u_ctrl[1, :], 
                                    labels=labels, title=title, path=path)
    
# %------------------------------------------ 3 Link Robot Classe -------------------------------------% #
class ThreeLinkManipulatorRobot(ManipulatorRobot):
    def __init__(self,
                 ms:List[float], 
                 Ls:list[float],
                 Kd:np.ndarray,
                 x_sys0:np.ndarray, 
                 trajectory, r_lin:np.ndarray = None,
                 sys_type:str = "Nonlinear",
                 dissipitivity:str = "nonsquare",
                 disturbance_input=None, 
                 disturbance_output=None) -> None:
        # Set DOF
        self.DOF = 3
        
        # Physical properties
        self.m1, self.m2, self.m3 = ms
        self.L1, self.L2, self.L3 = Ls
        
        # Store Damping Matrix
        self.Kd = Kd
        
        # Compute and Store intertias for each arm about center
        self._J1 = (self.m1 * self.L1**2) / 12
        self._J2 = (self.m2 * self.L2**2) / 12
        self._J3 = (self.m3 * self.L3**2) / 12
        
        # Initialize the system
        super().__init__(x_sys0=x_sys0,
                         trajectory=trajectory,
                         r_lin=r_lin,
                         sys_type=sys_type,
                         dissipitivity=dissipitivity,
                         disturbance_input=disturbance_input, 
                         disturbance_output=disturbance_output)
        
    # %---------------------------------------- Methods --------------------------------------% #
    def _M(self, x) -> np.ndarray:
        # Extract states.
        th2, th3 = x[1:self.DOF, 0]  # th2, th3
        
        M11 = self.L1**2 * (0.25*self.m1 + self.m2 + self.m3) + \
              self.L2**2 * (0.25*self.m2 + self.m3) + \
              self.L3**2 * (0.25*self.m3) + \
              self.L1 * self.L2 * (self.m2 + 2*self.m3) * np.cos(th2) + \
              self.L2 * self.L3 * self.m3 * np.cos(th3) + \
              self.L1 * self.L3 * self.m3 * np.cos(th2 + th3) + \
              self._J1 + self._J2 + self._J3
              
        M12 = self.L2**2 * (0.25*self.m2 + self.m3) + \
              self.L3**2 * (0.25*self.m3) + \
              self.L1 * self.L2 * (0.5*self.m2 + self.m3) * np.cos(th2) + \
              self.L2 * self.L3 * self.m3 * np.cos(th3) + \
              self.L1 * self.L3 * (0.5*self.m3) * np.cos(th2 + th3) + \
              self._J2 + self._J3
              
        M13 = self.L3**2 * (0.25*self.m3) + \
              self.L2 * self.L3 * (0.5*self.m3) * np.cos(th3) + \
              self.L1 * self.L3 * (0.5*self.m3) * np.cos(th2 + th3) + \
              self._J3
              
        M22 = self.L2**2 * (0.25*self.m2 + self.m3) + \
              self.L3**2 * (0.25*self.m3) + \
              self.L2 * self.L3 * self.m3 * np.cos(th3) + \
              self._J2 + self._J3
        
        M23 = self.L3**2 * (0.25*self.m3) + \
              self.L2 * self.L3 * (0.5*self.m3) * np.cos(th3) + \
              self._J3
              
        M33 = self.L3**2 * (0.25*self.m3) + \
              self._J3
        
        mass = np.array([[M11, M12, M13],
                         [M12, M22, M23],
                         [M13, M23, M33]])
        return mass

    def _fnon(self, x) -> np.ndarray:
        # Extract states
        _, th2, th3, dot_th1, dot_th2, dot_th3 = x.ravel()
        F1 = -(dot_th2**2) * \
              (0.5*self.L1 * self.L2 * self.m2 * np.sin(th2) + \
               self.L1 * self.L2 * self.m3 * np.sin(th2) + \
               0.5*self.L1 * self.L3 * self.m3 * np.sin(th2 + th3)) - \
              (dot_th3**2) * \
              (0.5*self.L2 * self.L3 * self.m3 * np.sin(th3) + \
               0.5*self.L1 * self.L3 * self.m3 * np.sin(th2 + th3)) - \
              dot_th1 * dot_th2 * \
              (self.L1 * self.L2 * self.m2 * np.sin(th2) + \
               2*self.L1 * self.L2 * self.m3 * np.sin(th2) + \
               self.L1 * self.L3 * self.m3 * np.sin(th2 + th3)) - \
              dot_th2 * dot_th3 * \
              (self.L1 * self.L3 * self.m3 * np.sin(th2 + th3) + \
               self.L2 * self.L3 * self.m3 * np.sin(th3)) - \
              dot_th1 * dot_th3 * \
               (self.L2 * self.L3 * self.m3 * np.sin(th3) + \
                self.L1 * self.L3 * self.m3 * np.sin(th2 + th3))
        
        F2 = (dot_th1**2) * \
             (0.5*self.L1 * self.L2 * self.m2 * np.sin(th2) + \
              self.L1 * self.L2 * self.m3 * np.sin(th2) + \
              0.5*self.L1 * self.L3 * self.m3 * np.sin(th2 + th3)) - \
             (0.5*dot_th3**2) * \
             (self.L2 * self.L3 * self.m3 * np.sin(th3)) - \
             dot_th1 * dot_th3 * \
             (self.L2 * self.L3 * self.m3 * np.sin(th3)) - \
             dot_th2 * dot_th3 * \
             (self.L2 * self.L3 * self.m3 * np.sin(th3))
        
        F3 = (dot_th1**2) * \
             (0.5*self.L2 * self.L3 * self.m3 * np.sin(th3) + \
              0.5*self.L1 * self.L3 * self.m3 * np.sin(th2 + th3)) + \
             (0.5*dot_th2**2 + dot_th1 * dot_th2) * \
             (self.L2 * self.L3 * self.m3 * np.sin(th3))
             
        nonlinear_forces = np.array([[F1], 
                                     [F2], 
                                     [F3]])
        return nonlinear_forces