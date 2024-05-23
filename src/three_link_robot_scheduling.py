# Purpose: Simulate the Control of the Three Link Robot in Figure 4 Using Gain Scheduling
# %------------------------------------------ Packages -------------------------------------------% #
import pathlib
import numpy as np

from scipy import integrate

from src import initialize
from src import paper_plot
# %----------------------------------------- Initializing ------------------------------------------% #
def init_linearization(tracking_type="static theta1, dynamic for the rest",
                       dissipitivity="nonsquare",
                       model_uncertainty="True"):
    match tracking_type:
        case "static theta1, dynamic for the rest":
            # [PLANT PARAMS]: Pick Linearization points
            r_lin_1 = np.array([[0],           # theta_1
                                [8*np.pi/9],   # theta_2
                                [ -np.pi/2]])  # theta_3
            
            r_lin_2 = np.array([[0],           # theta_1
                                [ np.pi/4],    # theta_2
                                [ np.pi/4]])   # theta_3
            
            r_lin_3 = np.array([[0],           # theta_1
                                [ -np.pi/2],   # theta_2
                                [8*np.pi/9]])  # theta_3
            
        case _:
            raise ValueError("Invalid tracking type.")

    # Plant Linearization points
    r_lin_points = [r_lin_1, r_lin_2, r_lin_3]
    
    # Construct Linearized Systems
    lin_systems = [initialize.init_sys(DOF=3,
                                       sys_type="Linearized", tracking_type=tracking_type, 
                                       dissipitivity=dissipitivity, r_lin=r,
                                       model_uncertainty=model_uncertainty) for r in r_lin_points]
    return lin_systems

def init_controllers(sys_non_linear, sys_linearized, ctrl_typ="QSR"):
    # Construct Controllers
    ctrls = [initialize.init_controller(sys=sys, ctrl_typ=ctrl_typ) for sys in sys_linearized]
                          
    # Set up closed-loop IC
    x_ctrls0 = np.tile(ctrls[0].x_ctrl0, (len(ctrls), 1))
    x_cl0 = np.concatenate((sys_non_linear.x_sys0, x_ctrls0, ))
    return ctrls, x_cl0
    
def init_scheduling_signals(Ss, 
                            scheduling_type="matrix", 
                            tracking_type="static theta1, dynamic for the rest"):
    match tracking_type:
        case "static theta1, dynamic for the rest":
            def s1(t):
                if t<1:
                    return 1
                elif t<=4:
                    return 1 - ((t-1)/3)**4
                else:
                    return 0
            def s2(t):
                if t<1:
                    return 0
                elif t<=9:
                    return 1 - ((t-5)/4)**4
                else:
                    return 0  
            def s3(t):
                if t<7:
                    return 0
                elif t<=9:
                    return 1 - ((t-9)/2)**4
                else:
                    return 1  
        case _:
            raise ValueError("Invalid tracking type.")
    
    # [Scheduling Signals]: Set up scheduling signals
    # Construct scheduling signals based on rank of QSR S matrix
    rank = np.linalg.matrix_rank(Ss[0])
    ny, nu = Ss[0].shape
    
    match scheduling_type:
        case "matrix":
            # Full Rank Case
            if rank == ny == nu:
                # Set alphas
                alphas = [1, 1, 1]
                
                # Set S_u
                Su_1 = lambda t: np.array([[s3(t), 0    , 0 ],
                                           [0    , s1(t), 0 ],
                                           [0    , 0    , s1(t) + s3(t)]])
                
                Su_2 = lambda t: np.array([[s2(t), 0    , 0 ],
                                           [0    , s2(t), 0 ],
                                           [0    , s1(t), s2(t)]])
                
                Su_3 = lambda t: np.array([[s1(t), 0    , 0 ],
                                           [0    , s3(t), 0 ],
                                           [0    , 0    , s1(t) + s3(t)]])

                # Set S_y = S_u.T
                Sy_1 = lambda t: alphas[0] * Su_1(t).T
                Sy_2 = lambda t: alphas[1] * Su_2(t).T
                Sy_3 = lambda t: alphas[2] * Su_3(t).T
                
            # Full Row Rank Case
            elif rank == ny < nu:
                """
                    For commuting property, we need
                    Sy_i.T @ Ss = Ss @ Su_i
                    
                    We showed 
                    AS = SB
                    
                    if and only if
                    A = U @ [sigma @ B_11 @ sigma^(-1), A_12;
                              0                       , A_22] @ U.T
                    B = V @ [B_11, 0;
                             B_21, B_22] @ V.T
                             
                    So, we have:
                    Sy_i = U @ [sigma^(-1) @ B_11.T @ sigma, 0;
                                A_12.T                     , A_22.T] @ U.T
                    Su_i = V @ [B_11, 0;
                                B_21, B_22] @ V.T
                                
                    For the non-square case, we have:
                    ~ u = [theta_1_dot, theta_2_dot, theta_3_dot]
                    ~ y = [tau_1, tau_2]
                    
                    For Ss being in 2x3 and full rank, we have:
                    ~ rank(Ss) = r = ny = 2 < nu = 3
                    ~ sigma is in 2x2
                    ~ U is 2x2
                    ~ V is 3x3
                    
                    So we have:
                    ~ B_11 is 2x2, B_21 is 1x2, B_22 is 1x1
                    ~ A_12 and A_22 vanish
                    
                    Here we have U = I and sigma = 1/2 * I, therefore:
                    Sy_i = B_11.T
                    Su_i = V @ [B_11, 0;
                                B_21, B_22] @ V.T
                                
                    To furthere improve performance, it is possible to premultiply 
                    V @ [B_11, 0;
                         B_21, B_22] @ V.T
                    symbolically.
                """  
                # Get SVD of S
                _, _, V1 = np.linalg.svd(Ss[0])
                _, _, V2 = np.linalg.svd(Ss[1])
                _, _, V3 = np.linalg.svd(Ss[2])
                
                # Set all alphas to 1
                alphas = [1, 1, 1]
                
                # Set Bk_11 for k = 1, 2, 3
                B1_11 = lambda t: np.array([[s1(t), -0.5*s1(t)],
                                             [0   , 0]])
                
                B2_11 = lambda t: np.diag([s2(t) + s1(t), 
                                           s2(t) + s3(t)])
                
                B3_11 = lambda t: np.array([[s3(t), 0],
                                            [s1(t), s3(t)]])
                
                # Set S_u
                O = np.zeros((2,1))
                Su_1 = lambda t: V1 @ np.block([[B1_11(t), O],
                                                [s3(t), s2(t), s1(t)]]) @ V1.T
                
                Su_2 = lambda t: V2 @ np.block([[B2_11(t), O],
                                                [0, 0, s2(t)]]) @ V2.T
                
                Su_3 = lambda t: V3 @ np.block([[B3_11(t), O],
                                                [0, 0, s2(t) + s3(t)]]) @ V3.T
                
                # Set S_y
                Sy_1 = lambda t: alphas[0] * B1_11(t).T 
                Sy_2 = lambda t: alphas[1] * B2_11(t).T 
                Sy_3 = lambda t: alphas[2] * B3_11(t).T 
                    
            # Full Column Rank Case
            elif rank == nu < ny:
                raise NotImplementedError
                
            # Rank Deficient Case
            else:
                raise NotImplementedError
                
        case "scalar":
            # Set all alphas to 1
            alphas = [1, 1, 1]
            
            # Set S_u = s * I
            Su_1 = lambda t: s1(t) * np.eye(nu)
            Su_2 = lambda t: s2(t) * np.eye(nu)
            Su_3 = lambda t: s3(t) * np.eye(nu)
            
            # Set S_y = s * I
            Sy_1 = lambda t: alphas[0] * s1(t) * np.eye(ny)
            Sy_2 = lambda t: alphas[1] * s2(t) * np.eye(ny)
            Sy_3 = lambda t: alphas[2] * s3(t) * np.eye(ny)
        
    signals = [(Su_1, Sy_1), (Su_2, Sy_2), (Su_3, Sy_3)]
    return signals

# %----------------------------------------- System Dynamics ----------------------------------------------% #
def gain_scheduled_closed_loop(sys, ctrls, signals, t, x) -> np.array:
        """Closed-loop system."""
        # Split state
        dim = sys.DOF * 2
        x_sys  = x[:dim]  # [theta_i, theta_dot_i]
        x_ctrl = x[dim:] # [x_c]
        
        # Dim of problem
        dim = len(x_sys)

        # Measurment
        y  = sys.g(x_sys)
        yp = sys.g_prewrap(x_sys)
        
        # Compute errors
        error     = sys.trajectory.r_des(t) - yp
        error_dot = sys.trajectory.r_des_dot(t) - y
        
        # Initialize control
        u_ctrl = np.zeros((ctrls[0].C.shape[0], 1))
        
        # Advance controller state.
        x_dot_ctrl = np.zeros_like(x_ctrl)
        
        # Iterate through controllers
        for i, (ctrl, signal) in enumerate(zip(ctrls, signals)):
            # Extract controller state
            x_ctrl_i = x_ctrl[i*dim:(i+1)*dim]
            
            # Compute signal
            Su_i, Sy_i = signal
            Su_i, Sy_i = Su_i(t), Sy_i(t)
            
            # Compute error for each controller
            error_dot_i = Su_i @ error_dot
            
            # Advance controller state.
            x_dot_ctrl[i*dim:(i+1)*dim] = ctrl.f(x_ctrl_i, error_dot_i)
            
            # Compute control
            u_ctrl = u_ctrl + Sy_i @ ctrl.g(x_ctrl_i, error_dot_i)
        
        # Add Proportional control prewrap
        u_ctrl = ctrl.g_prewrap(error) + sys.bhat @ u_ctrl
        
        # Advance system state
        x_dot_sys = sys.f(x_sys, u_ctrl)

        # Concatenate state derivatives
        return np.concatenate((x_dot_sys, x_dot_ctrl)) # x_dot

# %-------------------------------------------- Helper Functions ----------------------------------------------% #
def compute_control_effort(sys, t, N, x_ctrl, q, q_dot, ctrls, signals) -> np.ndarray:
    # Compute error and control (for plotting purposes)
    u_ctrl   = np.zeros((sys.DOF, N))
    error_th = np.zeros((sys.DOF, N))
    error_th_dot = np.zeros((sys.DOF, N))
         
    dim = len(q) + len(q_dot)
    
    # Control Dim
    u_dim = np.zeros((ctrls[0].C.shape[0], 1))
    
    for i, ti in enumerate(t):
        error_th[:, [i]]     = q[:, [i]] - sys.trajectory.r_des(ti)
        error_th_dot[:, [i]] = q_dot[:, [i]] - sys.trajectory.r_des_dot(ti)
        
        u_ctrl_i = np.zeros_like(u_dim)
        
        for j, (ctrl, signal) in enumerate(zip(ctrls, signals)):
            # Extract controller state
            x_ctrl_i = x_ctrl[j*dim:(j+1)*dim, [i]]
            
            # Compute signal
            Su_i, Sy_i = signal
            Su_i, Sy_i = Su_i(ti), Sy_i(ti)
            
            # Compute error for each controller
            error_dot_i = Su_i @ error_th_dot[:, [i]]
            
            # Compute control
            u_ctrl_i = u_ctrl_i + Sy_i @ ctrl.g(x_ctrl_i, error_dot_i)
       
        # Add Proportional control prewrap
        u_ctrl[:, [i]] = ctrl.g_prewrap(error_th[:, [i]]) + sys.bhat @ u_ctrl_i 
                
    return error_th, error_th_dot, u_ctrl

def extract_states(sys, sol, ctrls, signals):
    # Extract states
    t     = sol.t
    sol_x = sol.y
    
    nStates = 2*sys.DOF
    x_sys  = sol_x[:nStates, :]
    x_ctrl = sol_x[nStates:, :]
    
    q = x_sys[:sys.DOF, :]
    q_dot = x_sys[sys.DOF:, :]
    
    # Convert angles to degrees
    th     = [q[i,:]* 180 / np.pi for i in range(sys.DOF)]
    dot_th = [q_dot[i,:]* 180 / np.pi for i in range(sys.DOF)]
    
    # Number of points
    N = len(t)
    
    # Compute Error for plotting
    error_th, error_th_dot, u_ctrl = compute_control_effort(sys, t, N, x_ctrl, q, q_dot, ctrls, signals)
    
    return t, th, dot_th, u_ctrl, error_th, error_th_dot, sys.trajectory 
    
def plot_results(sys_non_linear, sol, ctrls, signals, save_fig=False):
    # Plotting Path
    path = None
    if save_fig:
        path = pathlib.Path('figs')
        path.mkdir(exist_ok=True)
        
    # Extract states
    DOF = sys_non_linear.DOF
    t, th, dot_th, u_ctrl, error_th, error_th_dot, _ = extract_states(sys_non_linear, sol, ctrls, signals)
    
    # Title
    title = f"{sys_non_linear.sys_type} model using gain scheduling VSP_lqr"
    
    # Plot theta and theta_dot
    labels = lambda indx : [rf'$\theta_{indx+1}(t)$ (deg)',
                            rf'$\dot{{\theta}}_{indx+1}(t)$ (deg/s)']

    for i in range(DOF):
        paper_plot.plt_subplot_vs_t(t, th[i], dot_th[i], 
                            labels=labels(i), title=title,
                            trajectory=sys_non_linear.trajectory, 
                            entry=i)
        
    # Plot error and error_dot
    labels = lambda indx : [rf'error in $\theta_{indx+1}(t)$ (deg)',
                            rf'error in $\dot{{\theta}}_{indx+1}(t)$ (deg/s)']
    
    err     = lambda i : [e * 180 / np.pi for e in error_th[i]]
    err_dot = lambda i : [e * 180 / np.pi for e in +error_th_dot[i]]
    
    for i in range(DOF):
        paper_plot.plt_subplot_vs_t(t, err(i), err_dot(i), labels=labels(i), title=title)
        
    # Plot control
    labels = [r'$u_1(t)$ (N/m)', r'$u_2(t)$ (N/m)']
    paper_plot.plt_subplot_vs_t(t, u_ctrl[0, :], u_ctrl[1, :], 
                        labels=labels, title=title, path=path)
    
# %----------------------------------------- simulate ------------------------------------------% #
def simulate(tracking_type="static theta1, dynamic for the rest",
             scheduling_type="matrix",
             controller_type="QSR",
             dissipitivity="nonsquare",
             model_uncertainty=True,
             disturbance_input=None,
             disturbance_output=None,
             T_END=15,
             plot=True,
             save_fig=False):

    # [SOLVER PARAMS]: Set IVP Solver parameters
    IVP_PARAM = initialize.init_ivp(T_END=T_END)

    # Construct Nonlinear System
    DOF=3
    sys_non_linear = initialize.init_sys(DOF=DOF,
                                         sys_type="Nonlinear", 
                                         tracking_type=tracking_type,
                                         dissipitivity=dissipitivity,
                                         disturbance_input=disturbance_input,
                                         disturbance_output=disturbance_output)
    
    # Construct Linearized Systems
    lin_systems = init_linearization(tracking_type=tracking_type,
                                     dissipitivity=dissipitivity,
                                     model_uncertainty=model_uncertainty)
    
    # Construct Controllers
    ctrls, x_cl0 = init_controllers(sys_non_linear=sys_non_linear, 
                                    sys_linearized=lin_systems, 
                                    ctrl_typ=controller_type)
    
    # Set Controller Dissiaitivity S
    match controller_type:
        case "QSR":
            Ss = [ctrl.S for ctrl in ctrls]
        case _:
            raise ValueError("Invalid controller type.")
    
    # Construct scheduling signals
    signals = init_scheduling_signals(Ss=Ss, 
                                      scheduling_type=scheduling_type, 
                                      tracking_type=tracking_type)
    
    # Construct gain scheduled system closed_loop dynamics
    sys_closed_loop = lambda t, x : gain_scheduled_closed_loop(sys=sys_non_linear, 
                                                               ctrls=ctrls, 
                                                               signals=signals, 
                                                               t=t, x=x)
    
    # Find time-domain response by integrating the ODE
    sol = integrate.solve_ivp(sys_closed_loop,
                              (IVP_PARAM.t_start, IVP_PARAM.t_end),
                              x_cl0.ravel(),
                              t_eval=IVP_PARAM.t_eval,
                              rtol=IVP_PARAM.rtol,
                              atol=IVP_PARAM.atol,
                              method=IVP_PARAM.method,
                              vectorized=True)
    
    # Plot results
    if plot:
        plot_results(sys_non_linear, sol, ctrls, signals, save_fig=save_fig)
    
    # Return solution
    return extract_states(sys_non_linear, sol, ctrls, signals)