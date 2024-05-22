# Purpose: Initialize the system and controller for the problem
# %------------------------------------------ Packages -------------------------------------------% #
import numpy as np

from typing import NamedTuple, Tuple

from src import controller
from src import system
from src import tracking
# %----------------------------------- IVP Solver Parameters ------------------------------------% #
# Purpose: Immutable Struct for solve_ivp params
class num_integ_settings(NamedTuple):
    t_start : float
    t_end   : float
    t_eval  : np.array
    rtol    : float
    atol    : float
    method  : str

# Purpose: Return only necessary parameters for solve_ivp
def init_ivp(DT=1e-3, T_START=0, T_END=30, 
             RTOL=1e-5, ATOL=1e-5,
             METHOD='RK45') -> NamedTuple:
    # Solve_ivp: Parameters
    T_EVAL = np.arange(T_START, T_END, DT)
    IVP_PARAM = num_integ_settings(t_start=T_START,
                                   t_end=T_END,
                                   t_eval=T_EVAL,
                                   rtol=RTOL,
                                   atol=ATOL,
                                   method=METHOD)
    return IVP_PARAM

# %--------------------------------------- Tracking ---------------------------------------------% #
# Purpose: Setup discerte points to track trajectory
def init_trajectory(DOF=3, 
                    tracking_type="static theta1, dynamic for the rest") -> tracking.Trajectory:
    match tracking_type:
        case "static theta1, dynamic for the rest":
            if DOF == 3:
                tk = [[ 0.0, np.array([[0], [8*np.pi/9], [-np.pi/2]])],
                      [ 2.0, np.array([[0], [8*np.pi/9], [-np.pi/2]])], 
                      [ 3.0, np.array([[0], [  np.pi/4], [ np.pi/4]])], 
                      [ 7.0, np.array([[0], [  np.pi/4], [ np.pi/4]])], 
                      [ 9.0, np.array([[0], [ -np.pi/2], [8*np.pi/9]])]]
        case _: 
            raise NotImplemented
    return tracking.Trajectory(DOF=DOF, tk=tk)

# %------------------------------------ System Parameters --------------------------------------% #
# Purpose: Construct the manipulator robot
def init_sys(DOF=3,
             sys_type="Nonlinear", 
             tracking_type="static theta1, dynamic for the rest", r_lin=None,
             dissipitivity="nonsquare",
             model_uncertainty=False,
             disturbance_input=None, 
             disturbance_output=None) -> system.ManipulatorRobot:
    # Construct Parameter Dictionary
    params = {"sys_type": sys_type, 
              "dissipitivity": dissipitivity,
              "disturbance_input": disturbance_input,
              "disturbance_output": disturbance_output}
    
    # Initialize mass and lengths based on DOF
    match DOF:
        case 3:
            M_1, M_2, M_3 = 2.0, 0.9, 0.3   # kg, mass
            L_1, L_2, L_3 = 1.1, 0.6, 0.5  # m
            ms = [M_1, M_2, M_3]
            Ls = [L_1, L_2, L_3]
            
            # Set the Damping Matrix
            Kd = np.diag([5, 2.5, 2.5])
            params.update({"Kd": Kd})
            
        case _:
            raise NotImplemented
        
    # Add Model Uncertainty
    def add_model_uncertainty(listVal, uncertainty):
        for i, l in enumerate(listVal):
            if i % 2 == 0:
                listVal[i] = l * (1.0 + uncertainty)
            else:
                listVal[i] = l * (1.0 - uncertainty)
        return listVal

    if model_uncertainty:
        print("Adding Model Uncertainty:")
        print("Original Masses: ", ms)
        print("Original Lengths: ", Ls)
        ms = add_model_uncertainty(listVal=ms, uncertainty=0.2)
        Ls = add_model_uncertainty(listVal=Ls, uncertainty=0.1)
        print("New Masses: ", ms)
        print("New Lengths: ", Ls)
    params.update({"ms": ms, "Ls": Ls})
    
    # initialize trajectory to track
    trajectory = init_trajectory(DOF=DOF,
                                 tracking_type=tracking_type)
    params.update({"trajectory": trajectory})
    
    # Initial condition
    x_sys0 = trajectory.x0
    params.update({"x_sys0": x_sys0})
    
    # linearization point
    if r_lin is None:
        r_lin = trajectory.r_end
    params.update({"r_lin": r_lin})
    
    # Initialize Manipulator Robot
    if DOF == 3:
        return system.ThreeLinkManipulatorRobot(**params)
    else:
        raise NotImplemented

# %------------------------------------ Controller Picker --------------------------------------% #
# Purpose: Initialize the controller
def init_controller(sys, ctrl_typ="QSR") -> controller.controller:
    match ctrl_typ:
        case "QSR":     ctrl = QSR(sys)
        case _: raise NotImplemented
    return ctrl

# %-------------------------------------- QSR Controllers --------------------------------------% #
# Purpose: Initialize the QSR controller
def QSR(sys) -> controller.controller:
    match sys.dissipitivity:
        case "nonsquare":
            if sys.DOF == 3:
                return init_3DOF_QSR_non_square_controllers(sys)
        case _:
            raise NotImplemented
    
def init_3DOF_QSR_non_square_controllers(sys) -> controller.controller:
    # [Controller]: VSP_lqr Proportional control prewrap and gain
    Kp = np.diag([5, 35, 35])
    
    # [Controller]: VSP_lqr State and input weight matrices
    u2_max = 25
    u3_max = 25
    
    theta_1_max = 15 / 180 * np.pi
    theta_2_max = 15 / 180 * np.pi
    theta_3_max = 15 / 180 * np.pi
    
    theta_1_dot_max = 10 / 180 * np.pi
    theta_2_dot_max = 10 / 180 * np.pi
    theta_3_dot_max = 10 / 180 * np.pi
    
    Qr = np.diag([1/theta_1_max, 1/theta_2_max, 1/theta_3_max, 1/theta_1_dot_max, 1/theta_2_dot_max, 1/theta_3_dot_max])**2
    Rr = np.diag([1/u2_max, 1/u3_max])**2
    
    # QSR: Initial condition
    x_ctrl0 = np.zeros((sys.DOF*2, 1))
    return controller.QSR(sys=sys, 
                          Qr=Qr, Rr=Rr, 
                          Kp=Kp,
                          x_ctrl0=x_ctrl0,
                          verbose=False)

# %------------------------------------- Problem Parameters ------------------------------------% #
# Purpose: Return only necessary parameters for the problem
def problem(DOF=3,
            tracking_type="static theta1, dynamic for the rest",
            controller_type="QSR",
            sys_type="Nonlinear", 
            dissipitivity="nonsquare",
            model_uncertainty=False,
            disturbance_input=None,
            disturbance_output=None) -> Tuple[system.ThreeLinkManipulatorRobot, 
                                              controller.controller, 
                                              np.array]:
    # Initialize True system with potential disturbance but no model uncertainty 
    sys = init_sys(DOF=DOF,
                   sys_type=sys_type, tracking_type=tracking_type,
                   dissipitivity=dissipitivity,
                   disturbance_input=disturbance_input,
                   disturbance_output=disturbance_output)
    
    # Initialize Measured system subject to model uncertainty
    sys_measured = init_sys(DOF=DOF,
                            sys_type=sys_type, tracking_type=tracking_type,
                            dissipitivity=dissipitivity,
                            model_uncertainty=model_uncertainty)
    
    # Initialize controller
    ctrl = init_controller(sys=sys_measured, 
                           ctrl_typ=controller_type)
    
    # Set up closed-loop IC.
    x_cl0 = np.concatenate((sys.x_sys0, 
                            ctrl.x_ctrl0))
    
    return sys, ctrl, x_cl0