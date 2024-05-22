# Purpose: QSR Controller Synthesized using the SDP given by the LMI in (72)
# %------------------------------------------ Packages -------------------------------------------% #
import numpy as np
import scipy
import control
import cvxpy

from src.toolbox import is_spd, is_snd, is_snsd
# %--------------------------------------- Abstract Class -----------------------------------------% #
class controller(object):
    def __init__(self, x_ctrl0, Kp):
        # Controller Initial condition
        self.x_ctrl0 = x_ctrl0
        
        # Prewrap P control
        self.Kp = Kp
        
    # Purpose: Overide print(class)
    def __repr__(self) -> str:
        # Printing Msg
        str = ""
        
        # Print Class Name
        str += f'Using {type(self).__name__}\n'
        
        # Format Numpy to print matrix in 1 line
        reformat = lambda m: np.array2string(m, 
                                             formatter={'all':lambda x: f"{x}"},
                                             separator=',').replace(',\n', ';')
        # Print all properties
        props_dic = vars(self)
        for prop, value in props_dic.items():
            if type(value) == np.ndarray:
                value = reformat(value)
            str += f'{prop} = {value}\n'
        return str
        
    # Purpose: Compute xc_dot
    def f(self, x, e) -> np.array:
        return self.A @ x + self.B @ e
    
    # Purpose: Compute u
    def g(self, x, e) -> np.array:
        return self.C @ x + self.D @ e
    
    # Purpose: Compute u_prewrap
    def g_prewrap(self, e) -> np.array:
        return self.Kp @ e

# %-------------------------------------- QSR Control Classes --------------------------------------% #
class QSR(controller):
    def __init__(self, sys, Qr, Rr, Kp, x_ctrl0, verbose=False) -> None:
        # Controller Initial condition
        super().__init__(x_ctrl0, Kp)
        
        DOF = sys.DOF
        Odof = np.zeros((DOF,DOF)) 
        Ap = sys.A + np.block([[Odof                        , Odof],
                               [-sys.M_inv(sys.r_lin) @ Kp  , Odof]])
        
        # Compute matrix gain using LQR
        K, _, _ = control.lqr(Ap, sys.B, Qr, Rr)
        
        # Set Dc and Cc
        self.D = np.zeros_like(sys.bhat.T)
        self.C = K
        
        # Find Q, S, R, Bc using QSR-Dissipativity Lemma
        self.Q, self.S, self.R, self.B = QSR.find_Q_S_R(sys=sys, A=Ap,
                                                        K=K,
                                                        verbose=verbose)
        
        # Construct Ac = Ap - B @ K - Bc @ C
        self.A = Ap - sys.B @ K - self.B @ sys.C
  
    def find_Q_S_R(sys, A, K, PD_TOL=1e-6, MOSEK_TOL=1e-12, verbose=False, solver_verbose=False):
        verbose = True
        
        # Extract System Parameters
        B, C = sys.B, sys.C
        Sp = sys.S
        
        # Set Dim
        x_dim = A.shape[0]  # x_dim = 6
        y_dim = C.shape[0]  # y_dim = 3
        u_dim = K.shape[0]  # u_dim = 2
        
        # Set Constants
        Ix = np.eye(x_dim)
        
        # Define Variables
        P = cvxpy.Variable(shape=(x_dim, x_dim), symmetric=True)
        
        Q = cvxpy.Variable(shape=(u_dim, u_dim), symmetric=True)
        
        R = np.zeros((y_dim, y_dim))                             
        
        # QSR-Dissipativity Lemma [20]
        S = Sp.T
        F = K.T@S                     
        constraints  = [P >> PD_TOL*Ix]
        constraints += [P@(A - B@K) + (A - B@K).T@P - F@C - (F@C).T - K.T@Q@K << 0] 
        
        # QSR-Dissipativity Stability Theorem [3]
        constraints += [Q << 1e-6*np.eye(u_dim)]

        # Subject to condition minimize trace(R) which leads to R = 0
        min_problem = cvxpy.Minimize(0)
    
        # Solve Constraint Problem
        mosek_params = {'MSK_DPAR_INTPNT_CO_TOL_DFEAS':   MOSEK_TOL,
                        'MSK_DPAR_INTPNT_CO_TOL_MU_RED':  MOSEK_TOL,
                        'MSK_DPAR_INTPNT_CO_TOL_PFEAS':   MOSEK_TOL,
                        'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': MOSEK_TOL}
        
        problem = cvxpy.Problem(min_problem, constraints)
        problem.solve(solver='MOSEK', mosek_params=mosek_params, verbose=solver_verbose)
        
        # Check if solver converged
        if problem.status not in [cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE]:
            assert False, f'Solver did not converge. Status: {problem.status}.'
        
        # Check Solution
        if verbose:
            print(problem.status)
        
        # Extract Solutions
        P = P.value
        Q = Q.value
        LMI_LQR = P@(A - B@K) + (A - B@K).T@P - F@C - (F@C).T - K.T@Q@K
        LMI_QSR = Q
        
        # Find Bc
        Bc = scipy.linalg.cho_solve(scipy.linalg.cho_factor(P), F)
        
        # Check if solution is valid
        with np.printoptions(precision=3):
            print("\nQ = ", Q)
            print("\nS = ", S)
            print("\nR = ", R)
            assert is_spd(P, verbose=verbose, var_name="P"), "P is not SPD"
            assert is_snsd(LMI_LQR, verbose=verbose, var_name="LMI_LQR"), "LMI_LQR is not SNSD"
            assert is_snd(LMI_QSR, verbose=verbose, var_name="LMI_QSR"), "LMI_QSR is not SND"
        return Q, S, R, Bc