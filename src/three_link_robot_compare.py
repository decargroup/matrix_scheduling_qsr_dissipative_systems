# Purpose: Generate Plots in Figures 6 and 9
# %------------------------------------------ Packages -------------------------------------------% #
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from src import three_link_robot
from src import three_link_robot_scheduling
# %------------------------------------------ Plots ----------------------------------------------% #
def simulate(tracking_type= "static theta1, dynamic for the rest",
             sys_type= "Nonlinear",
             dissipitivity="nonsquare",
             controller_type="QSR",
             model_uncertainty=True,
             disturbance_input=None,
             disturbance_output=None,
             T_END=15,
             plotting_type="paper_ready",
             save_fig=False):
    
    # Check if the directory exists
    if save_fig:
        Path('./Figures').mkdir(parents=True, exist_ok=True)
        
    # Set DOF
    DOF = 3
    
    # Print padding
    padding = 50
    
    # Results: t, th, dot_th, u_ctrl, error_th, error_th_dot
    print(f'{"Simulating No Scheduling " + controller_type:-^{padding}}')
    single_lqr_results = three_link_robot.simulate(DOF=DOF,
                                                          tracking_type=tracking_type, 
                                                          controller_type=controller_type,
                                                          sys_type=sys_type,
                                                          dissipitivity=dissipitivity,
                                                          model_uncertainty=model_uncertainty,
                                                          disturbance_input=disturbance_input,
                                                          disturbance_output=disturbance_output,
                                                          T_END=T_END,
                                                          plot=False)
    
    # Results: t, th, dot_th, u_ctrl, error_th, error_th_dot, trajectory
    print(f'{"Simulating Scalar Scheduling " + controller_type:-^{padding}}')
    scalar_scheduled_lqr_results = three_link_robot_scheduling.simulate(tracking_type=tracking_type, 
                                                                      controller_type=controller_type,
                                                                      scheduling_type="scalar",
                                                                      dissipitivity=dissipitivity, 
                                                                      model_uncertainty=model_uncertainty,
                                                                      disturbance_input=disturbance_input,
                                                                      disturbance_output=disturbance_output,
                                                                      T_END=T_END,
                                                                      plot=False)
    
    print(f'{"Simulating Matrix Scheduling " + controller_type:-^{padding}}')
    matrix_scheduled_lqr_results = three_link_robot_scheduling.simulate(tracking_type=tracking_type, 
                                                                      controller_type=controller_type,
                                                                      scheduling_type="matrix",
                                                                      dissipitivity=dissipitivity,
                                                                      model_uncertainty=model_uncertainty,
                                                                      disturbance_input=disturbance_input,
                                                                      disturbance_output=disturbance_output,
                                                                      T_END=T_END,
                                                                      plot=False)
    
    # Generate Tracking Trajectory
    print(f'{"Generating Plots":-^{padding}}')
    trajectory = scalar_scheduled_lqr_results[6]
    rs, r_dots = trajectory.generate_trajectory(single_lqr_results[0], type="deg")
    
    # Plot theta and theta_dot
    t = single_lqr_results[0]
    results_th           = [rs, single_lqr_results[1], scalar_scheduled_lqr_results[1], matrix_scheduled_lqr_results[1]]
    results_dot_th       = [r_dots, single_lqr_results[2], scalar_scheduled_lqr_results[2], matrix_scheduled_lqr_results[2]]
    results_u_ctrl       = [single_lqr_results[3], scalar_scheduled_lqr_results[3], matrix_scheduled_lqr_results[3]]
    results_error_th     = [single_lqr_results[4], scalar_scheduled_lqr_results[4], matrix_scheduled_lqr_results[4]]
    results_error_th_dot = [single_lqr_results[5], scalar_scheduled_lqr_results[5], matrix_scheduled_lqr_results[5]]
    
    # Setup plotting type based on paper_ready or full
    match plotting_type:
        case "paper_ready":
            n_rows = DOF - 1
            index_set = [1, 2]
            
        case "final":
            n_rows = DOF
            index_set = [0, 1, 2]
        
        case _:
            raise ValueError("Invalid plotting type.")
    plot_set = list(range(n_rows))
    # %----------------------------------------------- Trajectory ----------------------------------------------% #
    colors = ['k', '#377eb8', '#ff7f00','#e41a1c']
    line_styles = ['-', ':', '-.', '--']
    labels = ["Desired", "Unscheduled", "Scalar GS", "Matrix GS"]
    lws    = [6, 5, 5, 4]
    fig, axs = plt.subplots(n_rows, 2, sharex=True, figsize=(12, 9))
    for i in plot_set:
        for j in range(len(results_th)):
            axs[i, 0].plot(t, results_th[j][index_set[i]], c=colors[j], ls=line_styles[j], lw=lws[j], label=labels[j])
            axs[i, 1].plot(t, results_dot_th[j][index_set[i]], c=colors[j], ls=line_styles[j], lw=lws[j],  label=labels[j])
        axs[i, 0].set_xlim([0, T_END])
        axs[i, 1].set_xlim([0, T_END])
        axs[i, 0].set_ylabel(rf'$\theta_{index_set[i]+1}(t)$ [deg]')
        axs[i, 1].set_ylabel(rf'$\dot{{\theta}}_{index_set[i]+1}(t)$ [deg/s]')
    fig.align_labels()
    axs[-1, 0].set_xlabel(r'Time $[s]$')
    axs[-1, 1].set_xlabel(r'Time $[s]$')
    axs[-1][-1].legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), borderaxespad=0.05) 
    
    plt.subplots_adjust(wspace=0.4)
    plt.subplots_adjust(hspace=0.1)
    if save_fig:
        file_name = "trajctory"
        fig.savefig('./Figures/' + file_name + ".pdf")
    
    # %----------------------------------------------- Error ----------------------------------------------% #
    colors = ['#377eb8', '#ff7f00','#e41a1c']
    lws    = [5, 5, 4]
    line_styles = [':', '-.', '-']
    labels = ["No GS", "Scalar GS", "Matrix GS"]
    rmse_errors = []
    rmse_errors_dot = []
    fig, axs = plt.subplots(n_rows, 2, sharex=True, figsize=(12, 10))
    for i in plot_set:
        for j in range(len(results_error_th)):
            err     = [e * 180 / np.pi for e in results_error_th[j][index_set[i]]]
            err_dot = [e * 180 / np.pi for e in results_error_th_dot[j][index_set[i]]]
            axs[i, 0].plot(t, err, c=colors[j], ls=line_styles[j], lw=lws[j], label=labels[j])
            axs[i, 1].plot(t, err_dot, c=colors[j], ls=line_styles[j], lw=lws[j], label=labels[j])
            rmse_errors+=[f"RMSE Error of theta_{index_set[i]+1} for {labels[j]}: {np.sqrt(np.mean(np.square(err))):.4f}"]
            rmse_errors_dot+=[f"RMSE Error of dot theta_{index_set[i]+1} for {labels[j]}: {np.sqrt(np.mean(np.square(err_dot))):.4f}"]
        axs[i, 0].set_xlim([0, T_END])
        axs[i, 1].set_xlim([0, T_END]) 
        axs[i, 0].set_ylabel(rf'$e_{index_set[i]+1}(t)$ [deg]')
        axs[i, 1].set_ylabel(rf'$\dot{{e}}_{index_set[i]+1}(t)$ [deg/s]')
    fig.align_labels()
    axs[-1, 0].set_xlabel(r'Time $[s]$')
    axs[-1, 1].set_xlabel(r'Time $[s]$')
    axs[0][-1].legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), borderaxespad=0.05)
    plt.subplots_adjust(wspace=0.35)
    plt.subplots_adjust(hspace=0.1)
    if save_fig:
        file_name = "error"
        fig.savefig('./Figures/' + file_name + ".pdf")
    
    # %---------------------------------------------- Torque ----------------------------------------------% #
    total_torque = []
    fig, axs = plt.subplots(1, n_rows, sharex=True, figsize=(12, 4))
    for i in plot_set:
        for j in range(len(results_error_th)):
            axs[i].plot(t, results_u_ctrl[j][index_set[i]], c=colors[j], ls=line_styles[j], lw=lws[j], label=labels[j])
            total_torque+=[f"Total control effort |tau_{index_set[i]+1}| for {labels[j]}: {np.sum(np.abs(results_u_ctrl[j][index_set[i]])):.4f}"]
        axs[i].set_xlim([0, T_END])
        axs[i].set_ylabel(rf'$\tau_{index_set[i]+1}(t)$ [N$\cdot$m]')
        axs[i].set_xlabel(r'Time $[s]$')
    fig.align_labels()
    axs[-1].legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), borderaxespad=0.05)
    plt.subplots_adjust(wspace=0.35)
    if save_fig:
        file_name = "control_effort"
        fig.savefig('./Figures/' + file_name + ".pdf")
        
    # %----------------------------------------------- Signals ----------------------------------------------% #
    match tracking_type:
        case "dynamic":
            # [Scheduling Signals]: Set up scheduling signals   
            s1 = lambda t: 1 - (t/3)**4 if t<=3 else 0
            s2 = lambda t: 1 - ((t-3)/2.8)**4 if 0.2<=t<=5.8 else 0
            def s3(t):
                if t<5:
                    return 0
                elif t<=7:
                    return 1 - ((t-7.5)/2.5)**4
                else:
                    return 1 
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
    xnew = np.arange(trajectory.tk[0][0], 12, 0.01)
    lw=4
    fig = plt.figure(figsize=(12, 4))
    plt.plot(xnew, [s1(t) for t in xnew], c='#377eb8', ls='-', lw=lw, label=r"$s_1(t)$")
    plt.plot(xnew, [s2(t) for t in xnew], c='#ff7f00', ls='--', lw=lw, label=r"$s_2(t)$")
    plt.plot(xnew, [s3(t) for t in xnew], c='#e41a1c', ls='-.',lw=lw,  label=r"$s_3(t)$")
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 0.95), borderaxespad=0.05,fontsize=25)
    plt.xlabel(r'Time $[s]$')
    plt.ylabel(r'Scheduling Signals')
    plt.xlim(0, xnew[-1])
    if save_fig:
        fig.savefig('./Figures/scheduling_signal.pdf')
    # %----------------------------------------------- Print Errors ----------------------------------------------% #
    print(f'{"Printing Table Values":-^{padding}}')
    # Print RMSE Errors
    for rmse in rmse_errors:
        print(rmse)
    print("")
    # Print RMSE Error Rates
    for rmse in rmse_errors_dot:
        print(rmse)
    print("")
    # Print Total Torques
    for torque in total_torque:
        print(torque)
        