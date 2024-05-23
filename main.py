# Title   : Matrix-Scheduling of QSR-Dissipative Systems
# Authors : Sepehr Moalemi and James Richard Forbes
# Code    : Minimum Code to Reproduce the Application Example in Section VI
# %------------------------------------------ Packages -------------------------------------------% #
import argparse
import matplotlib.pyplot as plt

from src import paper_plot
from src import three_link_robot_compare 
# %-------------------------------------------- Main ---------------------------------------------% #
def main():
    # Set Figure Save Preferences
    paper_plot.set_fig_preferences()
    parser = argparse.ArgumentParser()
    parser.add_argument('--savefig', action='store_true', default=False)
    save_fig = parser.parse_args().savefig
    
    # Run Simulation
    three_link_robot_compare.simulate(tracking_type="static theta1, dynamic for the rest",
                                      sys_type="Nonlinear",
                                      controller_type="QSR",
                                      dissipitivity="nonsquare",
                                      model_uncertainty=True,
                                      T_END=15,
                                      plotting_type="paper_ready",
                                      save_fig=save_fig)
# %--------------------------------------------- Run ---------------------------------------------% #
if __name__ == '__main__':
    print(f'{"Start":-^{50}}')
    main()
    plt.show()
    print(f'{"End":-^{50}}')