# Title   : Matrix-Scheduling of QSR-Dissipative Systems
# Journal : IEEE Transactions on Automatic Control (TAC)
# Authors : Sepehr Moalemi and James Richard Forbes
# Code    :  Minimum Code to Reproduce the Application Example in Section VI
# %------------------------------------------ Packages -------------------------------------------% #
from src import paper_plot
from src import three_link_robot_compare 
# %-------------------------------------------- Main ---------------------------------------------% #
def main():
    three_link_robot_compare.simulate(tracking_type="static theta1, dynamic for the rest",
                                      sys_type="Nonlinear",
                                      controller_type="QSR",
                                      dissipitivity="nonsquare",
                                      model_uncertainty=True,
                                      T_END=15,
                                      plotting_type="paper_ready",
                                      save_fig=True)
# %--------------------------------------------- Run ---------------------------------------------% #
if __name__ == '__main__':
    print(f'{"Start":-^{50}}')
    paper_plot.set_fig_preferences()
    main()
    paper_plot.show()
    print(f'{"End":-^{50}}')