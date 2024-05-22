# Matrix-Scheduling of QSR-Dissipative Systems

Accompanying code for the control of the planar rigid three-link robotic manipulator used in Section VI of Matrix-Scheduling of QSR-Dissipative Systems

## Required License
The LMI solver used along CVXPY is MOSEK, which requires a license to use. A personal
academic license can be requested [here](https://www.mosek.com/products/academic-licenses/).

## Installation

To clone the repository, run
```sh
$ git clone git@github.com:SepehrMoalemi/Moalemi_Forbes_TAC_Matrix_Scheduling_of_QSR_Dissipative_Systems.git
```

To install all the required dependencies for this project, run
```sh
$ cd Moalemi_Forbes_TAC_Matrix_Scheduling_of_QSR_Dissipative_Systems
$ pip install -r ./requirements.txt
```

## Usage
To generate Figure 6 and Figure 9 in the paper, run
```
$ python main.py
```

The plots can be saved to `./Figures` by setting 
```py
save_fig = True
```
