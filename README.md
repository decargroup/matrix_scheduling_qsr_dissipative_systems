# Matrix-Scheduling of QSR-Dissipative Systems

Accompanying code for the control of the planar rigid three-link robotic manipulator used in Section VI of Matrix-Scheduling of QSR-Dissipative Systems.

## Required License
The LMI solver used along CVXPY is MOSEK, which requires a license to use. A personal
academic license can be requested [here](https://www.mosek.com/products/academic-licenses/).

## Installation

To clone the repository, run
```sh
$ git clone git@github.com:decargroup/matrix_scheduling_qsr_dissipative_systems.git
```

To install all the required dependencies for this project, run
```sh
$ cd matrix_scheduling_qsr_dissipative_systems
$ pip install -r ./requirements.txt
```

## Usage
To generate Figures 6-9 in the paper, run
```sh
$ python main.py
```

The plots can be saved to `./Figures` by running 
```sg
$ python main.py --savefig
```
