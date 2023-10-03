This page provides codes for Physics-Informed Neural Networks in a spring-slider problem.

# Paper information:
- Title: Physics-Informed Neural Networks for fault slip monitoring: simulation, frictional parameter estimation, and prediction on slow slip events in a spring-slider system
- Atuthors: Rikuto Fukushima, Masayuki Kano, and Kazuro Hirahara
- Journal: Journal of Geophysical Research: Solid Earth
- Abstract: The episodic transient fault slips called slow slip events (SSEs) have been observed in many subduction zones. These slips often occur in regions adjacent to the seismogenic zone during the interseismic period, making monitoring SSEs significant for understanding large earthquakes. Various fault slip behaviors, including SSEs and earthquakes, can be explained by the spatial heterogeneity of frictional properties on the fault. Therefore, estimating frictional properties from geodetic observations and physics-based models is crucial for fault slip monitoring. In this study, we propose a Physics-Informed Neural Network (PINN)-based new approach to simulate fault slip evolutions, estimate frictional parameters from observation data, and predict subsequent fault slips. PINNs, which integrate physical laws and observation data, represent the solution of physics-based differential equations. As a first step, we validate the effectiveness of the PINN-based approach using a simple single-degree-of-freedom spring-slider system to model SSEs. As a forward problem, we successfully reproduced the temporal evolution of SSEs using PINNs and indicated how we should choose the appropriate collocation points depending on the residuals of physics-based differential equations. As an inverse problem, we estimated the frictional parameters from synthetic observation data and demonstrated the ability to obtain accurate values regardless of the choice of first-guess values. Furthermore, we discussed the potential of the predictability of the subsequent fault slips using limited observation data, taking into account uncertainties. Our results indicate the significant potential of PINNs for fault slip monitoring.

# System requirements:
- Windows 11
- Python 3.10.9
- Pytorch 2.0.0

# Instructions :
## Forward Calculation
- The forward calculation can be conducted by runnnig Forward_Main.ipynb. 
- All parameters used in main file are written in Forward_Parameter.py.
- Forward_Main.ipynb imports PINN_Preparation.py, PINN_Save.py, and Forward_Plot.py as original libraries.

## Inverse Calculation
- The inverse calculaton can be conducted by running Inverse_Main.ipynb.
- All parameters used in main file are written in Inverse_Parameter.py.
- Inverse_Main.ipynb imports PINN_Preparation.py, PINN_Save.py, Inverse_Synthetic_data.py. and Inverse_Plot.py as original libraries.
- PINN_Interpolated_Numerical_Solution.pth is used as a reference to make synthetic observation data in Inverse_Synthetic_data.py. 
