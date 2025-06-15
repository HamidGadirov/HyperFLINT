# HyperFLINT
[EuroVis 2025] HyperFLINT: Hypernetwork-based Flow Estimation and Temporal Interpolation for Scientific Ensemble Visualization

We present HyperFLINT (Hypernetwork-based FLow estimation and temporal INTerpolation), a novel deep learning-based
approach for estimating flow fields, temporally interpolating scalar fields, and facilitating parameter space exploration in
spatio-temporal scientific ensemble data. This work addresses the critical need to explicitly incorporate ensemble parameters
into the learning process, as traditional methods often neglect these, limiting their ability to adapt to diverse simulation settings
and provide meaningful insights into the data dynamics. HyperFLINT introduces a hypernetwork to account for simulation
parameters, enabling it to generate accurate interpolations and flow fields for each timestep by dynamically adapting to varying
conditions, thereby outperforming existing parameter-agnostic approaches. The architecture features modular neural blocks with
convolutional and deconvolutional layers, supported by a hypernetwork that generates weights for the main network, allowing
the model to better capture intricate simulation dynamics. A series of experiments demonstrates HyperFLINT’s significantly
improved performance in flow field estimation and temporal interpolation, as well as its potential in enabling parameter space
exploration, offering valuable insights into complex scientific ensembles.
