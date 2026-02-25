## [EuroVis 2025] HyperFLINT: Hypernetwork-based Flow Estimation and Temporal Interpolation for Scientific Ensemble Visualization

**Hamid Gadirov**, [Qi Wu](https://wilsoncernwq.github.io/), [David Bauer](https://davidbauer.me), [Kwan-Liu Ma](https://www.cs.ucdavis.edu/~ma/index.html), Jos BTM Roerdink, Steffen Frey  
*Computer Graphics Forum (EuroVis 2025), Wiley*  
[Paper (CGF)](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.70134)

Official implementation of the EuroVis 2025 paper *"HyperFLINT: Hypernetwork-based Flow Estimation and Temporal Interpolation for Scientific Ensemble Visualization"*.

---

### Abstract

We present **HyperFLINT** (Hypernetwork-based FLow estimation and temporal INTerpolation), a novel deep learning-based approach for estimating flow fields, temporally interpolating scalar fields, and facilitating parameter space exploration in spatio-temporal scientific ensemble data.  
This work addresses the critical need to explicitly incorporate ensemble parameters into the learning process, as traditional methods often neglect these, limiting their ability to adapt to diverse simulation settings and provide meaningful insights into the data dynamics. HyperFLINT introduces a hypernetwork to account for simulation parameters, enabling it to generate accurate interpolations and flow fields for each timestep by dynamically adapting to varying conditions, thereby outperforming existing parameter-agnostic approaches.  
The architecture features modular neural blocks with convolutional and deconvolutional layers, supported by a hypernetwork that generates weights for the main network, allowing the model to better capture intricate simulation dynamics. A series of experiments demonstrates HyperFLINT’s significantly improved performance in flow field estimation and temporal interpolation, as well as its potential in enabling parameter space exploration, offering valuable insights into complex scientific ensembles.

---

### BibTeX

```bibtex
@inproceedings{gadirov2025hyperflint,
  title={HyperFLINT: Hypernetwork-based Flow Estimation and Temporal Interpolation for Scientific Ensemble Visualization},
  author={Gadirov, Hamid and Wu, Qi and Bauer, David and Ma, Kwan-Liu and Roerdink, Jos BTM and Frey, Steffen},
  booktitle={Computer Graphics Forum},
  volume={44},
  number={3},
  pages={e70134},
  year={2025},
  organization={Wiley Online Library}
}
