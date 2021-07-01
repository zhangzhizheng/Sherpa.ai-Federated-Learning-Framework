# Differential Privacy

Differential privacy (DP) allows achieving additional privacy protection in data communication. 
The packages provided in the Framework can be used in the federated learning experiments 
to assess the trade-off between privacy achieved and model's accuracy.

The structure of the implementation is as follows: 

- The [Mechanisms](../mechanisms/) package provides support for several DP tools that can be 
used in order to achieve additional privacy protection in federated learning experiments. 
- In package [Sensitivity Sampler](../sensitivity_sampler/) a sensitivity sampler is provided, 
  that can be used to estimate the noise calibration for a generic function seen as a black-box.
- The [Composition](../composition/) package implements the basic and advanced composition theorems 
  are for measuring the privacy budget spent. 
- The [Sampling](../sampling/) package offers the possibility to amplify the DP guarantee by sub-sampling.