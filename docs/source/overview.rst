Overview
========

The **Enhanced Surface Climate** module of the DTC Ice Sheets provides high-resolution surface climate information 
for the Antarctic ice sheets (a future update will include Greenland), focusing on surface mass balance (SMB) and 
surface melt. 
Climate models are too coarse to capture small local melt events in Antarctica, which can have a high influence on ice 
shelves stability.  
To overcome this limitation, the module offers two deep learning methods to sharpen the picture: 
one turns coarse weather data into detailed maps of surface mass gain and loss, and another builds detailed melt maps directly from 
satellite observations.  
The two workflows provide complementary but methodologically distinct types of super resolution:

*	The **Climate Model Emulator** reconstructs high-resolution surface mass balance (SMB) from coarse atmospheric forcing
*	**UMelt** combines satellite observations from different sensors to reconstruct high-resolution surface melt

Scientific background
---------------------

Regional climate models typically operate at resolutions of tens of kilometres over Antarctica, which can smooth 
the strong spatial gradients that control surface melt and surface mass balance (SMB), while satellite observations 
provide complementary information on melt but do not directly quantify all relevant surface-climate variables. 
The two components of this module use deep learning models to bridge spatial and/or temporal scale gaps, 
but differ fundamentally in their input data, training strategy, and target variable.

.. rubric:: Climate Model Emulator

The first module is climate-model emulator that follows the U-Net architecture introduced by Doury et al.(2022) 
and subsequently applied to Surface Mass Balance (SMB) downscaling over Antarctica by van der Meer et al. (2023).
The deep learning-based regional climate model (RCM) emulator learns to reconstruct high-resolution surface mass balance 
from coarse atmospheric forcing, thereby providing a computationally efficient alternative to explicitly rerunning a 
high-resolution dynamical RCM.

In the DTC implementation, atmospheric variables from ERA5 monthly means are used as forcing and 
2km SMB from RACMO are used as the target quantity.
The U-Net architecture follows an encoder-decoder configuration with skip connections. 
The encoder progressively extracts spatial features and large-scale context from the input predictors, 
while the decoder reconstructs the high-resolution target field. Skip connections transfer spatial information directly 
between corresponding encoder and decoder levels, helping the network retain fine-scale spatial structure that would 
otherwise be lost during successive downsampling operations. 
The training is formulated as a supervised reconstruction problem in which the high-resolution RCM simulation 
represents the target climate state and the coarse atmospheric fields contain the predictors from which 
that state is reconstructed. This enables the emulator to learn topographically and climatically controlled 
spatial structure rather than relying on purely geometric interpolation.


.. rubric:: UMelt

The second super-resolution component is based on UMelt, a modelling approach for surface melt based on satellite 
observations, introduced by de Roda Husman et al. (2024).

UMelt integrates complementary EO products, each characterised by different spatial resolutions, temporal sampling, 
and observational sensitivities. The principal dynamic inputs are coarse but frequently available microwave observations
from ASCAT and SSMIS, while Sentinel-1 provides the high-resolution melt information used as the training and 
validation reference. 
High-resolution surface elevation from REMA and a long-term Sentinel-1 melt-frequency field provide additional 
spatial context. The resulting input stack therefore combines information on the instantaneous melt state with 
persistent spatial controls on where melt is likely to occur. These observations are resampled onto a common 500 m grid 
and used to train the convolutional neural network, consisting of an attention UNet. 

The product has 500 m resolution and sub-daily sampling. It is precomputed, stored as GeoZarr in the EO Data Store, 
and accessible through the Query API. The model itself is not deployed for on-demand 
execution in DTC-IS given the computational resources required to preprocess the data, but it can be run locally.

References
----------

* De Roda Husman, Sophie, et al. (2024). "A high-resolution record of surface melt on Antarctic ice shelves using multi-source 
  remote sensing data and deep learning." Remote Sensing of Environment 301 : 113950.
  https://doi.org/10.1016/j.rse.2023.113950
* Van Der Meer, Marijn, Sophie de Roda Husman, and Stef Lhermitte (2023). "Deep learning regional climate model emulators: 
  A comparison of two downscaling training frameworks." Journal of Advances in Modeling Earth Systems 15.6: e2022MS003593.
  https://doi.org/10.1029/2022MS003593
* Doury, A., Somot, S., Gadat, S., Ribes, A., & Corre, L. (2022). Regional climate model emulator based on deep learning: Concept and first
  evaluation of a novel hybrid downscaling approach. Climate Dynamics, 60(5-6), 1751-1779. https://doi.org/10.1007/s00382-022-06343-9

