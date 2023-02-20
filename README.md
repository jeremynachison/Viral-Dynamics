# A Cross Scale Model for Viral Dynamics

This is a repository of all current files relating to a cross scale model for viral dynamics. The model is structured via differential equations nested inside a network. Each node in the network is a system of differential equations modeling the Within-Host dynamics of a given virus. An edge of the network represents a connection/interaction between individuals (i.e. a "path" the virus can take to spread from one host to another). All the edges combined simulate the social structure/mixing of individuals. Whereas standard epidemiological models represent individuals as members of discrete classes with the transmissibility of a virus as constant in a given class, this model does not discretize individuals and allows for transmissibility to be modeled as a continuous function of the within-host viral dynamics.

The files in this repository are specifically geared towards modeling COVID-19, but the framework is flexible enough to accommodate any virus given a suitable differential equation model for the Within-Host dynamics and sufficient data to calibrate all parameters.

### Folder Structure

- __Calibration__: All code dedicated towards calibrating the parameters/exploring the parameter space of the model can be found here.

- __DiffEqs__: Sample code for the within-host model for COVID-19 (i.e. a single node in the network) can be found here. Some examples of alternative within-host models are also stored here to demonstrate how one could implement their own custom model.

- __Networks__: The full network simulation can be found here. Code to generate some plots based on different network structures can also be found here.
