# confined-water-analysis

Package to analyse molecular dynamics trajectories (classical and PIMD) of water confined in low-dimensional materials, e.g. graphene or hexagonal boron nitride. So far, the application is tailored for trajectories generated with CP2K. 

Tests can be run using `pytest` from the `tests` directory.

To make sure the code runs properly and does not return false results, it is recommended to wrap the trajectories you would like to use. This can be done with VMD and an example can be found in `helpers/vmd/wrapping_trajectories`.
