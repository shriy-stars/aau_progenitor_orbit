Here is a repo to fit the progenitor orbit for the Atlas-Aliqa Uma stream. 


Inside /code directory you will find Python scripts to generate the likelihood, coordinate transformations, to generate the particle spray model. The likelihood is evaluated by fitting a spline function to simulated stream particles, and evaluating the likelihood of the data against this fitted spline.

Inside /potential_files directory you will find .ini files readable by Agama. The Milky Way potential used is the Milky Way potential from Shipp et al., 2021

Inside /scripts there is a Python file to execute the MCMC sampler and a sbatch file to submit to Slurm if desired.


Work in progress, and quite buggy! Will add in a python kernel file to add, so that the necessary packages are there when you need to run.
