# CCF-Segmentation
Implementation of the method decribed in the paper. Jupyter noteboooks offer a step by step guide to reproduce the results and process any kind of stationary stochastic signal.

## Contents
- [Introduction](#introduction)
- [Repository contents](#Repository-contents)
- [Citing this package](#citing-this-package)
- [Acknowledgements](#acknowledgements)


## Introduction
The current package aims at illustrating the method developed in the GJI paper and additionally to guide the user in a step by step application on real data. The method could be used in the future for extraterrestrial application if the seismic hum is supposed to be excited. 

## Repository contents

The scripts in this repository are described below


- The core routines can be found in `libCombCC.py`. It contains the class `CombCC` which is developped to support both cross and auto-correlation. The expansion to cross-correlation is currently being tested and the repository shall be updated in the future. The class currently takes a trace as input and runs a data selection based on amplitudes thresholds for specified frequency bands. The calculation are fully vectorized and make used of the very effective pyfftw python library to compute the fft of the segments. For exemple, auto-correlation of 6 months data at a sampling rate of 10s (~ 2 million data points) takes less than 3 seconds.
- A step by step illustration for synthetic data is proposed in `Illustration.ipynb`.
- An example of processing on a real data set of single station hum measurement is set up in  `ExampleStationBFO.ipynb`
- The figures for the inversion results can be reproduced using the `numpy` files together with the notebook `Inversion.ipynb`.

## Citing this package
If you make use of this code or the method in a different implementation, please acknowledge our previous work ! You cite the following paper:

[Dubois-Dognon, B., Nishida, K.,(20XX)](URL) "Reconstruction of the autocorrelation function from segmented data and its application to the Earth’s seismic hum", Geophysical Journal International, ggacXXX, doi:?.

You can cite in the following way: "We used the auto-correlation segmentation method to process the data, which is described in Dubois-Dognon and Nishdia (2025)."
This repository is associated with a Zenodo DOI :10.5281/zenodo.15591078. 

## Acknowledgements

This package was developed at the Earthquake Research Insitute in Tokyo by Balthazar Dubois-Dognon and Kiwamu Nishida. 
This project has been founded by the JSPS KAKENHI Grant Number 24K07170.
