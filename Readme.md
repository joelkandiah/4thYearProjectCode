# SEIR transmission models using various movement models

I have provide the code created for a research project into: "An analysis of metapopulation models with varying movement methods in epidemic modelling for COVID-19".

The code provided are for a metapopulation SEIR models which use one of four movement model types:
- Gravity (Grav) Model
- Intervening Opportunities (IO) Model
- Radiation (Rad) Model
- Population weighted Opportunities Model
Each of these models have the code provided in the form (Abbreviated Name)Model.c

(See https://www.nature.com/articles/s41598-019-46026-w for more details on the movement models)

Additionally code is provided to calculate the residual square error and Sorenson Similarity Index (SSI) for each movement model using the input data over a range of input parameters.

For the gravity model with 1 parameter, gravity model with 2 parameters and the radiation model there are implmentations of the newton method that minimise over the residual squared errror for the input parameters.

## Compilation

The c code requires the use of omp for parallelism. Outside of this only standard libraries are required. I compiled the code using the following line (using gcc on zsh in ubuntu 20.04):

``` bash

gcc -g main.c -o a.out -lm -fopenmp

```

## Input parameters for the epidemic models

A call to the compiled code should look like:

```
./a.out (optional parameters model specific args) (starting location) (output file 1) (output file 2)

```
The model specific args are as follows:
- Gravity model: alpha beta
- Intervening Opportunities model: alpha
The other two model types have no additional arguments.

The models all require a supplementary set of input files. These are the following:

### input.txt

This file contains 8 parameters, each on a new line with no other delimiters:
- Number of locations
- Calculation Step Size
- Maxtime for the simulations
- Print intervals
- Number of age classes
- Mean length of exposed period
- Mean Length of infectious period
- Proportion of infected individuals that die due to the disease

### aging.dat

This file contains the rate of movement out of each each class (youngest to eldest). The final entry in this file should be a 0. Each new value should be on a new line with no other delimiters.

### beta_time.dat

This file contains a list of the values of beta at each whole value timestep (i.e. each day). Each line should contain a new value and there should be no other delimiters between entries. The length of this file should be the same as the number of days the model runs for plus one.

### distmat.dat

This file contains a list of the values of the distances between the locations. Each line should contain a new value and there should be no other delimiters between entries. The values should be ordered such that we have all the entries for one row followed by the next row where our values come from a matrix with size n by n where n is the number of locations and the (i,j)th entry is the distance from i to j.

### doubledata.dat
This file contains a list of the values for each location of the birth rate followed by the death rate followed by the returning movement rate. Each line should contain a new value and there should be no other delimiters between entries. The values should be ordered such that we have all the entries for one row followed by the next row where our values come from a matrix with size n by n where n isthe number of locations and the (i,j)th entry is rate for population ij (living in i from j).

### SEEIR
This file contains a list of the values for each population group in each location. In other words S[i,j] E1[i,j] E2[i,j] E3[i,j] I[i, j] R[i,j]. Each line should contain the values in the format above and there should be no other delimiters between entries. The values should be ordered such that we have all the entries for one row followed by the next row where our values come from the initial sizes of the populations with the subscript _ij as specified in our model.

### transport_mod.dat

This file contains a list of the values of the proportion of traffic on the roads at each whole value timestep (i.e. each day). Each line should contain a new value and there should be no other delimiters between entries. The length of this file should be the same as the number of days the model runs for plus one.

### travmat.dat

This file contains a list of the values of the number of individuals that travel between two locations locations. Each line should contain a new value and there should be no other delimiters between entries. The values should be ordered such that we have all the entries for one row followed by the next row where our values come from a matrix with size n by n where n is the number of locations and the (i,j)th entry is the number of individuals that travelled from i to j.

## Outputs for the epidemic models

The first outputted results are printed to a file for each of these models with rows as follows:

time S E I R

Where S E I and R are the total population sizes of each group across all spatial regions.

The second set of results are printed to a different file at the times 80, 200 and 324 (these times are hardcoded and must be modified in the code for the specific model) with rows as follows:

time i S E I R

Where S E I and R are the population sizes for each group currently in the location i.


## Input parameters for the error calculations and SSI calculations

A call to the compiled code should look like:

```
./a.out (optional parameters model specific args) (total number of locations) (output file)
```
The model specific args are as follows:
- Gravity model: first_alpha last_alpha step_alpha first_beta last_beta step_beta
- Intervening Opportunities model: first_alpha last_alpha step_alpha
The other two model types have no additional arguments.

The step_alpha/beta argument is the step size between iterations to calculate the error/SSI at (alpha and beta are looped over independently). The first_alpha/beta and last_alpha/beta arguments mark the values to start and end the calculation of the errors/SSI at.

The models all require a supplementary set of input files. These are the following:

### distmat.dat

This file contains a list of the values of the distances between the locations. Each line should contain a new value and there should be no other delimiters between entries. The values should be ordered such that we have all the entries for one row followed by the next row where our values come from a matrix with size n by n where n is the number of locations and the (i,j)th entry is the distance from i to j.

### popsize.dat

This file contains a list of the values of the population sizes for each location. Each line should contain a new value and there should be no other delimiters between entries. The values should be ordered such that we have all the entries for one row followed by the next row where our values come from an array with size n where n is the number of locations and the i-th entry is the population size at location i.

### travmat.dat

This file contains a list of the values of the number of individuals that travel between two locations locations. Each line should contain a new value and there should be no other delimiters between entries. The values should be ordered such that we have all the entries for one row followed by the next row where our values come from a matrix with size n by n where n is the number of locations and the (i,j)th entry is the number of individuals that travelled from i to j.

## Outputs for the error and SSI calculations

The results of the errors for each set of results are printed to a file with the following in each row:

alpha beta error

Where alpha and beta are only present for the models which use these parameters (i.e for IO model it is just: alpha error and for PWO model it is just: error)

## Input parameters for the newton method for minimising errors

A call to the compiled code should look like:

```
./a.out (optional parameters model specific args) (total number of locations) (output file)
```
The model specific args are as follows:
- Gravity model 1: beta_guess
- Gravity model 1: alpha_guess beta_guess
- Gravity model 2: alpha_guess
- Intervening Opportunities model: first_alpha last_alpha step_alpha
The other two model types have no additional arguments.

The alpha/beta_guess arguments are the users best guess for the parameters that minimise the error.

The models all require a supplementary set of input files. These are the following:

### distmat.dat

This file contains a list of the values of the distances between the locations. Each line should contain a new value and there should be no other delimiters between entries. The values should be ordered such that we have all the entries for one row followed by the next row where our values come from a matrix with size n by n where n is the number of locations and the (i,j)th entry is the distance from i to j.

### popsize.dat

This file contains a list of the values of the population sizes for each location. Each line should contain a new value and there should be no other delimiters between entries. The values should be ordered such that we have all the entries for one row followed by the next row where our values come from an array with size n where n is the number of locations and the i-th entry is the population size at location i.

### travmat.dat

This file contains a list of the values of the number of individuals that travel between two locations locations. Each line should contain a new value and there should be no other delimiters between entries. The values should be ordered such that we have all the entries for one row followed by the next row where our values come from a matrix with size n by n where n is the number of locations and the (i,j)th entry is the number of individuals that travelled from i to j.

## Outputs for the minimisation codes

The results for these programs are just printed to the terminal window. These should print the parameters, error and difference between the last two iterations for each iteration. The final parameters (when difference is smaller than 1e-15) will be printed again noting that it is the final result.