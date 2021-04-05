// Set libraries to include
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Read a file of doubles of size MAT_SIZE to an array of size MAT_SIZE
void read_double_file(FILE *infile, int MAT_SIZE, double *MAT) {
    for(int i = 0; i < MAT_SIZE; i++) {
        fscanf(infile, "%lf", &MAT[i]);
    };
};

// Read a file of integers of size MAT_SIZE to an array of size MAT_SIZE
void read_int_file(FILE *infile, int MAT_SIZE, int *MAT) {
    for(int i = 0; i < MAT_SIZE; i++) {
        fscanf(infile, "%d", &MAT[i]);
    };
};

// Read a specified file into a double matrix of size MAT_SIZE
void read_custmat_double(int MAT_SIZE, double *doubmat, char *arr) {
    FILE *infile;
    if(!(infile = fopen(arr, "r"))) {
        printf("Error opening input file\n");
        exit(EXIT_FAILURE);
    }
    read_double_file(infile, MAT_SIZE, doubmat);
    fclose(infile);

}

// Read a specified file into a integer matrix of size MAT_SIZE
void read_custmat_int(int MAT_SIZE, int *intmat, char *arr) {
    FILE *infile;
    if(!(infile = fopen(arr, "r"))) {
        printf("Error opening input file\n");
        exit(EXIT_FAILURE);
    }
    read_int_file(infile, MAT_SIZE, intmat);
    fclose(infile);

}

int main(int argc, char** argv) {
    // Check that enough input arguments have been provided
    if(argc != 4) {
        printf("Too many or too few arguments, 5 required %d provided \n", argc);
        exit(1);
    }

    // Initialise variables to store input arguments
    double beta, step;
    int n;

    // Read in the input args
    // Read in best guess for beta
    sscanf(argv[1], "%lf", &beta);
    // Read in step size
    sscanf(argv[2], "%lf", &step);
    // Read in the number of locations
    sscanf(argv[3], "%d", &n);

    // Initialise arrays for input data and for partial derivatives
    double *distmat, *travmat, *O;
    double *term1, *term2, *term4, *term6;
    double *valmat;
    double *popmat;
    double error;

    // Allocate memory for the arrays
    distmat = (double *) malloc(n * n * sizeof(double));
    popmat = (double *) malloc(n * sizeof(double));
    O = (double *) malloc(n * sizeof(double));
    travmat = (double *) malloc(n * n * sizeof(double));
    valmat = (double *) malloc(n * n * sizeof(double));
    term1 = (double *) malloc(n * n * sizeof(double));
    term2 = (double *) malloc(n * n * sizeof(double));
    term4 = (double *) malloc(n * n * sizeof(double));
    term6 = (double *) malloc(n * n * sizeof(double));

    // Read in the input data for the distances, flow data and population sizes
    read_custmat_double(n * n, distmat, "distmat.dat");
    read_custmat_double(n * n, travmat, "travmat.dat");
    read_custmat_double(n, popmat, "popsize.dat");

    // Evaluate O_i for each location
    #pragma omp parallel for
    for(int i = 0; i < n; i ++) {
        O[i] = 0;
        for(int j = 0; j < n; j++) {
            if(i!=j) {
                O[i] += travmat[i * n + j];
            }
        }
    }

    // Initialise the values for the error and difference between terms at each step
    double olderror = INFINITY;

    double difference = INFINITY;

    // Initialise variables to store the value for the hessian
    double hessian, hessian_inv;


    // Run loop until the difference between the terms is small enough
    while(difference > 1e-15) {

            // Reset the error value
            error = 0;

            // double time0 = omp_get_wtime();

            // Calculate the error for the current value of beta
            #pragma omp parallel for reduction(+: error)
            for(int i = 0; i < n; i++) {
                double inv = 1.0 / (double) O[i];
                double denom = 0;
                double errorval = 0;
                // Evaluate the denominator once to reduce number of calculations
                for(int j = 0; j < n; j++) {
                    if(i != j) denom += popmat[j] * pow(distmat[i * n + j], -1.0 * beta);
                }
                // Invert the denominator value to reduce divisions
                double denomval = 1.0 / (double) denom;
                //Evaluate the error for each Tij
                for(int j = 0; j < n; j ++) {
                    if(i != j) {
                        int iter = i * n + j;
                        // evaluate the residual for the ij flow
                        double val = travmat[iter] * inv - (popmat[j] * pow(distmat[iter], (-1.0 * beta)) * denomval);
                        // Store the value for use in the partial derivatives
                        valmat[iter] = val;
                        // increase the error by the squared residual
                        errorval += val * val;
                    }
                }
                // add error from all i terms
                error += errorval;
            }

            // Evaluate components of the partial derivatives to prevent repeated calculations
            #pragma omp parallel for
            for(int i = 0; i < n; i++) {
                for(int j = 0; j < n; j++) {
                    int iter = i * n + j;
                    term1[iter] = popmat[j] * pow(distmat[iter], -1.0 * beta);
                    term2[iter] = 0;
                    term4[iter] = 0;
                    term6[iter] = 0;
                    for(int k = 0; k < n; k++) {
                        double val = popmat[k] * pow(distmat[i * n + k], -1.0 * beta);
                        if(k != i) term2[iter] += val;
                        if(k != i) term4[iter] += val * log(distmat[i * n + k]);
                        if(k != i) term6[iter] += val * log(distmat[i * n + k]) * log(distmat[i * n + k]);
                    }
                }
            }

            // Reset the value of the Hessian
            hessian = 0;

            // Reset the value of the gradient
            double grad = 0;

            // Calculate the Hessian and Gradient for the current value of beta
            #pragma omp parallel for reduction(+: hessian) reduction(+: grad)
            for(int i = 0; i < n; i ++) {
                for(int j = 0; j < n; j++) {
                    if(j != i) {
                       // reset partial deriv terms for current sum term
                       double db = 0;
                       double db2 = 0;
                       int iter = i * n + j;
                       // Evaluate the partial deriv terms  for beta using pervious components
                        double invsquare = 1 / (term2[iter] * term2[iter]);
                        double invquart = 1 / (term2[iter] * term2[iter] * term2[iter] * term2[iter]);
                        db = (term1[iter] * term4[iter] - term1[iter] * term2[iter] * log(distmat[iter])) * invsquare;
                        db2 = - (term1[iter] * log(distmat[iter]) * log(distmat[iter]) * term2[iter] * term2[iter] * term2[iter] + term1[iter] * log(distmat[iter]) * term2[iter] * term2[iter] * term4[iter] - term1[iter] * log(distmat[iter]) * term4[iter] * term2[iter] * term2[iter] - term1[iter] * log(distmat[iter]) * 2.0 * term4[iter] * term2[iter] * term2[iter] - term1[iter] * term6[iter] * term2[iter] * term2[iter] + term1[iter] * term4[iter] * 2.0 * term4[iter] * term2[iter]) * invquart;

                        // Using chain rule add the partial derivative terms to the hessian and grad terms
                        hessian += 2 * ( db * db + valmat[iter] * db2);

                        grad += 2 * valmat[iter] * db;

                    }
                }
            }

            // Invert the hessian
            hessian_inv = 1 / hessian;

            // Calculate the difference between the iterations
            difference = fabs(error - olderror);
            // Update the error
            olderror = error;

            // Print the current parameters, error and difference in error between iterations
            printf("current: \n beta: %g, error: %g, difference: %g\n", beta, error, difference);

            // Update the estimate for beta
            beta = beta + step * grad * hessian_inv;
    }

    //Print final estimate for beta
    printf("Result is: %g\n" , beta);

    // Free all of the allocated memory
    free(distmat);
    free(popmat);
    free(travmat);
    free(O);

    return 0;
}