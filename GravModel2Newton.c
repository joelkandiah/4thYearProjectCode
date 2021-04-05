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
    // Check we have the correct number of input arguments
    if(argc != 5) {
        printf("Too many or too few arguments, 5 required %d provided \n", argc);
        exit(1);
    }

    // Initialise the input variables
    double alpha, beta, step;
    int n;

    // Read in the args passed to the function
    // Initial guess for alpha
    sscanf(argv[1], "%lf", &alpha);
    // Initial guess for beta
    sscanf(argv[2], "%lf", &beta);
    // Step size between iterations
    sscanf(argv[3], "%lf", &step);
    // Number of locations
    sscanf(argv[4], "%d", &n);

    // Initialise variables to store input data and components of partial derivatives
    double *distmat, *popmatpow, *travmat, *O;
    double *term1, *term2, *term3, *term4, *term5, *term6, *term7;
    double *valmat;
    double *popmat;
    double error;

    // Allocate memory for the arrays
    distmat = (double *) malloc(n * n * sizeof(double));
    popmat = (double *) malloc(n * sizeof(double));
    popmatpow = (double *) malloc(n * sizeof(double));
    O = (double *) malloc(n * sizeof(double));
    travmat = (double *) malloc(n * n * sizeof(double));
    valmat = (double *) malloc(n * n * sizeof(double));
    term1 = (double *) malloc(n * n * sizeof(double));
    term2 = (double *) malloc(n * n * sizeof(double));
    term3 = (double *) malloc(n * n * sizeof(double));
    term4 = (double *) malloc(n * n * sizeof(double));
    term5 = (double *) malloc(n * n * sizeof(double));
    term6 = (double *) malloc(n * n * sizeof(double));
    term7 = (double *) malloc(n * n * sizeof(double));

    // Read in the data for the distances, flow sizes and population sizes
    read_custmat_double(n * n, distmat, "distmat.dat");
    read_custmat_double(n * n, travmat, "travmat.dat");
    read_custmat_double(n, popmat, "popsize.dat");

    // Calculate O_i
    #pragma omp parallel for
    for(int i = 0; i < n; i ++) {
        O[i] = 0;
        for(int j = 0; j < n; j++) {
            if(i!=j) {
                O[i] += travmat[i * n + j];
            }
        }
    }

    // Initialise values to store the old values for the error and the difference in error between iterations
    double olderror = INFINITY;

    double difference = INFINITY;

    // Initialise and allocate memory for the hessian and inverse hessian
    double *hessian = (double *) malloc(4 * sizeof(double));
    double *hessian_inv = (double *) malloc(4 * sizeof(double));

    // Iterate until two errors produce similar enough results
    while(difference > 1e-15) {

        // Evaluate the population sizes raised to a power to avoid repeated pow calculations
        #pragma omp parallel for
        for(int i = 0; i < n; i++) {
            popmatpow[i] = pow(popmat[i], alpha);
        }

        // Reset the error
        error = 0;

        // double time0 = omp_get_wtime();

        // Caclualate the error for this iteration and store the residuals for use in the partial derivatives
        #pragma omp parallel for reduction(+: error)
        for(int i = 0; i < n; i++) {
            double inv = 1.0 / (double) O[i];
            double denom = 0;
            double errorval = 0;
            // Evaluate the denominator terms
            for(int j = 0; j < n; j++) {
                if(i != j) denom += popmatpow[j] * pow(distmat[i * n + j], -1.0 * beta);
            }
            // Invert denominator value to avoid repeated division
            double denomval = 1.0 / (double) denom;
            // Evaluate the residuals an errors at each step
            for(int j = 0; j < n; j ++) {
                if(i != j) {
                    int iter = i * n + j;
                    // evaluate residual
                    double val = travmat[iter] * inv - (popmatpow[j] * pow(distmat[iter], (-1.0 * beta)) * denomval);
                    // Store residual
                    valmat[iter] = val;
                    // add square of residual to error
                    errorval += val * val;
                }
            }
            // add the i terms to the error
            error += errorval;
        }

        // Evaluate components used in the partial derivatives
        #pragma omp parallel for
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                int iter = i * n + j;
                // Evaluate terms for the partial derivatives
                term1[iter] = popmatpow[j] * pow(distmat[iter], -1.0 * beta);
                // Reset values
                term2[iter] = 0;
                term3[iter] = 0;
                term4[iter] = 0;
                term5[iter] = 0;
                term6[iter] = 0;
                term7[iter] = 0;
                // Sum the terms required for each component
                for(int k = 0; k < n; k++) {
                    double val = popmatpow[k] * pow(distmat[i * n + k], -1.0 * beta);
                    if(k != i) term2[iter] += val;
                    if(k != i) term3[iter] += val * log(popmat[k]);
                    if(k != i) term4[iter] += val * log(distmat[i * n + k]);
                    if(k != i) term5[iter] += val * log(popmat[k]) * log(popmat[k]);
                    if(k != i) term6[iter] += val * log(distmat[i * n + k]) * log(distmat[i * n + k]);
                    if(k != i) term7[iter] += val * log(popmat[k]) * log(distmat[i * n + k]);
                }
            }
        }

        //Reset hessian and grad values
        double hessiana = 0;
        double hessianb = 0;
        double hessianab = 0;

        double grada = 0;
        double gradb = 0;

        #pragma omp parallel for reduction(+: hessiana) reduction(+: hessianb) reduction(+: hessianab) reduction(+: grada) reduction(+: gradb)
        for(int i = 0; i < n; i ++) {
            for(int j = 0; j < n; j++) {
                if(j != i) {
                    // Reset partial derivatives for this term
                    double da = 0;
                    double db = 0;
                    double da2 = 0;
                    double db2 = 0;
                    double dab = 0;
                    int iter = i * n + j;

                    // Pre-calculate inverses to reduce divisions
                    double invsquare = 1 / (term2[iter] * term2[iter]);
                    double invquart = 1 / (term2[iter] * term2[iter] * term2[iter] * term2[iter]);

                    // Calculate partial derivates for the current term
                    da = (term1[iter] * term3[iter] - term1[iter] * term2[iter] * log(popmat[j])) * invsquare;
                    db = (term1[iter] * term4[iter] - term1[iter] * term2[iter] * log(distmat[iter])) * invsquare;
                    da2 = - (term1[iter] * log(popmat[j]) * log(popmat[j]) * term2[iter] * term2[iter] * term2[iter] + term1[iter] * log(popmat[j]) * term2[iter] * term2[iter] * term3[iter] - term1[iter] * log(popmat[j]) * 2.0 * term3[iter] * term2[iter] * term2[iter] - term1[iter] * log(popmat[j]) * term3[iter]  * term2[iter] * term2[iter] - term1[iter] * term5[iter] * term2[iter] * term2[iter] + term1[iter] * term3[iter] * 2.0 * term3[iter] * term2[iter]) * invquart;
                    db2 = - (term1[iter] * log(distmat[iter]) * log(distmat[iter]) * term2[iter] * term2[iter] * term2[iter] + term1[iter] * log(distmat[iter]) * term2[iter] * term2[iter] * term4[iter] - term1[iter] * log(distmat[iter]) * term4[iter] * term2[iter] * term2[iter] - term1[iter] * log(distmat[iter]) * 2.0 * term4[iter] * term2[iter] * term2[iter] - term1[iter] * term6[iter] * term2[iter] * term2[iter] + term1[iter] * term4[iter] * 2.0 * term4[iter] * term2[iter]) * invquart;
                    dab = - (term1[iter] * log(distmat[iter]) * log(popmat[j]) * term2[iter] * term2[iter] * term2[iter] + term1[iter] * log(popmat[j]) * term2[iter] * term2[iter] * term4[iter] - term1[iter] * log(popmat[j]) * 2.0 * term4[iter] * term2[iter] * term2[iter] - term1[iter] * log(distmat[iter]) * term3[iter]  * term2[iter] * term2[iter] - term1[iter] * term7[iter] * term2[iter] * term2[iter] + term1[iter] * term3[iter] * 2.0 * term4[iter] * term2[iter]) * invquart;

                    // Add terms to hessian following the chain rule and product rule
                    hessiana += 2 * ( da * da + valmat[iter] * da2);
                    hessianb += 2 * ( db * db + valmat[iter] * db2);
                    hessianab += 2 * ( db * da + valmat[iter] * dab);

                    // Add terms to the gradient following the chain rule
                    grada += 2 * valmat[iter] * da;
                    gradb += 2 * valmat[iter] * db;
                }
            }
        }

        // Set terms in hessian
        hessian[0] = hessiana;
        hessian[3] = hessianb;
        hessian[2] = hessianab;
        hessian[1] = hessianab;

        // Caclulate the scalar term to premultiply the hhessian inverse by
        double hessian_inv_pre = 1 /  (hessian[0] * hessian[3] - hessian[2] * hessian[1]);

        // Calculate the inverse of the hessian
        hessian_inv[0] = hessian[3] * hessian_inv_pre;
        hessian_inv[1] = - hessian[1] * hessian_inv_pre;
        hessian_inv[2] = - hessian[2] * hessian_inv_pre;
        hessian_inv[3] = hessian[0] * hessian_inv_pre;

        // Update the difference between the errors
        difference = fabs(error - olderror);
        // Store the error to calculate the next difference
        olderror = error;

        // Print the results for the current iteration
        printf("current: \n alpha: %g, beta: %g, error: %g, difference: %g\n", alpha, beta, error, difference);

        // Update our values of alpha and beta
        alpha = alpha - step * (grada * hessian_inv[0] - gradb * hessian_inv[1]);
        beta = beta - step * (grada * hessian_inv[2] - gradb * hessian_inv[3]);
    }

    // Print the final values of alpha and beta
    printf("Result is: %g %g\n" , alpha, beta);

    // Free the allocated memory
    free(distmat);
    free(popmat);
    free(travmat);
    free(popmatpow);
    free(O);

    return 0;
}