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
    // Check for the correct number of inputs
    if(argc != 4) {
        printf("Too many or too few arguments, 4 required %d provided \n", argc);
        exit(1);
    }

    // Initialise variables from args
    double starta, step;
    int n;

    // Read in the input arguments
    // Initial guess for alpha
    sscanf(argv[1], "%lf", &starta);
    // Step size
    sscanf(argv[2], "%lf", &step);
    // Population size
    sscanf(argv[3], "%d", &n);



    // Initialise variables to store input arrays and misc
    double *distmat, *popmatpow, *travmat, *O, *Sij;
    double *popmat, *valmat;
    double error;
    double alpha;

    // Allocate memory for the arrays
    distmat = (double *) malloc(n * n * sizeof(double));
    valmat =  (double *) malloc(n * n * sizeof(double));;
    popmat = (double *) malloc(n * sizeof(double));
    popmatpow = (double *) malloc(n * sizeof(double));
    O = (double *) malloc(n * sizeof(double));
    travmat = (double *) malloc(n * n * sizeof(double));
    Sij = (double *) malloc(n * n * sizeof(double));

    // Read in the input data for the distances between locations, the flow data and the population size
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

    // Set alpha to the start value
    alpha = starta;

    //double itime0 = omp_get_wtime();

    // Evaluate Sij
    #pragma omp parallel for
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            int iter = i * n + j;
            double distij = distmat[iter];
            double Sijval = 0;
            if( i != j ) {
                // Sum pops closer than d_ij
                for(int k = 0; k < n; k++) {
                    if(distij >= distmat[i * n + k]) {
                        Sijval += popmat[k];
                    }
                }
            }
            Sij[iter] = Sijval;
        }
    }

    // Set variable to store total popsize
    double totalpop = 0;

    // Find total popsize
    #pragma omp parallel for reduction(+: totalpop)
    for(int i = 0; i < n; i++) {
        totalpop += popmat[i];
    }
    
    // printf("Time taken for initial section = %g \n", omp_get_wtime() - itime0);

    // Initialise variables to store the differences between iterations
    double difference = INFINITY;
    double olderror = INFINITY;

    // Initialise variables to store hessian and gradient
    double hessian = 0;
    double grad = 0;


    // Iterate over values until consecutive results are close enough
    while(difference > 1e-15) {

        // double time0 = omp_get_wtime();

        // Reset the error
        error = 0;

        // Reset the hessian and the gradient
        hessian = 0;
        grad = 0;

        // Evaluate the denominator (invert to reduce divisions)
        double denom = 1 / (1 - exp(-1.0 * alpha * totalpop));

        // Calculate the error and store residual
        #pragma omp parallel for reduction(+: error)
        for(int i = 0; i < n; i++) {
            // Store inv O_i
            double inv = 1 / O[i];
            double errorval = 0;
            for(int j = 0; j < n; j++) {
                if (i != j) {
                    int iter = i * n + j;
                    // Calculate the residuals at step
                    double val = (travmat[iter] * inv) - ((exp(-1 * alpha * (Sij[iter] - popmat[j])) - exp(-1 * alpha * Sij[iter])) * denom);
                    // Store residual for partial derivative calcs
                    valmat[iter] = val;
                    // Add square of residuals to the error
                    errorval += val * val;
                }
            }
            error += errorval;
        }

        // printf("alpha = %g, error = %g, time taken = %g\n", alpha, error, omp_get_wtime() - time0);

       // Evaluate partial derivatives
       #pragma omp parallel for reduction(+: hessian) reduction(+: grad)
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                if(j != i) {
                    // Initialise useful variables
                    int iter = i * n + j;
                    double fg;
                    double dfg;
                    double dfg2;
                    double dh;
                    double dh2;
                    double h;

                    long double dmain;
                    long double dmain2;

                    // Evaluate components of partial derivs
                    h = 1.0 - exp(-1 * alpha * totalpop);
                    fg = exp(-1.0 * alpha  * (Sij[iter] - popmat[j])) - exp(-1.0 * alpha * Sij[iter]);
                    dfg = -1.0 *(Sij[iter] - popmat[j]) * exp(-1.0 * alpha  * (Sij[iter] - popmat[j])) + Sij[iter] * exp(-1.0 * alpha * Sij[iter]);
                    dfg2 = (Sij[iter] - popmat[j]) * (Sij[iter] - popmat[j]) *  exp(-1.0 * alpha  * (Sij[iter] - popmat[j])) - Sij[iter] * Sij[iter] * exp(-1.0 * alpha * Sij[iter]);
                    dh = totalpop * exp(-1.0 * alpha * totalpop);
                    dh2 = -1.0 * totalpop * dh;

                    // Evaluate partial derivs
                    dmain = (dfg * h - fg * dh) /(h * h);
                    dmain2 = (dfg2 * h - fg * dh2) /(h * h * h * h);

                    // Evaluate hessian and grad using chain rule and product rule
                    hessian += 2 * (dmain * dmain + valmat[iter] *  dmain2);
                    grad += 2 * valmat[iter] * dmain;
                }
            }
        }

        // Update errors
        difference = fabs(error - olderror);
        olderror = error;

        // Print current results
        printf("current: \n alpha: %g, error: %g, difference: %g\n", alpha, error, difference);

        // Update alpha for the next step
        alpha = alpha + step * grad / hessian;

    }

    // Free allocated memory from arrays
    free(distmat);
    free(popmat);
    free(travmat);
    free(popmatpow);
    free(O);
    free(Sij);


    return 0;
}