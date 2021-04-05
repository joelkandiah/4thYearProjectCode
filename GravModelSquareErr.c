// Set libraries to include
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Read a file of doubles of size MAT_SIZE to an array of size MAT_SIZE
void read_double_file(FILE *infile, int MAT_SIZE, double *MAT) {
    for(int i = 0; i < MAT_SIZE; i++) {
        fscanf(infile, "%lf", &MAT[i]);
        // printf("Mat[%d], %g\n", i,  MAT[i]);
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
    if(argc != 9) {
        printf("Too many or too few arguments, 9 required %d provided \n", argc);
        exit(1);
    }

    // Initialise variables to be read in which specify the range ogf alpha and beta test values
    double starta, enda, stepa, startb, endb, stepb;
    // The number of populations
    int n;

    // Read in the input args
    sscanf(argv[1], "%lf", &starta);    
    sscanf(argv[2], "%lf", &enda);
    sscanf(argv[3], "%lf", &stepa);
    sscanf(argv[4], "%lf", &startb);
    sscanf(argv[5], "%lf", &endb);
    sscanf(argv[6], "%lf", &stepb);
    sscanf(argv[7], "%d", &n);


    // Open the output file
    FILE *f;
    f = fopen(argv[8], "w");
    if (f == NULL) {

        printf("Error opening output file \n");
        exit(1);

    }

    // Initialise the data arrays
    double *distmat, *popmatpow, *travmat, *O;
    double *popmat;

    // Initialise the output variables
    double error;
    double alpha, beta;

    // Allocate memory for the arrays containing the input data
    distmat = (double *) malloc(n * n * sizeof(double));
    popmat = (double *) malloc(n * sizeof(double));
    popmatpow = (double *) malloc(n * sizeof(double));
    O = (double *) malloc(n * sizeof(double));
    travmat = (double *) malloc(n * n * sizeof(double));

    // Read in the arrays storing the distances, flow sizes and population sizes
    read_custmat_double(n * n, distmat, "distmat.dat");
    read_custmat_double(n * n, travmat, "travmat.dat");
    read_custmat_double(n, popmat, "popsize.dat");

    // Calculate O_i for all locations
    #pragma omp parallel for
    for(int i = 0; i < n; i ++) {
        O[i] = 0;
        for(int j = 0; j < n; j++) {
            if(i!=j) {
                O[i] += travmat[i * n + j];
            }
        }
    }

    // Set alpha to the first value to check
    alpha = starta;

    // Initialise variable to note whether this is the final loop
    int finala = 0;

    // Loop until all values of alpha have been checked
    while(alpha <= enda) {

        // Set beta to the first value to check
        beta = startb;

        // Initialise variable to note whether this is the final loop
        int finalb = 0;

        // Calculate the population size terms to minimise pow operations
        #pragma omp parallel for
        for(int i = 0; i < n; i++) {
            popmatpow[i] = pow(popmat[i], alpha);
        }

        // Loop until all values of beta have been checked
        while(beta <= endb) {

            // Reset error value
            error = 0;


            // double time0 = omp_get_wtime();

            // Calculate the error
            #pragma omp parallel for reduction(+: error)
            for(int i = 0; i < n; i++) {
                double inv = 1.0 / (double) O[i];
                double denom = 0;
                double errorval = 0;
                // Precalculate the denominator terms to avoid repeat calculations
                for(int j = 0; j < n; j++) {
                    if(i != j) denom += popmatpow[j] * pow(distmat[i * n + j], -1.0 * beta);
                }
                // Calulate inverse once
                double denomval = 1.0 / (double) denom;
                for(int j = 0; j < n; j ++) {
                    if(i != j) {
                        // Evaluate the error for this pair of ij
                        int iter = i * n + j;
                        // val is the residual
                        double val = travmat[iter] * inv - (popmatpow[j] * pow(distmat[iter], (-1.0 * beta)) * denomval);
                        // add the square of the residual
                        errorval += val * val;
                    }
                }
                // Sum the error for the i terms to the total error
                error += errorval;
            }
            
            
            // printf("alpha: %g, beta: %g, error: %g\n", alpha, beta, error);
            // printf("Time taken for alpha and beta %lf\n", omp_get_wtime() - time0);

            // Save the error for this alpha and beta to the file
            fprintf(f, "%g %g %g\n", alpha, beta, error);

            // Check if the next loop is the final loop
            if(beta + stepb < endb) {
                beta += stepb;
            } else if(!finalb) {
                beta = endb;
                finalb = 1;
            } else{
                break;
            }

        }

        // Check if the next loop is the final loop
        if(alpha + stepa < enda) {
            alpha += stepa;
        } else if(!finala) {
            alpha = enda;
            finala = 1;
        } else{
            break;
        }

    }

    // Free allocated memory
    free(distmat);
    free(popmat);
    free(travmat);
    free(popmatpow);
    free(O);

    // Close the file
    fclose(f);

    return 0;
}