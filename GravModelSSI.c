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
    // Check enough input arguments have been provided
    if(argc != 9) {
        printf("Too many or too few arguments, 9 required %d provided \n", argc);
        exit(1);
    }

    // Initialise variables to store the specification for alpha and beta
    double starta, enda, stepa, startb, endb, stepb;
    // Initialise variable to store the number of locations
    int n;

    // Read in the arguments for the ranges of parameters to evaluate over
    // First alpha value, second alpha value and step between alpha values
    sscanf(argv[1], "%lf", &starta);    
    sscanf(argv[2], "%lf", &enda);
    sscanf(argv[3], "%lf", &stepa);
    // First beta value, second beta value and step between beta values
    sscanf(argv[4], "%lf", &startb);
    sscanf(argv[5], "%lf", &endb);
    sscanf(argv[6], "%lf", &stepb);
    // Number of locations
    sscanf(argv[7], "%d", &n);

    // Open the file to save the outputs to
    FILE *f;
    f = fopen(argv[8], "w");
    if (f == NULL) {

        printf("Error opening output file \n");
        exit(1);

    }

    // Initialise the arrays to store the input data and a few other variables
    double *distmat, *popmatpow, *travmat, *O;
    double *popmat;
    double error;
    double alpha, beta;

    // Allocate memory for the input data
    distmat = (double *) malloc(n * n * sizeof(double));
    popmat = (double *) malloc(n * sizeof(double));
    popmatpow = (double *) malloc(n * sizeof(double));
    O = (double *) malloc(n * sizeof(double));
    travmat = (double *) malloc(n * n * sizeof(double));

    // Read in the distances, flow sizes and population sizes
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
        // printf("O[%d] = %g\n", i, O[i]);
    }

    // Set alpha to the start value
    alpha = starta;

    // variable to check if on the final iteration
    int finala = 0;

    // Loop over the values of alpha
    while(alpha <= enda) {

        // Set/Reset beta to the start value
        beta = startb;

        // variable to check if on the final iteration
        int finalb = 0;

        // Pre calculate the population raised to the power to reduce computation time
        #pragma omp parallel for
        for(int i = 0; i < n; i++) {
            popmatpow[i] = pow(popmat[i], alpha);
        }

        // Loop over beta values
        while(beta <= endb) {

            // Reset the SSI
            error = 0;

            //double time0 = omp_get_wtime();

            // Calculate the SSI over each flow
            #pragma omp parallel for reduction(+: error)
            for(int i = 0; i < n; i++) {
                double denom = 0;
                double errorval = 0;
                // Evaluate the denominator
                for(int j = 0; j < n; j++) {
                    if(i != j) denom += popmatpow[j] * pow(distmat[i * n + j], -1.0 * beta);
                }
                // Invert the denominator to reduce division operations
                double denomval = 1.0 / (double) denom;

                for(int j = 0; j < n; j ++) {
                    if(i != j) {
                        int iter = i * n + j;
                        // Find the predicted flow of Tij
                        double val = O[i] * (popmatpow[j] * pow(distmat[iter], (-1.0 * beta)) * denomval);
                        // Calculate this term of the SSI i.e. for T_ij
                        errorval += 2 * fmin(val, travmat[iter]) / (val + travmat[iter]);
                    }
                }
                // Add the error for the i terms
                error += errorval;
            }
        
            // Write the results to the file for this alpha and beta
            fprintf(f, "%g %g %g\n", alpha, beta, error / (n * (n-1)));

            // Check if at the end
            if(beta + stepb < endb) {
                beta += stepb;
            } else if(!finalb) {
                beta = endb;
                finalb = 1;
            } else{
                break;
            }

        }

        // Check if at the end of the loop
        if(alpha + stepa < enda) {
            alpha += stepa;
        } else if(!finala) {
            alpha = enda;
            finala = 1;
        } else{
            break;
        }

    }

    // Free the memory
    free(distmat);
    free(popmat);
    free(travmat);
    free(popmatpow);
    free(O);

    // Close the file
    fclose(f);

    return 0;
}