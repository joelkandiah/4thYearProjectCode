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
    // Check enough arguments have been provided
    if(argc != 6) {
        printf("Too many or too few arguments, 6 required %d provided \n", argc);
        exit(1);
    }

    // Initialise variables to store input args
    double starta, enda, stepa;
    int n;

    // Read in input args
    // First value for alpha
    sscanf(argv[1], "%lf", &starta);    
    // Last value for alpha
    sscanf(argv[2], "%lf", &enda);
    // Step size between alphas
    sscanf(argv[3], "%lf", &stepa);
    // Total number of locations
    sscanf(argv[4], "%d", &n);

    // Open the output file
    FILE *f;
    f = fopen(argv[5], "w");
    if (f == NULL) {

        printf("Error opening output file \n");
        exit(1);

    }

    // Initialise arrays for input data and misc variables
    double *distmat, *popmatpow, *travmat, *O, *Sij;
    double *popmat;
    double error;
    double alpha;

    // Allocate memory for each array
    distmat = (double *) malloc(n * n * sizeof(double));
    popmat = (double *) malloc(n * sizeof(double));
    popmatpow = (double *) malloc(n * sizeof(double));
    O = (double *) malloc(n * sizeof(double));
    travmat = (double *) malloc(n * n * sizeof(double));
    Sij = (double *) malloc(n * n * sizeof(double));

    // Read in the input arrays for the distances, the flow sizes and the population sizes
    read_custmat_double(n * n, distmat, "distmat.dat");
    read_custmat_double(n * n, travmat, "travmat.dat");
    read_custmat_double(n, popmat, "popsize.dat");

    // Evaluate the O_i values
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

    // Initialise value to check if we are on the final iteration
    int finala = 0;

    //double itime0 = omp_get_wtime();

    // Evaluate Sij
    #pragma omp parallel for
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            int iter = i * n + j;
            double distij = distmat[iter];
            double Sijval = 0;
            if( i != j ) {
                // add populations that are closer the location j
                for(int k = 0; k < n; k++) {
                    if(distij >= distmat[i * n + k]) {
                        Sijval += popmat[k];
                    }
                }
            }
            Sij[iter] = Sijval;
        }
    }

    // Initialise variable to store population size
    double totalpop = 0;

    // Calculate the total population size
    #pragma omp parallel for reduction(+: totalpop)
    for(int i = 0; i < n; i++) {
        totalpop += popmat[i];
    }
    
    // printf("Time taken for initial section = %g \n", omp_get_wtime() - itime0);

    // Loop over values of alpha
    while(alpha <= enda) {

        //double time0 = omp_get_wtime();

        // Reset the error variable
        error = 0;

        // Precalculate the denominator term and invert to reduce division operations
        double denom = 1.0 / (1 - exp(-1.0 * alpha * totalpop));

        // Evaluate the SSI
        #pragma omp parallel for reduction(+: error)
        for(int i = 0; i < n; i++) {
            double errorval = 0;
            for(int j = 0; j < n; j++) {
                if (i != j) {
                    int iter = i * n + j;
                    // Evaluate the predicted flow for this path
                    double val = O[i] * ((exp(-1 * alpha * (Sij[iter] - popmat[j])) - exp(-1 * alpha * Sij[iter])) * denom);
                    // Add values to the total SSI
                    errorval += 2 * fmin(val, travmat[iter]) / (val + travmat[iter]);
                }
            }
            // add the SSI from the i flows
            error += errorval;
        }

        // printf("alpha = %g, error = %g, time taken = %g\n", alpha, error, omp_get_wtime() - time0);

        // Save the  values from the current iteration
        fprintf(f, "%g %g\n", alpha, error / (n * (n-1)));

        // Check if the final loop has been reached
        if(alpha + stepa < enda) {
            alpha += stepa;
        } else if(!finala) {
            alpha = enda;
            finala = 1;
        } else{
            break;
        }

    }

    //printf("Total time taken was %g\n", omp_get_wtime() - itime0);

    // Free the memory from the arrays
    free(distmat);
    free(popmat);
    free(travmat);
    free(popmatpow);
    free(O);
    free(Sij);

    // Close the output file
    fclose(f);

    return 0;
}