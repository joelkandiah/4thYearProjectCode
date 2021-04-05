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
    // Check for correct number of input args
    if(argc != 3) {
        printf("Too many or too few arguments, 3 required %d provided \n", argc);
        exit(1);
    }

    // Initialise and read in the number of locations
    int n;
    sscanf(argv[1], "%d", &n);

    // Open output file
    FILE *f;
    f = fopen(argv[2], "w");
    if (f == NULL) {

        printf("Error opening output file \n");
        exit(1);

    }

    // Initialise variables for input arrays and the error
    double *distmat, *travmat, *O, *Sij;
    double *popmat;
    double error;

    // Allocate memory for arrays
    distmat = (double *) malloc(n * n * sizeof(double));
    popmat = (double *) malloc(n * sizeof(double));
    O = (double *) malloc(n * sizeof(double));
    travmat = (double *) malloc(n * n * sizeof(double));
    Sij = (double *) malloc(n * n * sizeof(double));

    // Read in the input data for the distances flow sizes and the population sizes
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


    // double itime0 = omp_get_wtime();

    // Calculate the S_ij values
    #pragma omp parallel for
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            int iter = i * n + j;
            double distij = distmat[iter];
            // Reset tmp sij value for this iteration
            double Sijval = 0;
            if( i != j ) {
                for(int k = 0; k < n; k++) {
                    // Add the population sizes for locations closer than j to i except location j
                    if(distij >= distmat[k * n + i] && j != k) {
                        Sijval += popmat[k];
                    }
                }
            }
            // Invert the Sij values for faster calculation later
            Sij[iter] =  1.0 / Sijval;
        }
    }

    // Initialise value to store total population size
    double totalpop = 0;

    // Calculate the total population size
    #pragma omp parallel for reduction(+: totalpop)
    for(int i = 0; i < n; i++) {
        totalpop += popmat[i];
    }

    // Invert the total population size
    double invM = 1.0 / totalpop;

    // Set error  (SSI) to 0 for calc steps
    error = 0;

    // Calculate the SSI
    #pragma omp parallel for reduction(+: error)
    for(int i = 0; i < n; i++) {
        // Reset the temp error for this iteration
        double errorval = 0;
        for(int j = 0; j < n; j++) {
            double denom = 0;
            if(i != j) {
                // Calculate the denominator
                for(int k = 0; k < n; k ++) {
                    if(i!=k) {
                        denom += popmat[k] * (Sij[i * n + k] - invM);
                    }
                }
                int iter = i * n + j;
                // Calcuate the flow for this iteration
                double val = O[i] * (popmat[j] * (Sij[i * n + j] - invM) / denom);
                // Calculate the SSI for this singular flow
                errorval += 2 * fmin(val, travmat[iter]) / (val + travmat[iter]);
            }
        }
        // Add the SSI value for this iteration to the total SSI
        error += errorval;
    }

    // printf("error = %g, time taken = %g\n", error / (n * (n - 1)), omp_get_wtime() - itime0);

    // Save the total SSI to the output file
    fprintf(f, "%g", error / (n * (n - 1)));

    // Free the arrays from memory
    free(distmat);
    free(popmat);
    free(travmat);
    free(O);
    free(Sij);

    // Close the file
    fclose(f);

    return 0;
}