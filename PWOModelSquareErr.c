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
    // Check that we have the correnct number of input args
    if(argc != 3) {
        printf("Too many or too few arguments, 3 required %d provided \n", argc);
        exit(1);
    }

    // Initialise and read in the number of locations
    int n;
    sscanf(argv[1], "%d", &n);

    // Open the output file
    FILE *f;
    f = fopen(argv[2], "w");
    if (f == NULL) {

        printf("Error opening output file \n");
        exit(1);

    }

    // Initialise the input data arrays
    double *distmat, *travmat, *O, *Sij;
    double *popmat;

    // Initialise the error
    double error;

    // Allocate the memory for all of the arrays
    distmat = (double *) malloc(n * n * sizeof(double));
    popmat = (double *) malloc(n * sizeof(double));
    O = (double *) malloc(n * sizeof(double));
    travmat = (double *) malloc(n * n * sizeof(double));
    Sij = (double *) malloc(n * n * sizeof(double));

    // Read in the input data for the distances, the flow sizes and the population sizes
    read_custmat_double(n * n, distmat, "distmat.dat");
    read_custmat_double(n * n, travmat, "travmat.dat");
    read_custmat_double(n, popmat, "popsize.dat");

    // Evaluate O_i
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

    // Calculate S_ij
    #pragma omp parallel for
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            int iter = i * n + j;
            double distij = distmat[iter];
            double Sijval = 0;
            if( i != j ) {
                for(int k = 0; k < n; k++) {
                    // Add populations if nearer to i than  j and not equal to j
                    if(distij >= distmat[k * n + i] && j != k) {
                        Sijval += popmat[k];
                    }
                }
            }
            // Invert S_ij values
            Sij[iter] =  1.0 / Sijval;
        }
    }

    // Initialiise variable to store total popsize
    double totalpop = 0;

    // Find total population size
    #pragma omp parallel for reduction(+: totalpop)
    for(int i = 0; i < n; i++) {
        totalpop += popmat[i];
    }

    // Find inverse of total pop
    double invM = 1.0 / totalpop;

    // Set error to 0
    error = 0;

    // find total error
    #pragma omp parallel for reduction(+: error)
    for(int i = 0; i < n; i++) {
        // Precalulate inv O_i
        double inv = 1 / O[i];
        // Reset temporary error
        double errorval = 0;
        for(int j = 0; j < n; j++) {
            double denom = 0;
            if(i != j) {
                // Calculate denominator
                for(int k = 0; k < n; k ++) {
                    if(i!=k) {
                        denom += popmat[k] * (Sij[i * n + k] - invM);
                    }
                }
                int iter = i * n + j;
                // Calculate residual
                double val = (travmat[iter] * inv) - (popmat[j] * (Sij[iter] - invM) / denom);
                // Add square of residual to error
                errorval += val * val; 
            }
        }
        // Add error from i to total error
        error += errorval;
    }

    // printf("error = %g, time taken = %g\n", error, omp_get_wtime() - itime0);

    // Save result to output
    fprintf(f, "%g", error);

    // free allocated memory
    free(distmat);
    free(popmat);
    free(travmat);
    free(O);
    free(Sij);

    // Close output file
    fclose(f);

    return 0;
}