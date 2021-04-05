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

    // Initialise the variables
    // The values determining the iterations over alpha
    double starta, enda, stepa;
    // The number of locations
    int n;

    // Read in the inputs
    sscanf(argv[1], "%lf", &starta);    
    sscanf(argv[2], "%lf", &enda);
    sscanf(argv[3], "%lf", &stepa);
    sscanf(argv[4], "%d", &n);

    // Open the file to write the data to
    FILE *f;
    f = fopen(argv[5], "w");
    if (f == NULL) {

        printf("Error opening output file \n");
        exit(1);

    }

    // Initialise the arrays of the input data and misc variables
    double *distmat, *popmatpow, *travmat, *O, *Sij;
    double *popmat;
    double error;
    double alpha;

    // Allocate the memory for each array
    distmat = (double *) malloc(n * n * sizeof(double));
    popmat = (double *) malloc(n * sizeof(double));
    popmatpow = (double *) malloc(n * sizeof(double));
    O = (double *) malloc(n * sizeof(double));
    travmat = (double *) malloc(n * n * sizeof(double));
    Sij = (double *) malloc(n * n * sizeof(double));

    // Read in the data for the distances, the flows and the population sizes
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

    // Set alpha to the starting values
    alpha = starta;

    // Variable to check if on the final iteration
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
                for(int k = 0; k < n; k++) {
                    if(distij >= distmat[i * n + k]) {
                        Sijval += popmat[k];
                    }
                }
            }
            Sij[iter] = Sijval;
        }
    }

    // Initialise variable to store total population size
    double totalpop = 0;

    // Calculate the total population size
    #pragma omp parallel for reduction(+: totalpop)
    for(int i = 0; i < n; i++) {
        totalpop += popmat[i];
    }
    
    // Loop over the values of alpha
    while(alpha <= enda) {

        //double time0 = omp_get_wtime();

        // Reset the error
        error = 0;

        // Precalcilate the denominator (and invert)
        double denom = 1 / (1 - exp(-1.0 * alpha * totalpop));

        #pragma omp parallel for reduction(+: error)
        for(int i = 0; i < n; i++) {
            // Find the inverse of O_i
            double inv = 1 / O[i];
            double errorval = 0;
            for(int j = 0; j < n; j++) {
                if(j != i) {
                    int iter = i * n + j;
                    // Calculate the residual at this step
                    double val = (travmat[iter] * inv) - ((exp(-1 * alpha * (Sij[iter] - popmat[j])) - exp(-1 * alpha * Sij[iter])) * denom);
                    // Add the square of the residual to the error
                    errorval += val * val;
                }
            }
            // Add the error from the i terms to the total error
            error += errorval;
        }

        // Save the value of alpha and the current error
        fprintf(f, "%g %g \n", alpha, error);

        // Check if the next loop is the last iteration
        if(alpha + stepa < enda) {
            alpha += stepa;
        } else if(!finala) {
            alpha = enda;
            finala = 1;
        } else{
            break;
        }

    }

    // printf("Total time taken was %g\n", omp_get_wtime() - itime0);

    // Free the memory for the arrays
    free(distmat);
    free(popmat);
    free(travmat);
    free(popmatpow);
    free(O);
    free(Sij);

    // Close the files
    fclose(f);

    return 0;
}