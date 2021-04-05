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
    // Check the number of input arguments is correct
    if(argc != 3) {
        printf("Too many or too few arguments, 3 required %d provided \n", argc);
        exit(1);
    }

    // Initialise a variable to store the number of locations and read in the value
    int n;
    sscanf(argv[1], "%d", &n);

    // Open the output file
    FILE *f;
    f = fopen(argv[2], "w");
    if (f == NULL) {

        printf("Error opening output file \n");
        exit(1);

    }

    // Initialise variables to store the input arrays and the errors
    double *distmat, *travmat, *O, *Sij;
    double *popmat;
    double error;

    // Allocate memory for arrays
    distmat = (double *) malloc(n * n * sizeof(double));
    popmat = (double *) malloc(n * sizeof(double));
    O = (double *) malloc(n * sizeof(double));
    travmat = (double *) malloc(n * n * sizeof(double));
    Sij = (double *) malloc(n * n * sizeof(double));

    // Read in the input data for the distances the flow sizes and population sizes
    read_custmat_double(n * n, distmat, "distmat.dat");
    read_custmat_double(n * n, travmat, "travmat.dat");
    read_custmat_double(n, popmat, "popsize.dat");

    //Calculate O_i
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

    // Calculate the values of S_ij
    #pragma omp parallel for
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            int iter = i * n + j;
            double distij = distmat[iter];
            // Reset the temp value of Sij for this iter
            double Sijval = 0;
            if( i != j ) {
                for(int k = 0; k < n; k++) {
                    // Add the popsize of k to sij if d_ik <= d_ij
                    if(distij >= distmat[i * n + k] ) {
                        Sijval += popmat[k];
                    }
                }
            }
            // Set the value of Sij from this iteration
            Sij[iter] = Sijval;
        }
    }

    // printf("Time taken for initial section = %g \n", omp_get_wtime() - itime0);

    // Initialise the SSI to 0
    error = 0;

    // Calculate the SSI
    #pragma omp parallel for reduction(+: error)
    for(int i = 0; i < n; i++) {
        // Reset the temporary SSI for this iteration
        double errorval = 0;
        for(int j = 0; j < n; j++) {
            if(i != j) {
                int iter = i * n + j;
                // Calculate the predicted flow for this iteration
                double val = O[i] * (popmat[i] * popmat[j] / (Sij[iter] * (Sij[iter] - popmat[j])));
                // Add the SSI for this iteration to the temporary SSI
                errorval += 2 * fmin(val, travmat[iter]) / (val + travmat[iter]);
            }
        }
        // Add the SSI for this iter to the total SSI
        error += errorval;
    }

    // printf("error = %g, time taken = %g\n", error, omp_get_wtime() - itime0);

    // Save the SSI to the output file
    fprintf(f, "%g", error / (n * (n - 1)));

    // Free all allocated memory
    free(distmat);
    free(popmat);
    free(travmat);
    free(O);
    free(Sij);

    // Close the output file
    fclose(f);

    return 0;
}