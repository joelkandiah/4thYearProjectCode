// Set libraries to include
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Create utitlity function to exchange arrays (by pointer)
void swap_arr(double **arr1, double **arr2, double **tmp) {
    *tmp = *arr1; *arr1 = *arr2; *arr2 = *tmp;
}

// Read a file of integers of size MAT_SIZE to an array of size MAT_SIZE
void read_int_file(FILE *infile, int MAT_SIZE, int *MAT) {
    for(int i = 0; i < MAT_SIZE; i++) {
        fscanf(infile, "%d", &MAT[i]);
    };
};

// Read a file of doubles of size MAT_SIZE to an array of size MAT_SIZE
void read_double_file(FILE *infile, int MAT_SIZE, double *MAT) {
    for(int i = 0; i < MAT_SIZE; i++) {
        fscanf(infile, "%lf", &MAT[i]);
    };
};

// Read all of the variables as specified from input.txt
void read_single_input(int *n, double *dt, double *t_final, double *t_print, int *age_classes, double *exposed_period, double *infected_period, double *prob_death) {
    FILE *infile;
    if(!(infile = fopen("input.txt", "r"))) {
        printf("Error opening input single file\n");
        exit(EXIT_FAILURE);
    }
    if(8 != fscanf(infile, "%d %lf %lf %lf %d %lf %lf %lf", n, dt, t_final, t_print, age_classes, exposed_period, infected_period, prob_death)) {
        printf("Error reading parameters from file\n");
        exit(EXIT_FAILURE);
    }
    fclose(infile);
};

// Read a specified file into a double matrix of size MAT_SIZE
void read_custmat_double(int MAT_SIZE, double *doubmat, char *arr) {
    FILE *infile;
    if(!(infile = fopen(arr, "r"))) {
        printf("Error opening input double file\n");
        exit(EXIT_FAILURE);
    }
    read_double_file(infile, MAT_SIZE, doubmat);
    fclose(infile);

}

// Read a specified file into a integer matrix of size MAT_SIZE
void read_custmat_int(int MAT_SIZE, int *intmat, char *arr) {
    FILE *infile;
    if(!(infile = fopen(arr, "r"))) {
        printf("Error opening input int file\n");
        printf("%s\n", arr);
        exit(EXIT_FAILURE);
    }
    read_int_file(infile, MAT_SIZE, intmat);
    fclose(infile);

}

// Read in the SEIR initial conditions from a file
void read_SEIR(int MAT_SIZE, double *S, double *E1, double *E2, double *E3, double *I, double *R) {
    FILE *infile;
    if(!(infile = fopen("SEEEIR.dat", "r"))) {
        printf("Error opening input SEIR file\n");
        exit(EXIT_FAILURE);
    };
    for(int i = 0; i < MAT_SIZE; i++) {
        if(6 != fscanf(infile, "%lf %lf %lf %lf %lf %lf", &S[i], &E1[i], &E2[i], &E3[i], &I[i], &R[i])) {
        printf("Error reading parameters from file\n");
        exit(EXIT_FAILURE);
        }
    }
};

// Read in the parameters that may differ by location
void read_double_mats(int MAT_SIZE, double *nu, double *mu, double *r) {
    FILE *infile;
    if(!(infile = fopen("doubledata.dat", "r"))) {
        printf("Error opening input SEIR file\n");
        exit(EXIT_FAILURE);
    };
    read_double_file(infile, MAT_SIZE, nu);
    read_double_file(infile, MAT_SIZE, mu);
    read_double_file(infile, MAT_SIZE, r);
    fclose(infile);
};
// Define a function which calculates a single step of the Runge-Kutta 4 method (which can then easily be repeated)
void main_calc_steps(int n, int age_classes, double *S, double *E1, double *E2, double *E3, double *I, double *R, double *D,
                        double *S_knext, double *E1_knext, double *E2_knext, double *E3_knext, double *I_knext, double *R_knext, double *D_knext,
                        double *S_next, double *E1_next, double *E2_next, double *E3_next, double *I_next, double *R_next, double *D_next,
                        double *nu, double *mu, double beta, double *sigma, double *gamma, double *kappa, double *aging,
                        double *inf_prop, double *tmp_inf_prop, double *l, double *r, double *l_prime,
                        double *r_prime_S, double *r_prime_E1, double *r_prime_E2, double *r_prime_E3, double *r_prime_I, double *r_prime_R,
                        double *popmat, double alpha, double *O, double *distmat, double tranport_mod_val, double *Sij,
                        double dt, int final, int debug) {

        double time0 = omp_get_wtime();
        double totalpop = 0;
        // Caclulate part of the force of infection at each location (the proportion of infecteds to population size at location)
        // Additionally calculate the population size at each location and the total population
        #pragma omp parallel for reduction(+: totalpop)
        for(int i = 0; i < n; i++) {
            double tmp_value_I = 0;
            double tmp_value_N = 0; // Assume final sum will be non-zero
            for(int k = 0; k < age_classes; k ++) {
                // #pragma omp parallel for reduction(+: tmp_value_I) reduction(+: tmp_value_N)
                for(int j = 0; j < n; j++) {
                    int iter = k * n * n + i * n + j; 
                    tmp_value_I += I[iter];
                    tmp_value_N += S[iter] + E1[iter] + E2[iter] + E3[iter] + I[iter] + R[iter];
                }
            }
            tmp_inf_prop[i] = tmp_value_I / tmp_value_N;
            popmat[i] = tmp_value_N;
            totalpop += tmp_value_N;
        }

        if (debug) printf("S1: %lf\n", omp_get_wtime()-time0);

        // Pre-calculate the Sij matrix for use in our calculation of the steps for the IO model
        // We sum the populations sizes for each circle of distances between the original and final locations
        time0 = omp_get_wtime();
        #pragma omp parallel for
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                int distval = distmat[i * n + j];
                double val = 0;
                if(i != j) {
                    for(int k = 0; k < n; k++) {
                        if(distval >= distmat[i * n + k]) { 
                            val += popmat[k];
                        }
                    }
                }
                Sij[i * n + j] = val;
            }
        }
        if (debug) printf("S2: %lf\n", omp_get_wtime()-time0);

        if(debug) time0 = omp_get_wtime();

        double denomIO = 1.0 / (1 - exp(-1 * alpha * totalpop));
        // Caculate l for each grid term using the Intervening opportunities model
        #pragma omp parallel for
        for(int i = 0; i < n; i++) {
            // Pre-calculate the inverse terms for faster calculation
            double tmpinv = 1.0 / popmat[i];
            // Caclulate the IJ terms but "swap" due to difference in IO model and metapop model specs
            for(int j = 0; j < n; j++) {
                int iter = j * n + i;
                if(i != j) {
                    l[iter] = O[i] * tmpinv * (exp(-1 * alpha * (Sij[i * n + j] - popmat[i])) - exp( -1 * alpha * Sij[i * n + j])) * denomIO;
                } else {
                    l[iter] = 0;
                }
            }
        }
        if (debug) printf("S2: %lf\n", omp_get_wtime()-time0);
                            
        if (debug) time0 = omp_get_wtime();
        if (debug) printf("S3: %lf\n", omp_get_wtime()-time0);
        
        if (debug) time0 = omp_get_wtime();
        // Set the infected proportions for each matrix entry i.e. the sum(I_ij)/sum(N_ij)
        // Set the leaving rates for each group l_ij off diagonal
        // Set the returning rates for each group r_ij off diagonal
       
        for(int k = 0; k < age_classes; k++) {
            #pragma omp parallel for
            for(int i = 0; i < n; i++) {
                for(int j = 0; j < n; j++) {
                    int iter = k * n * n + i * n + j;
                    inf_prop[iter] = tmp_inf_prop[i];
                    l_prime[iter] = l[iter - (k * n * n)];
                    r_prime_S[iter] = r[iter] * S[iter];
                    r_prime_E1[iter] = r[iter] * E1[iter];
                    r_prime_E2[iter] = r[iter] * E2[iter];
                    r_prime_E3[iter] = r[iter] * E3[iter];
                    r_prime_I[iter] = r[iter] * I[iter];
                    r_prime_R[iter] = r[iter] * R[iter];
                }
            }
        }

        if (debug) printf("S4: %lf\n", omp_get_wtime()-time0);

        if (debug) time0 = omp_get_wtime();
        // Set diagonal terms for leaving and returning matrices
        for(int k = 0; k < age_classes; k++) {
            #pragma omp parallel for
            for(int i = 0; i < n; i++) {
                int diag_iter = k * n * n + i * n + i;
                double l_prime_temp = 0;
                double r_prime_S_temp = 0;
                double r_prime_E1_temp = 0;
                double r_prime_E2_temp = 0;
                double r_prime_E3_temp = 0;
                double r_prime_I_temp = 0;
                double r_prime_R_temp = 0;
                l_prime[diag_iter] = 0;
                r_prime_S[diag_iter] = 0;
                r_prime_E1[diag_iter] = 0;
                r_prime_E2[diag_iter] = 0;
                r_prime_E3[diag_iter] = 0;
                r_prime_I[diag_iter] = 0;
                r_prime_R[diag_iter] = 0;
                for(int j = 0; j < n; j++) {
                    int rev_iter = k * n * n + j * n + i;
                    l_prime_temp -= l_prime[rev_iter]; // reverse order to sum those from i moving toj
                    r_prime_S_temp -= r_prime_S[rev_iter]; // reverse order to sum those in j returning to i
                    r_prime_E1_temp -= r_prime_E1[rev_iter]; // reverse order to sum those in j returning to i
                    r_prime_E2_temp -= r_prime_E2[rev_iter]; // reverse order to sum those in j returning to i
                    r_prime_E3_temp -= r_prime_E3[rev_iter]; // reverse order to sum those in j returning to i
                    r_prime_I_temp -= r_prime_I[rev_iter]; // reverse order to sum those in j returning to i
                    r_prime_R_temp -= r_prime_R[rev_iter]; // reverse order to sum those in j returning to i
                }
                l_prime[diag_iter] = l_prime_temp;
                r_prime_S[diag_iter] = r_prime_S_temp;
                r_prime_E1[diag_iter] = r_prime_E1_temp;
                r_prime_E2[diag_iter] = r_prime_E2_temp;
                r_prime_E3[diag_iter] = r_prime_E3_temp;
                r_prime_I[diag_iter] = r_prime_I_temp;
                r_prime_R[diag_iter] = r_prime_R_temp;
            }
        }

        if (debug) printf("S5: %lf\n", omp_get_wtime()-time0);
        //! Step 2 
        if (debug) time0 = omp_get_wtime();
        // Calculate the values from this step for SEIR (and only the base age class i.e. has births)
        #pragma omp parallel for
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                int iter = i * n + j;
                S_knext[iter] = nu[iter] * (S[iter] + E1[iter] + E2[iter] + E3[iter] + I[iter] + R[iter]) - beta * S[iter] * inf_prop[iter] + tranport_mod_val * l_prime[iter] * S[j * n + j] - r_prime_S[iter] - mu[iter] * S[iter] - aging[0] * S[iter];
                E1_knext[iter] = beta * S[iter] * inf_prop[iter] -  3.0 * sigma[i] * E1[iter] + tranport_mod_val * l_prime[iter] * E1[j * n + j] - r_prime_E1[iter] - mu[iter] * E1[iter] - aging[0] * E1[iter];
                E2_knext[iter] = 3.0 * sigma[i] * E1[iter] - 3 * sigma[i] * E2[iter] + tranport_mod_val * l_prime[iter] * E2[j * n + j] - r_prime_E2[iter] - mu[iter] * E2[iter] - aging[0] * E2[iter];
                E3_knext[iter] = 3.0 * sigma[i] * E2[iter] - 3 * sigma[i] * E3[iter] + tranport_mod_val * l_prime[iter] * E3[j * n + j] - r_prime_E3[iter] - mu[iter] * E3[iter] - aging[0] * E3[iter];
                I_knext[iter] = 3.0 * sigma[i] * E3[iter] - gamma[i] * I[iter] + tranport_mod_val * l_prime[iter] * I[j * n + j] - r_prime_I[iter] - mu[iter] * I[iter] - aging[0] * I[iter] - kappa[i] * I[iter];
                R_knext[iter] = gamma[i] * I[iter] + tranport_mod_val * l_prime[iter] * R[j * n + j] - r_prime_R[iter] - mu[iter] * R[iter] - aging[0] * R[iter];
                D_knext[iter] = mu[iter] * (E1[iter] + E2[iter] + E3[iter] + I[iter] + R[iter]) + kappa[i] * I[iter];
            }
        }
        if (debug) printf("S6: %lf\n", omp_get_wtime()-time0);

        if (debug) time0 = omp_get_wtime();
        // Calculate the terms from this step for SEIR where we arent in the base age class
        for(int k = 1; k < age_classes; k++) {
            #pragma omp parallel for
            for(int i = 0; i<n; i++) {
                for(int j = 0; j < n; j++) {
                    int iter = k * n * n + i * n + j;
                    S_knext[iter] = nu[iter] * (S[iter] + E1[iter] + E2[iter] + E3[iter] + I[iter] + R[iter]) - tranport_mod_val * beta * S[iter] * inf_prop[iter] + tranport_mod_val * l_prime[iter] * S[j * n + j] - r_prime_S[iter] - mu[iter] * S[iter] - aging[k] * S[iter];
                    E1_knext[iter] = tranport_mod_val * beta * S[iter] * inf_prop[iter] -  3.0 * sigma[i] * E1[iter] + tranport_mod_val * l_prime[iter] * E1[j * n + j] - r_prime_E1[iter] - mu[iter] * E1[iter] - aging[k] * E1[iter];
                    E2_knext[iter] = 3.0 * sigma[i] * E1[iter] - 3 * sigma[i] * E2[iter] + tranport_mod_val * l_prime[iter] * E2[j * n + j] - r_prime_E2[iter] - mu[iter] * E2[iter] - aging[k] * E2[iter];
                    E3_knext[iter] = 3.0 * sigma[i] * E2[iter] - 3 * sigma[i] * E3[iter] + tranport_mod_val * l_prime[iter] * E3[j * n + j] - r_prime_E3[iter] - mu[iter] * E3[iter] - aging[k] * E3[iter];
                    I_knext[iter] = 3.0 * sigma[i] * E3[iter] - gamma[i] * I[iter] + tranport_mod_val * l_prime[iter] * I[j * n + j] - r_prime_I[iter] - mu[iter] * I[iter] - aging[k] * I[iter] - kappa[i] * I[iter];
                    R_knext[iter] = gamma[i] * I[iter] + tranport_mod_val * l_prime[iter] * R[j * n + j] - r_prime_R[iter] - mu[iter] * R[iter] - aging[k] * R[iter];
                    D_knext[iter] = mu[iter] * (E1[iter] + E2[iter] + E3[iter] + I[iter] + R[iter]) + kappa[k * n + i] * I[iter];
                }
            }
        }

        if (debug) printf("S7: %lf\n", omp_get_wtime()-time0);

        // Calculate the values over the timestep if this is not the last step

        if(!final) {
            // Set S, I and N for next iteration
            #pragma omp parallel for
            for(int k = 0; k < age_classes; k++) {
                for(int i = 0; i < n; i++) {
                    for(int j = 0; j < n; j++) {
                        int iter = k * n * n + i * n + j;
                        S_next[iter] = S[iter] + dt * S_knext[iter]; 
                        E1_next[iter] = E1[iter] + dt * E1_knext[iter]; 
                        E2_next[iter] = E2[iter] + dt * E2_knext[iter]; 
                        E3_next[iter] = E3[iter] + dt * E3_knext[iter]; 
                        I_next[iter] = I[iter] + dt * I_knext[iter]; 
                        R_next[iter] = R[iter] + dt * R_knext[iter]; 
                        D_next[iter] = D[iter] + dt * D_knext[iter];
                    }
                }
            }
        }
}

int main(int argc, char** argv) {

   
   // Check there are 5 arguments to our program
    if(argc != 5) {
        printf("Too few or too many arguments passed to program");
        exit(1);
    }

    // Read in arguments to doubles for IO parameter and read in the contact rate and infection location
    double alpha;
    int infloc;
    sscanf(argv[1],"%lf",&alpha);
    sscanf(argv[2],"%d",&infloc);


    // Check that we can open the file to write our outputs to
    FILE *f; // Pointer to open file
    f = fopen(argv[3], "w");
    if (f == NULL) {

        printf("Error opening output file \n");
        exit(1);

    }

    // Check that we can open the file to write our outputs to
    FILE *g; // Pointer to open file
    g = fopen(argv[4], "w");
    if (f == NULL) {

        printf("Error opening output file \n");
        exit(1);

    }

    //! Initialise Matrices
    
    // SEIR stores for full steps
    double *S, *E1, *E2, *E3, *I, *R, *D;
    double *S_next, *E1_next, *E2_next, *E3_next, *I_next, *R_next, *D_next;
    double *S_next_tmp, *E1_next_tmp, *E2_next_tmp, *E3_next_tmp, *I_next_tmp, *R_next_tmp, *D_next_tmp;

    // SEIR stores for individual runge-kutta steps
    double *S_k1, *E1_k1, *E2_k1, *E3_k1, *I_k1, *R_k1, *D_k1;
    double *S_k2, *E1_k2, *E2_k2, *E3_k2, *I_k2, *R_k2, *D_k2;
    double *S_k3, *E1_k3, *E2_k3, *E3_k3, *I_k3, *R_k3, *D_k3;
    double *S_k4, *E1_k4, *E2_k4, *E3_k4, *I_k4, *R_k4, *D_k4;

    //Stores for demography and movement parameters
    double *nu, *mu, *l, *r;
    double *inf_prop, *l_prime, *r_prime_S, *r_prime_E1, *r_prime_E2, *r_prime_E3, *r_prime_I, *r_prime_R;

    // Store for parameters for the movement model
    double *distmat, *popmat, *O, *transport_mod, *travmat;
    double *Sij;

    // Stores for utility arrays
    double *tmp_inf_prop;
    double *tmp_swap;

    // Matrix width/heght i.e number of population groups
    int n;
    double dt; 
    double t_final;
    double t_print;
    int age_classes;
    double exposed_period;
    double infected_period;
    double prob_death;

    // Read in inital inputs
    read_single_input(&n, &dt, &t_final, &t_print, &age_classes, &exposed_period, &infected_period, &prob_death);

    long ARR_SIZE = sizeof(double) * n * age_classes;

    double *beta = (double *) malloc(((int) t_final + 1) * sizeof(double));

    double *sigma = (double *) malloc(ARR_SIZE); // Infectious rate per town

    double *gamma = (double *) malloc(ARR_SIZE); // Recovery rate per town

    double *kappa = (double *) malloc(ARR_SIZE); // Recovery rate per town
    
    double *aging = (double *) malloc(age_classes * sizeof(double));

    // Read in the beta data that changes with time
    read_custmat_double((int) t_final + 1, beta, "beta_time.dat");

    // -------------------

    //! Allocate Initial Matrices for non-demographic and main SEIR parameters

    for(int i = 0; i < n * age_classes; i++) {

        sigma[i] = 1.0/exposed_period;
        gamma[i] = 1.0/infected_period;
        kappa[i] = prob_death;

    }

    // Read in aging data
    read_custmat_double(age_classes, aging, "aging.dat");

    // Check that number of age classes matches aging parameters
    if(aging[age_classes - 1] != 0) {
        printf("The number of age classes is inconsistent with the aging parameters inputted");
        exit(EXIT_FAILURE);
    }

    // Set avaraible to store the size of the matrices due to frequent usage in code
    long MAT_SIZE = sizeof(double) * n * n * age_classes; // Size of the matrix

    S = (double *) malloc(MAT_SIZE);
    E1 = (double *) malloc(MAT_SIZE);
    E2 = (double *) malloc(MAT_SIZE);
    E3 = (double *) malloc(MAT_SIZE);
    I = (double *) malloc(MAT_SIZE);
    R = (double *) malloc(MAT_SIZE);
    D = (double *) calloc(n * n * age_classes, sizeof(double));

    // read in the initial conditions (except extra infection or initial infection)
    read_SEIR(n * n * age_classes, S, E1, E2, E3, I, R);

    S_next = (double *) malloc(MAT_SIZE);
    E1_next = (double *) malloc(MAT_SIZE);
    E2_next = (double *) malloc(MAT_SIZE);
    E3_next = (double *) malloc(MAT_SIZE);
    I_next = (double *) malloc(MAT_SIZE);
    R_next = (double *) malloc(MAT_SIZE);
    D_next = (double *) calloc(n * n * age_classes, sizeof(double));

    S_next_tmp = (double *) malloc(MAT_SIZE);
    E1_next_tmp = (double *) malloc(MAT_SIZE);
    E2_next_tmp = (double *) malloc(MAT_SIZE);
    E3_next_tmp = (double *) malloc(MAT_SIZE);
    I_next_tmp = (double *) malloc(MAT_SIZE);
    R_next_tmp = (double *) malloc(MAT_SIZE);
    D_next_tmp = (double *) calloc(n * n * age_classes, sizeof(double));


    S_k1 = (double *) malloc(MAT_SIZE);
    E1_k1 = (double *) malloc(MAT_SIZE);
    E2_k1 = (double *) malloc(MAT_SIZE);
    E3_k1 = (double *) malloc(MAT_SIZE);
    I_k1 = (double *) malloc(MAT_SIZE);
    R_k1 = (double *) malloc(MAT_SIZE);
    D_k1 = (double *) calloc(n * n * age_classes, sizeof(double));

    S_k2 = (double *) malloc(MAT_SIZE);
    E1_k2 = (double *) malloc(MAT_SIZE);
    E2_k2 = (double *) malloc(MAT_SIZE);
    E3_k2 = (double *) malloc(MAT_SIZE);
    I_k2 = (double *) malloc(MAT_SIZE);
    R_k2 = (double *) malloc(MAT_SIZE);
    D_k2 = (double *) calloc(n * n * age_classes, sizeof(double));
    
    S_k3 = (double *) malloc(MAT_SIZE);
    E1_k3 = (double *) malloc(MAT_SIZE);
    E2_k3 = (double *) malloc(MAT_SIZE);
    E3_k3 = (double *) malloc(MAT_SIZE);
    I_k3 = (double *) malloc(MAT_SIZE);
    R_k3 = (double *) malloc(MAT_SIZE);
    D_k3 = (double *) calloc(n * n * age_classes, sizeof(double));

    S_k4 = (double *) malloc(MAT_SIZE);
    E1_k4 = (double *) malloc(MAT_SIZE);
    E2_k4 = (double *) malloc(MAT_SIZE);
    E3_k4 = (double *) malloc(MAT_SIZE);
    I_k4 = (double *) malloc(MAT_SIZE);
    R_k4 = (double *) malloc(MAT_SIZE);
    D_k4 = (double *) calloc(n * n * age_classes, sizeof(double));

    nu = (double *) malloc(MAT_SIZE);
    mu = (double *) malloc(MAT_SIZE);
    l = (double *) malloc(MAT_SIZE);
    r = (double *) malloc(MAT_SIZE);

    // Due to contents of the following parameters we allocate these arrays with custom sizes
    distmat = (double *) malloc(n * n * sizeof(double));
    popmat = (double *) malloc(n * sizeof(double));
    travmat = (double *) malloc(n * n * sizeof(double));
    Sij = (double *) malloc(n * n * sizeof(double));
    O = (double *) malloc(n * sizeof(double));

    transport_mod = (double *) malloc(t_final * sizeof(double));

    // Read in additional parameters from movement data
    read_double_mats(n * n * age_classes, nu, mu, r);
    read_custmat_double(n * n, distmat, "distmat.dat");
    read_custmat_double(n * n, travmat, "travmat.dat");
    read_custmat_double(t_final, transport_mod, "transportmod.dat");

    // Calculate O as the sum of all movement of people
    #pragma omp parallel for
    for(int i = 0; i < n; i ++) {
        O[i] = 0;
        for(int j = 0; j < n; j++) {
            if(i!=j) {
                O[i] += travmat[i * n + j];
            }
        }
    }

    // Additional matrices for calculation steps
    inf_prop = (double *) malloc(MAT_SIZE);
    l_prime = (double *) malloc(MAT_SIZE);
    r_prime_S = (double *) malloc(MAT_SIZE);
    r_prime_E1 = (double *) malloc(MAT_SIZE);
    r_prime_E2 = (double *) malloc(MAT_SIZE);
    r_prime_E3 = (double *) malloc(MAT_SIZE);
    r_prime_I = (double *) malloc(MAT_SIZE);
    r_prime_R = (double *) malloc(MAT_SIZE);

    tmp_inf_prop = (double *) malloc(sizeof(double) * n);

    //Add initial infection where specified
    if(infloc < n) {
        int iter = infloc * n + infloc;
        S[iter] -= 1;
        I[iter] += 1;
    } else {
        printf("Location for infection not in range (argument 4 to cmdline");
        exit(1);
    }

    const static int test = 0;
    
    // MEthod to sum variables and ensure we print only a summary of the results at each stage
    double Sprint = 0;
    double Eprint = 0;
    double Iprint = 0;
    double Rprint = 0;
    double Dprint = 0;
    #pragma omp parallel for reduction(+: Sprint) reduction(+: Eprint) reduction(+: Iprint) reduction(+: Rprint) reduction(+: Dprint)
    for(int k = 0; k < age_classes; k++) {
        for(int i = 0; i<n; i++) {
            for(int j = 0; j<n; j++) {
                int iter = i * n + j;
                Sprint += S[iter];
                Eprint += E1[iter] + E2[iter] + E3[iter];
                Iprint += I[iter];
                Rprint += R[iter];
                Dprint += D[iter];
            }
        }
    }
    if(!test) {
        fprintf(f, "%g,  %g, %g, %g, %g, %g\n", 0.0, Sprint, Eprint, Iprint, Rprint, Dprint);
    } else {
        fprintf(f, "%g, 0, %g, %g, %g, %g, %g, %g\n", 0.0,  S_next[0], E1_next[0], E2_next[0], E3_next[0], I_next[0], R_next[0]);
        fprintf(f, "%g, 1, %g, %g, %g, %g, %g, %g\n", 0.0,  S_next[1], E1_next[1], E2_next[1], E3_next[1], I_next[1], R_next[1]);
        fprintf(f, "%g, 2, %g, %g, %g, %g, %g, %g\n", 0.0,  S_next[2], E1_next[2], E2_next[2], E3_next[2], I_next[2], R_next[2]);
    }    

    // Initialise timers
    double t_prime = 0;
    double t_print_prime = t_print;

    double dt_prime = dt;

    // Hardcoded print regional times
    int print2times[4] = {80, 200, 324, INFINITY};

    // Initialise variable to keep track of when to print regional data
    int *print2_prime;
    print2_prime = &print2times[0];

    // initialise control variables
    int print = 0;
    int print2 = 0;
    int final = 0;
    static const int debug = 0;

    if (debug) printf("%g,  %g, %g, %g, %g\n", 0.0,  Sprint, Eprint, Iprint, Rprint);


    // Main calulation loop
    while(t_prime <= t_final - 1) {

        // Create a timer to measure the time taken for the whole iteration
        double t_whole = omp_get_wtime();

        // Set current timestep
        dt_prime = dt;

        // Check whether we need to print on this iteration and modify timestep acordingly
        if (t_prime + dt >= t_print_prime) {
            dt_prime = t_print_prime - t_prime;
            t_print_prime += t_print;
            print = 1;
        }

        t_prime += dt_prime;
        double half_dt = dt_prime / 2.0;

        // Check whether to print regional data this iteration
        if (t_prime >= (double) *print2_prime) {
            print2 = 1;
            print2_prime = print2_prime + 1;
        }

        // Check if this is the final iteration
        if(t_prime >= t_final) {
            dt_prime -= t_prime - t_final;
            t_prime = t_final;
            print = 1;
            final = 1;
        }

        // Find the transport modification to use on this calculation and store for later.
        double transport_mod_val = transport_mod[((int) ceil(t_prime)) - 1] / 100;
        double beta_val = beta[((int) ceil(t_prime)) - 1];

        //! Use RK4 method
        // X_n+1 = X_n + 1/6 * dt * (k1 + 2*X_k2 + 2*X_k3 + X_k4)
        // 0  |
        // 1/2|1/2
        // 1/2| 0  1/2
        // 1  | 0   0  1
        // ---|---------------
        //    |1/6 1/3 1/3 1/6

        //! Calculate k1 
        double time0 = omp_get_wtime();
        main_calc_steps(n, age_classes, S, E1, E2, E3, I, R, D,
                        S_k1, E1_k1, E2_k1, E3_k2, I_k1, R_k1, D_k1,
                        S_next, E1_next, E2_next, E3_next, I_next, R_next, D_next,
                        nu, mu, beta_val, sigma, gamma, kappa, aging,
                        inf_prop, tmp_inf_prop, l, r,
                        l_prime, r_prime_S, r_prime_E1, r_prime_E2, r_prime_E3, r_prime_I, r_prime_R,
                        popmat, alpha, O, distmat, transport_mod_val, Sij,
                        half_dt, 0, debug);

        if (debug) printf("Step 1: %lf\n", omp_get_wtime() - time0);

        if (debug) time0 = omp_get_wtime();

        //! Calculate k2
        main_calc_steps(n, age_classes, S_next, E1_next, E2_next, E3_next, I_next, R_next, D_next,
                        S_k2, E1_k2, E2_k2, E3_k2, I_k2, R_k2, D_k2,
                        S_next_tmp, E1_next_tmp, E2_next_tmp, E3_next_tmp, I_next_tmp, R_next_tmp, D_next_tmp,
                        nu, mu, beta_val, sigma, gamma, kappa, aging,
                        inf_prop, tmp_inf_prop, l, r,
                        l_prime, r_prime_S, r_prime_E1, r_prime_E2, r_prime_E3, r_prime_I, r_prime_R,
                        popmat, alpha, O, distmat, transport_mod_val, Sij,
                        half_dt, 0, debug);

        if (debug) printf("Step 2: %lf\n", omp_get_wtime() - time0);

        if (debug) time0 = omp_get_wtime();

        //! Calculate k3
        main_calc_steps(n, age_classes, S_next_tmp, E1_next_tmp, E2_next_tmp, E3_next_tmp, I_next_tmp, R_next_tmp, D_next_tmp,
                        S_k3, E1_k3, E2_k3, E3_k3, I_k3, R_k3, D_k3,
                        S_next, E1_next, E2_next, E3_next, I_next, R_next, D_next,
                        nu, mu, beta_val, sigma, gamma, kappa, aging,
                        inf_prop, tmp_inf_prop, l, r,
                        l_prime, r_prime_S, r_prime_E1, r_prime_E2, r_prime_E3, r_prime_I, r_prime_R,
                        popmat, alpha, O, distmat, transport_mod_val, Sij,
                        dt_prime, 0, debug);

        if (debug) printf("Step 3: %lf\n", omp_get_wtime() - time0);

        if (debug) time0 = omp_get_wtime();

        //! Calculate k4
        main_calc_steps(n, age_classes, S_next, E1_next, E2_next, E3_next, I_next, R_next, D_next,
                        S_k4, E1_k4, E2_k4, E3_k4, I_k4, R_k4, D_k4,
                        S_next_tmp, E1_next_tmp, E2_next_tmp, E3_next_tmp, I_next_tmp, R_next_tmp, D_next_tmp,
                        nu, mu, beta_val, sigma, gamma, kappa, aging,
                        inf_prop, tmp_inf_prop, l, r,
                        l_prime, r_prime_S, r_prime_E1, r_prime_E2, r_prime_E3, r_prime_I, r_prime_R,
                        popmat, alpha, O, distmat, transport_mod_val, Sij,
                        dt_prime, 1, debug);

        if (debug) printf("Step 4: %lf\n", omp_get_wtime() - time0);

        if (debug) time0 = omp_get_wtime();

        double sixth = 1.0/6.0;
        
        // Caclulate final iteration result from data found using RK4
        #pragma omp parallel for
        for(int i = 0; i<n; i++) {
            for(int j = 0; j<n; j++) {
                int iter = i * n + j;
                S_next[iter] = S[iter] + sixth * dt_prime * (S_k1[iter] + 2.0 * S_k2[iter] + 2.0 * S_k3[iter] + S_k4[iter]); 
                E1_next[iter] = E1[iter] + sixth * dt_prime * (E1_k1[iter] + 2.0 * E1_k2[iter] + 2.0 * E1_k3[iter] + E1_k4[iter]); 
                E2_next[iter] = E2[iter] + sixth * dt_prime * (E2_k1[iter] + 2.0 * E2_k2[iter] + 2.0 * E2_k3[iter] + E2_k4[iter]); 
                E3_next[iter] = E3[iter] + sixth * dt_prime * (E3_k1[iter] + 2.0 * E3_k2[iter] + 2.0 * E3_k3[iter] + E3_k4[iter]); 
                I_next[iter] = I[iter] + sixth * dt_prime * (I_k1[iter] + 2.0 * I_k2[iter] + 2.0 * I_k3[iter] + I_k4[iter]); 
                R_next[iter] = R[iter] + sixth * dt_prime * (R_k1[iter] + 2.0 * R_k2[iter] + 2.0 * R_k3[iter] + R_k4[iter]); 
                D_next[iter] = D[iter] + sixth * dt_prime * (D_k1[iter] + 2.0 * D_k2[iter] + 2.0 * D_k3[iter] + D_k4[iter]);
            }
        }

        if (debug) printf("Step 5: %lf\n", omp_get_wtime() - time0);

        if (debug) time0 = omp_get_wtime();
        // Print the results when requested to print (only of the summarised results)
        if (print) {
            Sprint = 0;
            Eprint = 0;
            Iprint = 0;
            Rprint = 0;
            Dprint = 0;
            for(int k = 0; k < age_classes; k++) {
                #pragma omp parallel for reduction(+: Sprint) reduction(+: Eprint) reduction(+: Iprint) reduction(+: Rprint) reduction(+: Dprint)
                for(int i = 0; i<n; i++) {
                    for(int j = 0; j<n; j++) {
                        int iter = i * n + j;
                        Sprint += S_next[iter];
                        Eprint += E1_next[iter] + E2_next[iter] + E3_next[iter];
                        Iprint += I_next[iter];
                        Rprint += R_next[iter];
                        Dprint += D_next[iter];
                    }
                }
            }
            if(!test) {
                fprintf(f, "%g,  %g, %g, %g, %g, %g\n", t_prime,  Sprint, Eprint, Iprint, Rprint, Dprint);
            } else {
                fprintf(f, "%g, 0, %g, %g, %g, %g, %g, %g\n", t_prime,  S_next[0], E1_next[0], E2_next[0], E3_next[0], I_next[0], R_next[0]);
                fprintf(f, "%g, 1, %g, %g, %g, %g, %g, %g\n", t_prime,  S_next[1], E1_next[1], E2_next[1], E3_next[1], I_next[1], R_next[1]);
                fprintf(f, "%g, 2, %g, %g, %g, %g, %g, %g\n", t_prime,  S_next[2], E1_next[2], E2_next[2], E3_next[2], I_next[2], R_next[2]);
            }
            if(debug) printf("%g,  %g, %g, %g, %g\n", t_prime,  Sprint, Eprint, Iprint, Rprint);
        }

        // Print the regional data
        if(print2) {
            for(int loc = 0; loc<n; loc++) {
                double S_val = 0;
                double E_val = 0;
                double I_val = 0;
                double R_val = 0;
                double D_val = 0;
                #pragma omp parallel for reduction(+: S_val) reduction(+: E_val) reduction(+: I_val) reduction(+:R_val) reduction(+: D_val)
                for(int j = 0; j<n; j++) {
                    int iter = loc * n + j;
                    S_val += S_next[iter];
                    E_val += E1_next[iter] + E2_next[iter] + E3_next[iter];
                    I_val += I_next[iter];
                    R_val += R_next[iter];
                    D_val += D_next[iter];
                }
                fprintf(g, "%g, %d, %g, %g, %g, %g, %g\n", t_prime, loc + 1,  S_val, E_val, I_val, R_val, D_val);
            }
        }

        if (debug) printf("Step 6: %lf\n", omp_get_wtime() - time0);

        if (debug) time0 = omp_get_wtime();

        // Exchange the arrays
        swap_arr(&S, &S_next, &tmp_swap);
        swap_arr(&E1, &E1_next, &tmp_swap);
        swap_arr(&E2, &E2_next, &tmp_swap);
        swap_arr(&E3, &E3_next, &tmp_swap);
        swap_arr(&I, &I_next, &tmp_swap);
        swap_arr(&R, &R_next, &tmp_swap);
        swap_arr(&D, &D_next, &tmp_swap);

        if (debug) printf("Step 7: %lf\n", omp_get_wtime() - time0);


        // reset print variables
        print = 0;
        print2 = 0;

        if (debug) printf("Whole thing took: %lf\n", omp_get_wtime() - t_whole);

        // End loop if t_final reached
        if(final) {
            t_prime += dt;
            break;
        }

    }

    //! Ensure all memory is freed and files closed
    free(beta);
    free(sigma);
    free(gamma);
    free(kappa);
    free(aging);

    free(S);
    free(E1);
    free(E2);
    free(E3);
    free(I);
    free(R);
    free(D);

    free(S_next);
    free(E1_next);
    free(E2_next);
    free(E3_next);
    free(I_next);
    free(R_next);
    free(D_next);

    free(S_next_tmp);
    free(E1_next_tmp);
    free(E2_next_tmp);
    free(E3_next_tmp);
    free(I_next_tmp);
    free(R_next_tmp);
    free(D_next_tmp);

    free(S_k1);
    free(E1_k1);
    free(E2_k1);
    free(E3_k1);
    free(I_k1);
    free(R_k1);
    free(D_k1);

    free(S_k2);
    free(E1_k2);
    free(E2_k2);
    free(E3_k2);
    free(I_k2);
    free(R_k2);
    free(D_k2);
    
    free(S_k3);
    free(E1_k3);
    free(E2_k3);
    free(E3_k3);
    free(I_k3);
    free(R_k3);
    free(D_k3);

    free(S_k4);
    free(E1_k4);
    free(E2_k4);
    free(E3_k4);
    free(I_k4);
    free(R_k4);
    free(D_k4);

    free(nu);
    free(mu);
    free(l);
    free(r);

    free(inf_prop);
    free(l_prime);
    free(r_prime_S);    
    free(r_prime_E1);    
    free(r_prime_E2);    
    free(r_prime_E3);    
    free(r_prime_I);
    free(r_prime_R);

    free(tmp_inf_prop);

    //No need to free pointer to tmp array
    // due to it pointing to the same array as either SEIR or SEIR_next

    free(distmat);
    free(popmat);
    free(travmat);
    free(transport_mod);
    free(O);

    free(Sij);

    // Close output files
    fclose(f);
    fclose(g);

    //! Final return
    return 0;
}