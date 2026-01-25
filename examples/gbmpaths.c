#include <math.h>
#include <ndarray.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    double mu;    /* drift coefficient */
    double sigma; /* volatility coefficient */
} GBMParams;

typedef struct {
    double T; /* time to maturity */
    int M;    /* number of steps by period */
    int I;    /* simulation paths */
} SimParams;

NDArray generateGBMPaths(GBMParams gbmPar, SimParams simPar, double S0) {
    /* Auxiliar variables */
    int steps = simPar.M * (int)simPar.T + 1;
    double dt = 1.0 / (double)simPar.M;
    double sdt = sqrt(dt);
    double drift = (gbmPar.mu - 0.5 * gbmPar.sigma * gbmPar.sigma) * dt;
    /* Prices array */
    NDArray S = ndarray_new_zeros(NDA_DIMS(steps, simPar.I));
    ndarray_fill_slice(S, 0, 0, S0);
    /* Wiener process */
    NDArray z = ndarray_new_randnorm(NDA_DIMS(steps, simPar.I), 0.0, 1.0);
    /* Auxiliar arrays */
    NDArray S_t = ndarray_new(NDA_DIMS(1, simPar.I));
    NDArray z_t = ndarray_new(NDA_DIMS(1, simPar.I));
    for (int step = 1; step < steps; step++) {
        /* z_t = z[step, :] */
        ndarray_copy_slice(z_t, 0, 0, z, 0, step);
        /* z_t = z_t * sigma * sdt + drift */
        ndarray_mul_scalar(z_t, gbmPar.sigma * sdt);
        ndarray_add_scalar(z_t, drift);
        /* z_t = exp(z_t) */
        ndarray_mapfnc(z_t, exp);
        /* S_t = S[step-1, :] */
        ndarray_copy_slice(S_t, 0, 0, S, 0, step - 1);
        /* S_t = S_t * z_t */
        ndarray_mul(S_t, z_t);
        /* S[step, :] = S_t */
        ndarray_copy_slice(S, 0, step, S_t, 0, 0);
    }
    ndarray_free_all(NDA_LIST(z, S_t, z_t));
    return S;
}

int main() {
    GBMParams gbmParams = {
        .mu = 0.05,
        .sigma = 0.2,
    };
    SimParams simParams = {
        .T = 1.0,
        .M = 252,
        .I = 10000
    };
    double S0 = 100.0;
    NDArray paths = generateGBMPaths(gbmParams, simParams, S0);
    ndarray_print(paths, "Prices", 4);
    ndarray_free(paths);
    return 0;
}
