import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from elfi.examples import stochastic_volatility_model
from elfi.methods.bsl.select_penalty import select_penalty
import elfi
import time


def run_asvm():
    m = stochastic_volatility_model.get_model(seed_obs=123)

    batch_size = 20000
    true_params = [1.2, 0.5]
    elfi.SyntheticLikelihood("semiBsl", m['identity'], name="semiSL")
    asvm_bsl = elfi.BSL(
        m['semiSL'],
        batch_size=batch_size,
        # method="semiBsl",
        seed=3
    )
    M = 30
    log_SL = asvm_bsl.log_SL_stdev(true_params, batch_size, M)
    print('log_SL', log_SL)

    # asvm_bsl.plot_summary_statistics(batch_size=20000, theta_point=true_params)
    # plt.savefig("asvm_summaries.png")
    # print(1/0)
    tic = time.time()
    mcmc_iterations = 200
    bsl_res = asvm_bsl.sample(
        mcmc_iterations,
        sigma_proposals=0.01*np.eye(2),
        params0=true_params,
        burn_in=20
    )
    toc = time.time()
    print('time: ', toc - tic)

    print(bsl_res)
    ess = bsl_res.compute_ess()
    print(ess)
    est_cov_mat = bsl_res.get_sample_covariance()
    print('est_cov_mat', est_cov_mat)
    bsl_res.plot_marginals(bins=50)

    plt.savefig("asvm_marginals_identity.png")
    bsl_res.plot_pairs(bins=50)
    plt.savefig("asvm_pairs_identity.png")
    bsl_res.plot_traces()
    plt.savefig("plot_traces_asvm.png")

if __name__ == '__main__':
    run_asvm()
