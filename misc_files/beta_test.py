import time

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import elfi

def run_beta():
    seed = 1
    np.random.seed(seed)

    def sim_fn(a, b, batch_size=1, random_state=None):
        random_state = random_state or np.random
        n_obs = 20
        # a = a if isinstance(a, float) else a[0]
        # b = b if isinstance(b, float) else b[0]
        return np.transpose(random_state.beta(a, b, size=(n_obs, batch_size)))
        
    def identity(x):
        return x

    true_params = np.array([2.0, 5.0])

    y_obs = sim_fn(2.0, 5.0)
    m = elfi.ElfiModel()
    elfi.Prior('uniform', 0, 10, model=m, name='a')
    elfi.Prior('uniform', 0, 10, model=m, name='b')
    elfi.Simulator(sim_fn, m['a'], m['b'], observed=y_obs, name='beta') #, parallelise=True, num_processes=None)
    elfi.Summary(identity, m['beta'], name='identity')
    elfi.SyntheticLikelihood("bsl", m['identity'], name="SL")

    batch_size = 300
    bsl_obj = elfi.BSL(
                m['SL'],
                batch_size=batch_size,
                seed=123,
                )

    M = 30
    log_SL = bsl_obj.log_SL_stdev(true_params, batch_size, M)
    print('log_SL', log_SL)

    bsl_res = bsl_obj.sample(5000, batch_size=batch_size, sigma_proposals=np.eye(2), burn_in=1000
                            #  logitTransformBound=np.array([[0, 10], [0, 10]])
                             )
    print('bsl_res', bsl_res)
    bsl_res.plot_pairs()
    plt.savefig("beta_pairs.png")
    # bsl_res - elfi.BSL(2000, batch_size)


if __name__ == '__main__':
    run_beta()