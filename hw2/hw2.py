import re
import numpy as np
import random
import matplotlib.pyplot as plt

def bandit (k, mu, sigma):
    return np.random.normal(mu[k], sigma[k])

def greedy(epsilon, Q):
    k = np.argmax(Q)
    dim = len(Q)
    bandit_options = range(0, dim, 1)
    sample = np.random.uniform(0, 1)
    if sample <= 1-epsilon:
        return k
    else:
       return np.random.choice(bandit_options)

def ubc(Q, c, time, frecuency):
    avoid_zero = 1e-8
    a = Q + c*np.sqrt(np.log(time+1)/(frecuency+avoid_zero))
    return np.argmax(a) 

if __name__=='__main__':

    #init data
    # np.random.seed(0)
    T=1000
    k=10
    U = list(range(k))
    mu = np.random.uniform(1, 3, k)
    sigma = np.random.uniform(1,5, k)
    # sigma = 0.2 * np.ones(k)

    #Greedy data
    reward_greedy = np.zeros(T)
    epsilon = .9
    Q_greedy = np.zeros(k)
    frecuency_greedy = np.zeros(k)
    
    #UBC data
    reward_ubc = np.zeros(T)
    Q_ubc = np.zeros(k)
    frecuency_ubc = np.zeros(k)
    c=2

    for t in range(T):
        #greedy
        k_greedy = greedy(epsilon, Q_greedy)
        r_greedy = bandit(k_greedy, mu, sigma)
        reward_greedy[t] = (sum(reward_greedy) + r_greedy)/(t+1)
        frecuency_greedy[k_greedy] += 1
        Q_greedy[k_greedy] += (1/frecuency_greedy[k_greedy]) * \
                                (r_greedy - Q_greedy[k_greedy])

        #UBC
        k_ubc = ubc(Q_ubc, c, t, frecuency_ubc)
        r_ubc = bandit(k_ubc, mu, sigma)
        reward_ubc[t] = (sum(reward_ubc) + r_ubc)/(t+1)
        frecuency_ubc[k_ubc] += 1
        Q_ubc[k_ubc] += (1/frecuency_ubc[k_ubc]) * (r_ubc - Q_ubc[k_ubc])

    # print(reward_greedy)
    plt.figure()
    plt.grid()
    plt.title('Cumulative average reward', fontsize=50)
    plt.ylabel('Reward', fontsize=50)
    plt.xlabel('Time', fontsize=50) 
    plt.plot(range(T), reward_ubc, label='UBC', linewidth=7)
    plt.plot(range(T), reward_greedy, label='greedy', linewidth=7)
    plt.legend()
    plt.show()