from cProfile import label
import numpy as np
# import networkx as nt
import matplotlib.pyplot as plt
# from matplotlib import animation
import scipy.spatial.distance as dist

def consensus(initial_state):
    p = len(initial_state)
    adjacency = np.ones((p, p)) - np.eye(p)
    degree = (p-1) * np.eye(p)
    laplacian = (degree - adjacency) / p
    return np.kron(laplacian, np.eye(2))

def cluster_initial_state(initial_state, gamma):
    adjacency = dist.squareform(np.exp( -gamma *dist.pdist(initial_state)))
    degree = np.diag(sum(adjacency))
    laplacian = (degree - adjacency) 
    return np.kron(laplacian, np.eye(2))

def cluster_distances(initial_state, gamma, control):
    p = len(initial_state)
    control = np.reshape(control, initial_state.shape)
    adjacency = dist.squareform(np.exp( -gamma *dist.pdist(control)))
    degree = np.diag(sum(adjacency))
    laplacian = (degree - adjacency) 
    return np.kron(laplacian/p, np.eye(2))


if __name__ == '__main__':
    np.random.seed(10)
    nodes = 20
    initial_state = np.random.rand(nodes, 2)
    time_limit = 20000
    gamma = 15

    states = np.zeros((nodes*2, time_limit))
    states [:, 0] = np.reshape(initial_state, nodes*2)
    control = np.reshape(initial_state, nodes*2)
    
    states2 = np.zeros((nodes*2, time_limit))
    states2 [:, 0] = np.reshape(initial_state, nodes*2) 
    control2 = np.reshape(initial_state, nodes*2)

    states3 = np.zeros((nodes*2, time_limit))
    states3 [:, 0] = np.reshape(initial_state, nodes*2) 
    control3 = np.reshape(initial_state, nodes*2)

    M = consensus(initial_state)
    M2 = cluster_initial_state(initial_state, gamma)
    

    for t in range(time_limit-1):
        control = control - np.dot(M, control)
        states[:, t+1] = control

        control2 = control2 - np.dot(M2, control2)
        states2[:, t+1] = control2
        
        M3 = cluster_distances(initial_state, gamma, control3)
        control3 = control3 - np.dot(M3, control3)
        states3[:, t+1] = control3
        
    plt.figure()
    plt.plot(initial_state[:, 0], initial_state[:, 1], 'ko', markersize=12)
    for i in range(nodes):
        line1, = plt.plot(states[2*i, :], states[2*i+1, :], '--b')
        line2, =plt.plot(states2[2*i, :], states2[2*i+1, :], 'r')
        line3, =plt.plot(states3[2*i, :], states3[2*i+1, :], '.g')

    plt.grid()
    line1.set_label('Consensus')
    line2.set_label('Cluster initial state')
    line3.set_label('Cluster distances')
    plt.legend(fontsize=40)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks(fontsize=44)
    plt.yticks(fontsize=44)
    plt.ylabel('y', fontsize=50)
    plt.xlabel('x', fontsize=50) 
    plt.show()


