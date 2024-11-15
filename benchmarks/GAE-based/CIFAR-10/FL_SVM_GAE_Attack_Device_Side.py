import numpy as np
import torch
import copy
import math, time
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cosine
#import networkx as nx
import matplotlib.pyplot as plt
from utilities.SVM import SVM
from utilities.FL_to_GAE import fl_to_gae
from utilities.data_processing import load_mnist_return_required_digits, get_clients, get_total_from_clients, load_cifar10_return_required_digits, create_kmeans_clusters, load_fashion_mnist_return_required_digits
from scipy.io import savemat
import scipy.io as sio


GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)



class Federated_SVM:
    def __init__(self, x_train, y_train, n_clients, n_iters, val=True, val_type='k_fold', k=5, opt='mini_batch_GD',
                 batch_size= 32, learning_rate=0.001, lambda_param=0.01):
        self.n_clients = n_clients
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.val = val
        self.val_type = val_type
        self.client_distribution = [] # [3372, 3928, 3721] data size of each client
        self.k = k
        self.opt = opt
        self.batch_size = batch_size
        self.X_test = None
        self.y_test = None
        self.x_train = x_train
        self.y_train = y_train
        self.Loss = []
        self.global_accuracy = []
        array_loss_clients = np.ones((self.n_clients,1))*0
        self.Loss_clients = array_loss_clients.tolist()
        self.local_clients_accuracy = array_loss_clients.tolist()
        self.timefit = []

    def create_clients(self, X_train, y_train, X_test, y_test):
        self.clients = []
        for i in range(self.n_clients):
            self.client_distribution.append(X_train[i][0].shape[0] + X_train[i][1].shape[0])
            self.clients.append(SVM(X_train[i], y_train[i], X_test, y_test, self.n_iters, self.val, self.val_type, self.k, self.opt, self.batch_size,
                     self.learning_rate, self.lambda_param))
        self.X_test = copy.deepcopy(X_test)
        self.y_test = copy.deepcopy(y_test)

    def average_aggregator(self, parameter_list, num_attcks):
        l = [100] * num_attcks # atackers claimed datasize
        inter = self.client_distribution + l
        self.client_distribution = inter
        w = np.zeros(parameter_list[0].shape[0])
        #b = 0
        for i in range(0, len(parameter_list)):
            w = np.add(w, parameter_list[i] * self.client_distribution[i] / sum(self.client_distribution))
           # b = b + parameter_list[i + 1] * self.client_distribution[i // 2] / sum(self.client_distribution)
        return w

    def kl_divergence(self, p, q):
        epsilon = 1e-10
        p = np.clip(p, epsilon, 1)  # Clip to avoid zeros
        q = np.clip(q, epsilon, 1)  # Clip to avoid zeros
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))

    def angle_between_vectors(self, a, b):
        a = np.array(a)
        b = np.array(b)
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        cos_theta = dot_product / (norm_a * norm_b)
        angle_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clipping for numerical stability
        # Convert the angle to degrees if needed (optional)
        angle_degrees = np.degrees(angle_radians)
        return angle_radians, angle_degrees

    def jains_fairness_index(self, allocations):
        """
        Calculate Jain's Fairness Index for a list of allocations.
        Parameters:
        allocations (list or np.array): A list or array of resource allocations (positive numbers).

        Returns:
        float: Jain's fairness index, ranging from 0 to 1.
        """
        allocations = np.array(allocations)
        # Calculate the numerator: (sum of allocations)^2
        numerator = np.sum(allocations) ** 2
        # Calculate the denominator: n * (sum of squared allocations)
        denominator = len(allocations) * np.sum(allocations ** 2)
        # Compute Jain's fairness index
        fairness_index = numerator / denominator if denominator != 0 else 0
        return fairness_index



    def loss(self, w, b):
        return np.mean([max(0, 1 - x * y) for x, y in zip(np.where(np.concatenate(self.y_train, axis=None) <= 0, -1, 1),
                                                          np.where(np.sign(np.dot(np.vstack(self.x_train), w) -b) < 0,
                                                                   -1, 1))])

    def fit(self, g_iters, aggregator, num_attcks):
        w_best = np.zeros(self.X_test.shape[1])
        b_best = 0
        KL_div, cosine_distance = [], []
        mould_len_set, angle_set = [], []
        length_jains_set, angle_jains_set = [], []
        for i in range(0, g_iters):
            print('global round', i + 1)
            for j in range(0, self.n_clients):

                if i == 0:
                    self.clients[j].fit()
                else:
                    self.clients[j].w = copy.deepcopy(w_agg)
                    self.clients[j].b = copy.deepcopy(b_agg)
                    self.clients[j].fit()
                self.Loss_clients[j].append(self.clients[j].loss())
                self.local_clients_accuracy[j].append(self.clients[j].accuracy())
                print('client', j + 1, self.clients[j].accuracy())

            parameter_list = [self.clients[k].w for k in range(0, self.n_clients)]   # all weights of clients
            bias_list = np.array([self.clients[k].b for k in range(0, self.n_clients)])
            b_agg = np.sum(bias_list) / len(bias_list)


            local_weights_array = np.array(parameter_list) # benign clients; print(local_weights_array.shape) # n*784, n is the number of clients

            num_select = self.n_clients #5, 10, 15, 20
            random_pick_indices = np.random.choice(self.n_clients, size=num_select, replace=False)
            select_local_weights = local_weights_array[random_pick_indices]

            #generates malicious clients
            w_attack_set = []
            for mali in range(num_attcks):
                 if i == 0:
                    rng = np.random.default_rng(seed=42)
                    initialization_graph_matrix = rng.random((num_select, num_select))
                     #initialization_graph_matrix = np.random.rand(self.n_clients, self.n_clients)
                    w_attack, new_adj_matrix = fl_to_gae(select_local_weights, initialization_graph_matrix)
                 else:
                    new_graph_edges = copy.deepcopy(new_adj_matrix)
                    w_attack, new_adj_matrix = fl_to_gae(select_local_weights, new_graph_edges)
                 w_attack.flatten()
                 w_attack_set.append(w_attack)
            w_attack_arr = np.array(w_attack_set)
            w_all = np.row_stack((local_weights_array, w_attack_arr))
            #w_agg = copy.deepcopy(np.sum(w_all, axis=0) / w_all.shape[0])
            w_agg = aggregator(w_all, num_attcks)


            tem_kl_div, tem_cos = [], []
            lengths, angles = [], []
            for idx in range(w_all.shape[0]):
                tem_kl_div.append(self.kl_divergence(w_all[idx, :], w_agg))
                tem_cos.append(cosine(w_all[idx, :], w_agg))
                length = math.sqrt(sum(x ** 2 for x in w_all[idx, :]))
                lengths.append(length)
            mould_len_set.append(lengths)
            length_jains_idx = self.jains_fairness_index(lengths)
            length_jains_set.append(length_jains_idx)
            KL_div.append(tem_kl_div)
            cosine_distance.append(tem_cos)

            for k in range(1, w_all.shape[0]):
                angle_radians, angle_degrees = self.angle_between_vectors(w_all[0, :], w_all[k, :])
                angles.append(angle_degrees)
            angle_set.append(angles)
            angle_jains_idx = self.jains_fairness_index(angles)
            angle_jains_set.append(angle_jains_idx)

            if self.accuracy(w_agg, b_agg) > self.accuracy(w_best, b_best) or i == 0:
                w_best = copy.deepcopy(w_agg)
                b_best = copy.deepcopy(b_agg)
            # print('global test acc',self.accuracy(w_best,b_best))
            self.Loss.append(self.loss(w_best, b_best))
            self.timefit.append(time.time())
            print('global test acc', self.accuracy(w_agg, b_agg))
            self.global_accuracy.append(self.accuracy(w_agg, b_agg))
        return self.Loss, self.global_accuracy, self.local_clients_accuracy, KL_div, cosine_distance, length_jains_set, angle_jains_set



    def predict(self, w, b):
        approx = np.dot(self.X_test, w) - b
        approx = np.sign(approx)
        return np.where(approx< 0, 0, 1)

    def accuracy(self, w, b):
        return accuracy_score(self.y_test, self.predict(w, b))


if __name__ == '__main__':
    #dataset = ["mnist", "fashion_mnist", "cifar10"]
    dataset = ["cifar10"]

    for x in dataset:
        # choose dataset
        if x == "mnist":
            """Loading the data"""
            data = load_mnist_return_required_digits(0, 6)  # load data, image of digit 0 and digit 6
            # noise_mat = sio.loadmat('./noise_set/mnist_noise.mat')
            # noise = noise_mat['mnist_noise']
        elif x == "fashion_mnist":
            data = load_fashion_mnist_return_required_digits(3, 8)
            noise_mat = sio.loadmat('./noise_set/fmnist_noise.mat')
            noise = noise_mat['fmnist_noise']
        elif x == "svhn":
            data = load_svhn_return_required_digits(3, 8)
            # noise_mat = sio.loadmat('./noise_set/svhn_noise.mat')
            # noise = noise_mat['svhn_noise']
        else:
            """0:airplane; 1:automobile; 2:bird; 3:cat; 4:deer;
            5:dog; 6:frog; 7:horse; 8:ship; 9:truck"""
            data = load_cifar10_return_required_digits(0, 6)  # load data, image of label 4 and label 5
            # noise_mat = sio.loadmat('./noise_set/cifar_noise.mat')
            # noise = noise_mat['cifar_noise']

        """Creation of individual train sets for the clients, 
        global train set for the SVM model and a global test set 
        containing the data from all the clusters"""
        # n_clients = n_clusters # number of clients
        #num_clients_index = [5, 10, 15, 20, 25, 30]
        num_clients_index = [20]
        #num_malicious_clients_index = [2, 4, 6, 8, 10]
        #num_malicious_clients_index = [2, 10]
        num_malicious_clients_index = [5]
        for n_clients in num_clients_index:
            clients_X, clients_y, X_test, y_test = get_clients(data[0][0], data[1][0], n_clients)
            xtrain_gl, ytrain_gl = get_total_from_clients(clients_X, clients_y)



            """ Batch global / SGD+Batch"""
            num_iters_index=[2,3,4,5,10]
            n_iters = num_iters_index[2] # number of local iterations

            num_global_commu_round_index = [110, 200, 300, 400, 500]
            n_global_commu_round =  num_global_commu_round_index[0] # number of global communicaton round


            for m_clients in num_malicious_clients_index:
                f_svm = Federated_SVM(xtrain_gl, ytrain_gl, n_clients, n_iters, val=False,  opt='batch_GD')
                f_svm.create_clients(clients_X, clients_y, X_test, y_test)
                Loss, global_accuracy, local_clients_accuracy, KL_div, cosine_distance, length_jains_set, angle_jains_set = f_svm.fit(n_global_commu_round, f_svm.average_aggregator, m_clients)
                KL_div = np.array(KL_div)
                cosine_distance = np.array(cosine_distance)

                savemat("./FL_GAE_results_{}_{}_{}_{}_{}.mat".format(x, n_clients, n_global_commu_round, n_iters,
                                                                     m_clients),
                        {"Global_model_loss": Loss, "Global_model_accuracy": global_accuracy,
                         "Local_model_accuracy": local_clients_accuracy, "KL_distance": KL_div, "cosine_distance": cosine_distance, "length_jains_set": length_jains_set, "angle_jains_set": angle_jains_set})

                # plot loss curve
                # plt.figure()
                # plt.plot(range(n_global_commu_round), Loss, color="red")
                # plt.xlabel('Communication rounds')
                # plt.ylabel('Train_loss')
                # plt.savefig('./FL_GAE_attack_results/fed_{}_{}_{}_{}.png'.format(x, n_clients, n_global_commu_round, n_iters))

                # # plot global accuracy
                plt.figure()
                plt.plot(range(n_global_commu_round), global_accuracy)
                plt.xlabel('Communication rounds')
                plt.ylabel('Accuracy of global model')
                plt.savefig('./fed_glob_acc_{}_{}_{}_{}_{}.png'.format(x, n_clients,  n_global_commu_round, n_iters, m_clients))
                # #plt.savefig('./fed_glob_acc_{}_{}_{}_{}.eps'.format(x,  n_clients, n_global_commu_round, n_iters))


                # # plot local accuracy
                plt.figure()
                #color_list = ['green', 'red', 'yellow', 'blue', 'cyan']
                # # #color_list = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black']
                #label_list = ['Device 1', 'Device 2', 'Device 3', 'Device 4', 'Device 5']
                for i in range(n_clients):
                    #plt.plot(range(n_global_commu_round), local_clients_accuracy[i][1:n_global_commu_round + 1], color=color_list[i], label=label_list[i])
                    plt.plot(range(n_global_commu_round), local_clients_accuracy[i][1:n_global_commu_round + 1])
                # plt.legend()
                plt.xlabel('Communication rounds')
                plt.ylabel('Accuracy of clients')
                # #plt.title('Communication rounds ={}, local iterations = {}'.format(n_global_commu_round, n_iters))  # show legend
                plt.savefig('./local_devices_accuracy_{}_{}_{}_{}_{}.png'.format(x, n_clients, n_global_commu_round,n_iters, m_clients))

                N, M = KL_div.shape
                plt.figure()
                for i in range(M):
                    # plt.plot(range(n_global_commu_round), local_clients_accuracy[i][1:n_global_commu_round + 1], color=color_list[i], label=label_list[i])
                    plt.plot(range(N), KL_div[:, i], label=f'Device {i}')
                plt.legend()
                plt.xlabel('Communication rounds')
                plt.ylabel('KL divergence')
                # #plt.title('Communication rounds ={}, local iterations = {}'.format(n_global_commu_round, n_iters))  # show legend
                plt.savefig('./local_devices_KL_{}_{}_{}_{}_{}.png'.format(x, n_clients, n_global_commu_round, n_iters,
                                                                           m_clients))

                plt.figure()
                for i in range(M):
                    # plt.plot(range(n_global_commu_round), local_clients_accuracy[i][1:n_global_commu_round + 1], color=color_list[i], label=label_list[i])
                    plt.plot(range(N), cosine_distance[:, i], label=f'Device {i}')
                plt.legend()
                plt.xlabel('Communication rounds')
                plt.ylabel('Cosine distance')
                # #plt.title('Communication rounds ={}, local iterations = {}'.format(n_global_commu_round, n_iters))  # show legend
                plt.savefig('./local_devices_cosine_{}_{}_{}_{}_{}.png'.format(x, n_clients, n_global_commu_round, n_iters,
                                                                       m_clients))

                plt.figure()
                plt.plot(range(len(length_jains_set)), length_jains_set)
                plt.legend()
                plt.xlabel('Communication rounds')
                plt.ylabel('Jains Fairness Index of Length')
                plt.savefig(
                    './jains_lengths_{}_{}_{}_{}_{}.png'.format(x, n_clients, n_global_commu_round, n_iters,
                                                                m_clients))

                plt.figure()
                plt.plot(range(len(angle_jains_set)), angle_jains_set)
                plt.legend()
                plt.xlabel('Communication rounds')
                plt.ylabel('Jains Fairness Index of Angle')
                plt.savefig(
                    './jains_angles_{}_{}_{}_{}_{}.png'.format(x, n_clients, n_global_commu_round, n_iters,
                                                               m_clients))

                plt.show()

                # #plt.savefig('./FL_GAE_attack_results/local_devices_accuracy_{}_{}_{}_{}.eps'.format(x, n_clients, n_global_commu_round, n_iters))
                #
                #
                # # plot gae loss curve
                # plt.figure()
                # plt.plot(range(n_global_commu_round), gae_loss_list)
                # plt.xlabel('Communication rounds')
                # plt.ylabel('gae training loss')
                # plt.savefig('./FL_GAE_attack_results/gae_loss_{}_{}_{}.png'.format(x,  n_clients, n_global_commu_round, n_iters))
                #
                # # plot distance
                # plt.figure()
                # plt.plot(range(n_global_commu_round), distance)
                # plt.xlabel('Communication rounds')
                # plt.ylabel('distance')
                # plt.savefig('./FL_GAE_attack_results/distance_{}_{}_{}_{}.png'.format(x,  n_clients, n_global_commu_round, n_iters))
                #

