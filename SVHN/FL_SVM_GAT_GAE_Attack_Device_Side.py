import numpy as np
import torch
import copy
import time
from sklearn.metrics import accuracy_score
#import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from utilities.SVM import SVM
from utilities.FL_to_GAT import FL_to_GAT
from utilities.GAT_to_GAE import GAT_to_GAE
from utilities.data_processing import load_mnist_return_required_digits, load_svhn_return_required_digits, get_clients, get_total_from_clients, load_cifar10_return_required_digits, create_kmeans_clusters, load_fashion_mnist_return_required_digits, load_imdb_return_required_digits
from scipy.io import savemat

GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)



class Federated_SVM:
    def __init__(self, x_train, y_train, n_clients, n_iters, val=True, val_type='k_fold', k=5, opt='mini_batch_GD',
                 batch_size=64, learning_rate=0.03, lambda_param=0.01):
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
        self.dist = []
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

    def average_aggregator(self, parameter_list, bias_list, num_attacks):
        # l = [int(sum(self.client_distribution) / len(self.client_distribution))] * num_attacks # atackers claimed datasize
        l = [100] * num_attacks
        all_client_distribution = self.client_distribution + l
        #print(all_client_distribution)
        b = np.sum(bias_list) / len(bias_list)

        w = np.zeros(parameter_list[0].shape[0])
        for i in range(0, self.n_clients + num_attacks):
            w = np.add(w, parameter_list[i] * all_client_distribution[i] / sum(all_client_distribution))
        return w, b

    def loss(self, w, b):
        return np.mean([max(0, 1 - x * y) for x, y in zip(np.where(np.concatenate(self.y_train, axis=None) <= 0, -1, 1),
                                                          np.where(np.sign(np.dot(np.vstack(self.x_train), w) - b) < 0,
                                                                   -1, 1))]) #+ 1/2 * np.dot(w, w)



    def kl_divergence(self, p, q):
        epsilon = 1e-10
        p = np.clip(p, epsilon, 1)  # Clip to avoid zeros
        q = np.clip(q, epsilon, 1)  # Clip to avoid zeros
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))

    def fit(self, g_iters,  aggregator, num_malicious):
        w_best = np.zeros(self.X_test.shape[1])
        b_best = 0
        w_surrogate_list = []
        KL_div, cosine_distance = [], []

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
            """Partial local models are added noise """
            num_change = self.n_clients
            mean = np.random.random(num_change) 
            #std = [0.1 for _ in range(self.n_clients)]
            std = np.random.random(num_change) * 5
            #p1 = [self.clients[k].w * np.random.normal(mean[k], std[k], len(w_best)) for k in
            #      range(0, num_change)]  # all weights of clients
            p1 = [self.clients[k].w for k in range(0, num_change)]  # all weights of clients   * np.random.normal(0, 1, len(w_best)) 
            #p2 = [self.clients[i].w for i in range(num_change, self.n_clients)]
            parameter_list = p1 #+ p2
            b_list = np.array([self.clients[k].b  for k in range(0, self.n_clients)])

            #parameter_list = [self.clients[k].w for k in range(0, self.n_clients)]   # all weights of clients
            local_weights_array = np.array(parameter_list) # benign clients; print(local_weights_array.shape) # n*784, n is the number of clients

            if i < 1:
                p_list = np.array([self.clients[k].w  for k in range(0, self.n_clients)])
                bias_list = np.array([self.clients[k].b  for k in range(0, self.n_clients)])
                #w_surrogate = copy.deepcopy(np.sum(p_list, axis=0) / p_list.shape[0])
                #w_surrogate, bias = aggregator(p_list, bias_list, num_malicious)
                w_surrogate, bias = np.mean(p_list, axis=0), np.mean(bias_list, axis=0)
                w_surrogate_list.append(w_surrogate)
                w_agg = copy.deepcopy(w_surrogate)
                b_agg = copy.deepcopy(bias)
            else:
                num_select = self.n_clients # 5, 10, 15, 20. eavesdropped the number of benign models
                random_pick_indices = np.random.choice(self.n_clients, size=num_select, replace=False)
                select_local_weights = np.row_stack((local_weights_array[random_pick_indices], w_surrogate_list[-1]))
                N, M = select_local_weights.shape

                # generates malicious clients
                w_attack_set = []
                for mali in range(num_malicious):
                    rng = np.random.default_rng(seed=42)
                    initialization_graph_matrix = rng.random((N, N))
                    # initialization_graph_matrix = np.random.rand(self.n_clients, self.n_clients)
                    #select_local_weights = torch.from_numpy(select_local_weights).float().to(device)
                    #initialization_graph_matrix = torch.from_numpy(initialization_graph_matrix).float().to(device)
                    gat_nodes_feature = FL_to_GAT(select_local_weights, initialization_graph_matrix)
                    gat_nodes_feature = gat_nodes_feature.cpu().numpy()
                    attack_node = GAT_to_GAE(gat_nodes_feature, initialization_graph_matrix)

                    # replace benign local model with attack node
                    attack_base = select_local_weights[np.random.choice(num_select, size=1, replace=False)][0]
                    indices_to_replace = np.random.choice(len(attack_base), len(attack_node), replace=False)
                    attack_base[indices_to_replace] = attack_node
                    w_attack_set.append(attack_base)

                w_attack_arr = np.array(w_attack_set)
                w_all = np.row_stack((local_weights_array, w_attack_arr))
                #w_agg = copy.deepcopy(np.sum(w_all, axis=0) / w_all.shape[0])
                w_agg,  b_agg = aggregator(w_all, b_list, num_malicious)

                tem_kl_div, tem_cos = [], []
                for idx in range(w_all.shape[0]):
                    tem_kl_div.append(self.kl_divergence(w_all[idx, :], w_agg))
                    tem_cos.append(cosine(w_all[idx, :], w_agg))
                KL_div.append(tem_kl_div)
                cosine_distance.append(tem_cos)
                    #if self.accuracy(w_agg) > self.accuracy(w_best) or i == 0:
            #if i == 0:
            #if i == 0:
            #    w_best = copy.deepcopy(w_agg)
            #    b_best = copy.deepcopy(b_agg)
            if self.accuracy(w_agg,b_agg)>self.accuracy(w_best,b_best) or i==0:
               w_best=copy.deepcopy(w_agg)
               b_best=copy.deepcopy(b_agg)
            #print('global test acc',self.accuracy(w_best,b_best))
            self.Loss.append(self.loss(w_agg,b_agg))
            self.timefit.append(time.time())
            print('global test acc', self.accuracy(w_agg, b_agg))
            self.global_accuracy.append(self.accuracy(w_agg, b_agg))
        return self.Loss, self.global_accuracy, self.local_clients_accuracy, KL_div, cosine_distance



    def predict(self, w, b):
        approx = np.dot(self.X_test, w) - b
        approx = np.sign(approx) # The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0. nan is returned for nan inputs.
        return np.where(approx< 0, 0, 1)

    def accuracy(self, w, b):
        return accuracy_score(self.y_test, self.predict(w,b))


if __name__ == '__main__':
    #dataset = ["mnist", "fashion_mnist", "cifar10"]
    #dataset = ["cifar10"]
    dataset = ["svhn"]
    #dataset = ["fashion_mnist"]
    #dataset = ["imdb"]

    for x in dataset:
        # choose dataset
        if x == "mnist":
            """Loading the data"""
            data = load_mnist_return_required_digits(0, 6)  # load data, image of digit 0 and digit 6
        elif x == "fashion_mnist":
            data = load_fashion_mnist_return_required_digits(3, 8)
        elif x == "svhn":
            data = load_svhn_return_required_digits(0, 9)
        elif x == "imdb":
            data = load_imdb_return_required_digits(0, 1)
        else:
            """0:airplane; 1:automobile; 2:bird; 3:cat; 4:deer;
                5:dog; 6:frog; 7:horse; 8:ship; 9:truck"""
            data = load_cifar10_return_required_digits(1, 7)  # load data, image of label 1 and label 7
        """Creation of individual train sets for the clients, 
        global train set for the SVM model and a global test set 
        containing the data from all the clusters"""
        # n_clients = n_clusters # number of clients
        num_clients_index = [5, 10, 15, 20, 25, 30]
        #num_clients_index = [5]
        #num_malicious_clients_index = [1]
        num_malicious_clients_index = [1,2,3,4,5]
        for n_clients in num_clients_index:
            clients_X, clients_y, X_test, y_test = get_clients(data[0][0], data[1][0], n_clients)
            xtrain_gl, ytrain_gl = get_total_from_clients(clients_X, clients_y)



            """ Batch global / SGD+Batch"""
            num_iters_index=[2,8,4,5]
            n_iters = num_iters_index[1] # number of local iterations

            num_global_commu_round_index = [100, 200, 300]
            n_global_commu_round = num_global_commu_round_index[0] # number of global communicaton round


            for m_clients in num_malicious_clients_index:
                f_svm = Federated_SVM(xtrain_gl, ytrain_gl, n_clients, n_iters, val=False,  opt='batch_GD')
                f_svm.create_clients(clients_X, clients_y, X_test, y_test)
                Loss, global_accuracy,local_clients_accuracy, KL_div, cosine_distance = f_svm.fit(n_global_commu_round, f_svm.average_aggregator, m_clients)
                KL_div = np.array(KL_div)
                cosine_distance = np.array(cosine_distance)
                #print(KL_div)

                savemat("./FL_GAT_GAE_results_{}_{}_{}_{}_{}.mat".format(x, n_clients, n_global_commu_round, n_iters,
                                                                         m_clients),
                        {"Global_model_loss": Loss, "Global_model_accuracy": global_accuracy,
                         "Local_model_accuracy": local_clients_accuracy, "KL_divergence": KL_div,
                         "cosine_distance": cosine_distance})


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
                #.close()


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
                #plt.close()
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
                # # plot local accuracy
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

                #plt.show()
                #plt.close()
