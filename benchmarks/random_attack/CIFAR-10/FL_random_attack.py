import matplotlib.pyplot as plt
import numpy as np
import random
import copy
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cosine
import time
from utilities.SVM import SVM
from utilities.data_processing import create_kmeans_clusters, load_mnist_return_required_digits, load_fashion_mnist_return_required_digits, load_cifar10_return_required_digits,load_svhn_return_required_digits, get_clients, get_total_from_clients
from scipy.io import savemat
import scipy.io as sio

class Federated_SVM:
    def __init__(self, x_train ,y_train, n_clients, n_iters, val=True, val_type='k_fold', k=5, opt='mini_batch_GD',
                 batch_size=64, learning_rate=0.0001, lambda_param=0.01):
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
        #self.Loss_clients = [[0], [0], [0]]
        array_loss_clients = np.ones((self.n_clients, 1)) * 0
        self.Loss_clients = array_loss_clients.tolist()
        self.local_clients_accuracy = array_loss_clients.tolist()
        self.timefit = []
        self.dist = []

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

    def loss(self, w):
        # print(np.concatenate(self.y_train,axis=None))
        # print(np.vstack(self.x_train).shape)#, np.where(np.sign(np.dot(self.x_train, w) - b))<0, 0, 1).shape)
        return np.mean([max(0, 1 - x * y) for x, y in zip(np.where(np.concatenate(self.y_train, axis=None) <= 0, -1, 1),
                                                          np.where(np.sign(np.dot(np.vstack(self.x_train), w)) < 0,
                                                                   -1, 1))])
    def kl_divergence(self, p, q):
        epsilon = 1e-10
        p = np.clip(p, epsilon, 1)  # Clip to avoid zeros
        q = np.clip(q, epsilon, 1)  # Clip to avoid zeros
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))


    def fit(self, g_iters, aggregator, num_attcks, noise):
        w_best = np.zeros(self.X_test.shape[1])
        #b_best = 0
        KL_div, cosine_distance = [], []
        for i in range(0, g_iters):
            print('global round', i + 1)
            for j in range(0, self.n_clients):
                if i == 0:
                    self.clients[j].fit()
                else:
                    self.clients[j].w = copy.deepcopy(w_agg)
                    #self.clients[j].b = copy.deepcopy(b_agg)
                    self.clients[j].fit()
                self.Loss_clients[j].append(self.clients[j].loss())
                self.local_clients_accuracy[j].append(self.clients[j].accuracy())
                print('client', j + 1, self.clients[j].accuracy())
            parameter_list = []
            for k in range(0, self.n_clients):
                parameter_list.append(self.clients[k].w)  # all weights of clients
                #parameter_list.append(self.clients[k].b)
            # w_attack = FL_GAE(parameter_list)  ## GAE generates new local weight according to other local weights
            # parameter_list = parameter_list.append(w_attack)
            #w_attack = np.random.random(len(self.clients[k].w))


            w_attack = noise[num_attcks*i : num_attcks*(i+1)]
            for mail in range(num_attcks):
               parameter_list.append(w_attack[mail]*1e-3)
                
                
            # for i in range(num_attcks):
            #     w_attack = np.random.normal(0, 1, size=len(self.clients[k].w))*1e-3
            #     parameter_list.append(w_attack)

                    
            w_all = np.array(parameter_list)

            w_agg = aggregator(parameter_list, num_attcks)

            tem_kl_div, tem_cos = [], []
            for idx in range(w_all.shape[0]):
                tem_kl_div.append(self.kl_divergence(w_all[idx, :], w_agg))
                tem_cos.append(cosine(w_all[idx, :], w_agg))
            KL_div.append(tem_kl_div)
            cosine_distance.append(tem_cos)
            # print("agg",self.accuracy(w_agg,b_agg),"best",self.accuracy(w_best,b_best))
            if self.accuracy(w_agg) > self.accuracy(w_best) or i == 0:
                w_best = copy.deepcopy(w_agg)
                #b_best = copy.deepcopy(b_agg)
            self.Loss.append(self.loss(w_best))
            self.timefit.append(time.time())
            print('global test acc', self.accuracy(w_agg))
            self.global_accuracy.append(self.accuracy(w_agg))

        return self.Loss, self.global_accuracy, self.local_clients_accuracy, KL_div, cosine_distance

    def predict(self, w):
        approx = np.dot(self.X_test, w)
        approx = np.sign(approx)
        return np.where(approx < 0, 0, 1)

    def accuracy(self, w):
        return accuracy_score(self.y_test, self.predict(w))




if __name__ == '__main__':

    #dataset = ["mnist", "fashion_mnist", "cifar10"]
    dataset = ["cifar10"]
    #dataset = ["svhn"]

    for x in dataset:
      # choose dataset
      if x == "mnist":
         """Loading the data"""
         data = load_mnist_return_required_digits(0, 6)  # load data, image of digit 0 and digit 6
         #noise_mat = sio.loadmat('./noise_set/mnist_noise.mat')
         #noise = noise_mat['mnist_noise']
      elif x == "fashion_mnist":
         data = load_fashion_mnist_return_required_digits(3, 8)
         # noise_mat = sio.loadmat('./utilities/fmnist_noise.mat')
         # noise = noise_mat['fmnist_noise']
      elif x == "svhn":
          data = load_svhn_return_required_digits(3, 8)
          #noise_mat = sio.loadmat('./noise_set/svhn_noise.mat')
          #noise = noise_mat['svhn_noise']
      else:
         """0:airplane; 1:automobile; 2:bird; 3:cat; 4:deer;
         5:dog; 6:frog; 7:horse; 8:ship; 9:truck"""
         data = load_cifar10_return_required_digits(0, 6)  # load data, image of label 4 and label 5
         noise_mat = sio.loadmat('./utilities/cifar_noise.mat')
         noise = noise_mat['cifar_noise']


      """Creation of individual train sets for the clients, 
      global train set for the SVM model and a global test set 
      containing the data from all the clusters"""

      #num_clients_index = [5, 10, 15, 20, 25, 30]
      num_clients_index = [20, 25, 30]
      #num_attacks_index = [1, 2, 3, 4, 5]
      num_attacks_index = [5]
      #n_clients = num_clients_index[0] # number of clients
      for n_clients in num_clients_index:
         clients_X, clients_y, X_test, y_test = get_clients(data[0][0], data[1][0], n_clients)
         xtrain_gl, ytrain_gl = get_total_from_clients(clients_X, clients_y)


         """ Batch global / SGD+Batch"""
         num_iters_index = [2, 3, 4, 5]
         n_iters = num_iters_index[2] # number of local iterations

         num_global_commu_round_index = [100, 200, 300]
         n_global_commu_round = num_global_commu_round_index[0] # number of global communicaton round
         f_svm = Federated_SVM(xtrain_gl, ytrain_gl, n_clients, n_iters, val=False,  opt='batch_GD')

         f_svm.create_clients(clients_X, clients_y, X_test, y_test)

         for num_attcks in num_attacks_index:
           Loss, global_accuracy, local_clients_accuracy, KL_div, cosine_distance = f_svm.fit(n_global_commu_round, f_svm.average_aggregator, num_attcks, noise)
           KL_div = np.array(KL_div)
           cosine_distance = np.array(cosine_distance)


           savemat("./FL_random_attack_results_{}_{}_{}_{}_{}.mat".format(x, n_clients, n_global_commu_round, n_iters, num_attcks), {"Global_model_loss": Loss, "Global_model_accuracy": global_accuracy,
                                         "Local_model_accuracy": local_clients_accuracy,  "KL_divergence": KL_div, "cosine_distance": cosine_distance})


           # plot loss curve
           plt.figure()
           plt.plot(range(n_global_commu_round), Loss)
           # plt.xlabel('Communication rounds')
           plt.ylabel('train_loss')
           plt.savefig('./fed_{}_{}_{}_{}_{}.png'.format(x, n_clients, n_global_commu_round,  n_iters, num_attcks))

           # plot accuracy curve
           plt.figure()
           plt.plot(range(n_global_commu_round), global_accuracy)
           plt.xlabel('Communication rounds')
           plt.ylabel('Accuracy of global model')
           plt.savefig('./fed_glob_acc_{}_{}_{}_{}_{}.png'.format(x,n_clients, n_global_commu_round, n_iters, num_attcks))


           # plot local accuracy
           plt.figure()
           #color_list = ['green', 'red', 'yellow', 'blue', 'cyan']
           #label_list = ['Device 1', 'Device 2', 'Device 3', 'Device 4', 'Device 5']
           for i in range(n_clients):
              #plt.plot(range(n_global_commu_round), local_clients_accuracy[i][1:n_global_commu_round + 1], color=color_list[i], label=label_list[i])
              plt.plot(range(n_global_commu_round), local_clients_accuracy[i][1:n_global_commu_round + 1])
           plt.legend()
           plt.xlabel('Communication rounds')
           plt.ylabel('Accuracy of local clients')
           #plt.title('Communication rounds ={}, local iterations = {}'.format(n_global_commu_round, n_iters))  # 显示图例
           plt.savefig('./local_clients_acc_{}_{}_{}_{}_{}.png'.format(x, n_clients, n_global_commu_round, n_iters, num_attcks))
           #plt.savefig('./FL_no_attack_results/no_bias/local_clients_acc_{}_{}_{}_{}.eps'.format(x, n_clients, n_global_commu_round, n_iters))

           N, M = KL_div.shape
           plt.figure()
           for i in range(M):
               # plt.plot(range(n_global_commu_round), local_clients_accuracy[i][1:n_global_commu_round + 1], color=color_list[i], label=label_list[i])
               plt.plot(range(N), KL_div[:, i], label=f'Device {i}')
           plt.legend()
           plt.xlabel('Communication rounds')
           plt.ylabel('KL divergence')
           # #plt.title('Communication rounds ={}, local iterations = {}'.format(n_global_commu_round, n_iters))  # show legend
           plt.savefig('./local_devices_KL_{}_{}_{}_{}_{}.png'.format(x, n_clients,  n_global_commu_round, n_iters, num_attcks))

           plt.figure()
           for i in range(M):
               # plt.plot(range(n_global_commu_round), local_clients_accuracy[i][1:n_global_commu_round + 1], color=color_list[i], label=label_list[i])
               plt.plot(range(N), cosine_distance[:, i], label=f'Device {i}')
           plt.legend()
           plt.xlabel('Communication rounds')
           plt.ylabel('Cosine distance')
           # #plt.title('Communication rounds ={}, local iterations = {}'.format(n_global_commu_round, n_iters))  # show legend
           plt.savefig('./local_devices_cosine_{}_{}_{}_{}_{}.png'.format(x, n_clients, n_global_commu_round, n_iters,
                                                                        num_attcks))

           # plt.show()
           # plt.close()
