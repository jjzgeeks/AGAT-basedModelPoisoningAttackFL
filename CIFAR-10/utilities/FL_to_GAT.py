import argparse
import time
import torch
import networkx as nx
import torch.nn as nn
from torch.optim import Adam
from .GAT import GAT
from .constants import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

def get_main_loop(config, gat, cross_entropy_loss, optimizer, node_features, node_labels, edge_index, train_indices):
    node_dim = 0  # node axis
    train_labels = node_labels.index_select(node_dim, train_indices)
    # node_features shape = (N, FIN), edge_index shape = (2, E)
    #node_features = torch.from_numpy(node_features).to(device)
    #edge_index = torch.from_numpy(edge_index).to(device)
    graph_data = (node_features, edge_index)  # I pack data into tuples because GAT uses nn.Sequential which requires it


    def main_loop(phase, epoch=0):
        #global BEST_VAL_PERF, BEST_VAL_LOSS, PATIENCE_CNT, writer

        # Certain modules behave differently depending on whether we're training the model or not.
        # e.g. nn.Dropout - we only want to drop model weights during the training.

        gat.train()
        node_indices = train_indices
        gt_node_labels = train_labels  # gt stands for ground truth

        nodes_unnormalized_scores = gat(graph_data)[0].index_select(node_dim, node_indices)
        loss = cross_entropy_loss(nodes_unnormalized_scores, gt_node_labels)
        optimizer.zero_grad()  # clean the trainable weights gradients in the computational graph (.grad fields)
        loss.backward()  # compute the gradients for every trainable weight in the computational graph
        optimizer.step()  # apply the gradients to weights

        # Calculate the main metric - accuracy

        # Finds the index of maximum (unnormalized) score for every node and that's the class prediction for that node.
        # Compare those to true (ground truth) labels and find the fraction of correct predictions -> accuracy metric.
        class_predictions = torch.argmax(nodes_unnormalized_scores, dim=-1)
        accuracy = torch.sum(torch.eq(class_predictions, gt_node_labels).long()).item() / len(gt_node_labels)

        # Logging
        if phase == LoopPhase.TRAIN:
            # Log metrics
            if config['enable_tensorboard']:
                writer.add_scalar('training_loss', loss.item(), epoch)
                writer.add_scalar('training_acc', accuracy, epoch)

            # Save model checkpoint
            if config['checkpoint_freq'] is not None and (epoch + 1) % config['checkpoint_freq'] == 0:
                config['test_perf'] = -1
                #torch.save(utils.get_training_state(config, gat), os.path.join(CHECKPOINTS_PATH, ckpt_model_name))
            return accuracy, nodes_unnormalized_scores
        else:
            return accuracy, nodes_unnormalized_scores  # in the case of test phase we just report back the test accuracy
    return main_loop #, nodes_unnormalized_scores  # return the decorated function




def train_gat_flmodels(config, node_features, node_labels, edge_index, train_indices):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!
    # Step 2: prepare the model
    gat = GAT(
        num_of_layers=config['num_of_layers'],
        num_heads_per_layer=config['num_heads_per_layer'],
        num_features_per_layer=config['num_features_per_layer'],
        add_skip_connection=config['add_skip_connection'],
        bias=config['bias'],
        dropout=config['dropout'],
        layer_type=config['layer_type'],
        log_attention_weights=False  # no need to store attentions, used only in playground.py for visualizations
    ).to(device)


    # Step 3: Prepare other training related utilities (loss & optimizer and decorator function)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = Adam(gat.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # The decorator function makes things cleaner since there is a lot of redundancy between the train and val loops
    main_loop = get_main_loop(
        config,
        gat,
        loss_fn,
        optimizer,
        node_features,
        node_labels,
        edge_index,
        train_indices)

    # Step 4: Start the training procedure
    for epoch in range(config['num_of_epochs']):
        # Training loop
        #print(f"{epoch}-th epoch is starting...")
        accuracy, nodes_unnormalized_scores = main_loop(phase=LoopPhase.TRAIN, epoch=epoch)
        #print("----------------------------------------------------------------------------------------------")
    return nodes_unnormalized_scores


def get_training_args(num_F):
    parser = argparse.ArgumentParser()

    # Training related
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=1000) # 10000
    parser.add_argument("--patience_period", type=int, help="number of epochs with no improvement on val before terminating", default=1000)
    parser.add_argument("--lr", type=float, help="model learning rate", default=5e-3)
    parser.add_argument("--weight_decay", type=float, help="L2 regularization on model weights", default=5e-4)
    parser.add_argument("--should_test", action='store_true', help='should test the model on the test dataset? (no by default)')

    # Dataset related
    parser.add_argument("--dataset_name", choices=[el.name for el in DatasetType], help='dataset to use for training', default=DatasetType.CORA.name)
    parser.add_argument("--should_visualize", action='store_true', help='should visualize the dataset? (no by default)')

    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--enable_tensorboard", action='store_true', help="enable tensorboard logging (no by default)")
    parser.add_argument("--console_log_freq", type=int, help="log to output console (epoch) freq (None for no logging)", default=100)
    parser.add_argument("--checkpoint_freq", type=int, help="checkpoint model saving (epoch) freq (None for no logging)", default=1000)
    args = parser.parse_args()

    # Model architecture related
    gat_config = {
        "num_of_layers": 2,  # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
        "num_heads_per_layer": [8, 1],
        "ww":[CORA_NUM_INPUT_FEATURES],
        #"num_features_per_layer": [CORA_NUM_INPUT_FEATURES, 8, CORA_NUM_CLASSES],
        "num_features_per_layer": [num_F, 8, 56],  # 98, 196, 392 is control the num of final output features
        "add_skip_connection": False,  # hurts perf on Cora
        "bias": True,  # result is not so sensitive to bias
        "dropout": 0.2,  # result is sensitive to dropout
        "layer_type": LayerType.IMP3  # fastest implementation enabled by default
    }

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)
    # Add additional config information
    training_config.update(gat_config)
    return training_config


def FL_to_GAT(select_local_weights, initialization_graph_matrix):
    N, F = select_local_weights.shape # N number of local models
    node_features = torch.from_numpy(select_local_weights).float().to(device)
    initialization_graph_matrix = torch.from_numpy(initialization_graph_matrix).float().to(device)
    #node_features = torch.from_numpy(select_local_weights).to(torch.float)  # the input select_local_weights is double, that is dtype=torch.float64
    
    #initialization_graph_matrix = torch.from_numpy(initialization_graph_matrix)
    edge_index = initialization_graph_matrix.nonzero().t().contiguous()
    node_labels = torch.tensor([1 for _ in range(N-1)] + [0], device=device) # 0 is surrogate
    train_indices = torch.tensor(list(range(N)), device=device)
    gat_nodes_feature = train_gat_flmodels(get_training_args(F), node_features, node_labels, edge_index, train_indices)
    return gat_nodes_feature.detach()
