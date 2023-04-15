import json
import pickle
import csv
import torch
import math
import random
import timeit
import torch_geometric
from torch.nn import Linear
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import from_smiles
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv,global_mean_pool,global_add_pool
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
from conv import GNN_node,GNN_node_Virtualnode
from torch.nn import TransformerEncoderLayer
import logging


logging.basicConfig(filename= 'test.log', level= logging.DEBUG)

# reg_criterion = torch.nn.L1Loss()

def dump_dictionary(dictionary,filename):
    with open(filename,'wb') as file:
        pickle.dump(dictionary,file)

class GNN(torch.nn.Module):
    def __init__(self,num_tasks = 1,num_layers = 3,emb_dim = 32,window = 11,gnn_type = 'gin',virtual_node = True, residual = False, drop_ratio = 0, JK = "last", graph_pooling = "mean"):
        super(GNN,self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.embed_word = nn.Embedding(n_word, emb_dim)
        self.W_cnn = nn.ModuleList([nn.Conv2d(in_channels=1,out_channels=1,kernel_size=2*window+1,stride=1,padding=window) for _ in range(num_layers)])
        self.W_out = nn.ModuleList([nn.Linear(2*emb_dim,2*emb_dim) for _ in range(num_layers)])
        self.W_interaction = nn.Linear(2*emb_dim,num_tasks)
        self.attention_layers = TransformerEncoderLayer(emb_dim,4)
 
        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layers, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
        else:
            self.gnn_node = GNN_node(num_layers, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
        
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        # if graph_pooling == "set2set":
        #     self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_tasks)
        # else:
        #     self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)
    

    def cnn(self,batched_data):
        # print("1111")
        # print(xs.shape)
        cnn_list = []
        for i in range(len(batched_data)):
            xs = batched_data[i].proteins
            xs = self.embed_word(xs)
            xs = torch.unsqueeze(torch.unsqueeze(xs,0),0)
            for i in range(self.num_layers):
                xs = torch.relu(self.W_cnn[i](xs))
            xs = torch.squeeze(torch.squeeze(xs,0),0)
            
            #Apply attention
            xs = xs.unsqueeze(0)
            xs = xs.permute(1,0,2)
            xs = self.attention_layers(xs)
            xs = xs.permute(1,0,2)
            xs = xs.squeeze(0)
            
            xs = torch.unsqueeze(torch.mean(xs,0),0)
            cnn_list.append(xs)
        cnn_vector = cnn_list[0]
        for i in range(1,len(cnn_list)):
            cnn_vector = torch.cat((cnn_vector,cnn_list[i]),0)
        return cnn_vector

    def forward(self,batched_data):
        # print("!!!")
        # print(batched_data)
        h_node = self.gnn_node(batched_data)
        # print(batched_data.batch)
        # print("h_node shape:",format(h_node.size()))
        h_graph = self.pool(h_node, batched_data.batch)
        protein_vector = self.cnn(batched_data)
        cat_vector = torch.cat((h_graph,protein_vector),1)
        for j in range(self.num_layers):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        output = self.W_interaction(cat_vector)
        return output
        # if self.training:
        #     return output
        # else:
        #     # At inference time, we clamp the value between 0 and 20
        #     return torch.clamp(output, min=0, max=20)

    def __call__(self,batched_data,train=True):
        correct_interaction = batched_data.y
        predicted_interaction = self.forward(batched_data)

        if train:
            loss = F.mse_loss(predicted_interaction,correct_interaction)
            correct_values = correct_interaction.to('cpu').data.numpy()
            predicted_values = predicted_interaction.to('cpu').data.numpy()
            return loss,correct_values,predicted_values
        else:
            correct_values = correct_interaction.to('cpu').data.numpy()
            predicted_values = predicted_interaction.to('cpu').data.numpy()
            return correct_values,predicted_values

class Trainer(object):
    def __init__(self,model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),lr=1e-3,weight_decay=1e-6)

    def train(self,loader):
        #np.random.shuffle(dataset)
        N = len(loader)
        loss_total = 0
        trainCorrect,trainPredict = [],[]
        for data in loader:
            loss,correct_values,predicted_values = self.model(data)
            # print(correct_values,predicted_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
            
            # print(correct_values)
            # print("!!!")
            # print(correct_values)
            # print("!!")
            # print(predicted_values)
            for i in range(len(predicted_values)):
                correct_value = math.log10(math.pow(2,correct_values[i]))
                predicted_value = math.log10(math.pow(2,predicted_values[i]))
                trainCorrect.append(correct_value)
                trainPredict.append(predicted_value)
        rmse_train = np.sqrt(mean_squared_error(trainCorrect,trainPredict) )
        r2_train = r2_score(trainCorrect,trainPredict)
        return loss,rmse_train,r2_train

    

class Tester(object):
    def __init__(self,model):
        self.model = model

    def test(self,loader):
        N = len(loader)
        SAE = 0
        testY, testPredict = [],[]
        for data in loader:
            (correct_values, predicted_values) = self.model(data, train=False)
            for i in range(len(predicted_values)):
                correct_value = math.log10(math.pow(2,correct_values[i]))
                predicted_value = math.log10(math.pow(2,predicted_values[i]))
                SAE += np.abs(predicted_value-correct_value)
                testY.append(correct_value)
                testPredict.append(predicted_value)
        MAE = SAE / 1684 #这是数据集的长度
        rmse = np.sqrt(mean_squared_error(testY,testPredict))
        r2 = r2_score(testY,testPredict)
        return MAE,rmse,r2
    
    def save_MAEs(self,model,filename):
        with open(filename,'a') as f:
            f.write('\t'.join(map(str,MAEs)) + '\n')
    
    def save_model(self,model,filename):
        torch.save(model.state_dict(),filename)

def shuffle_dataset(dataset,seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def split_dataset(dataset,ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n],dataset[n:] 
    return dataset_1,dataset_2

def load_tensor(file_name,dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy',allow_pickle=True)]
    
def load_pickle(file_name):
    with open(file_name,'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Now using GPU!')
    else:
        device = torch.device('cpu')
        print('Now using CPU!')
    # np.random.seed(42)
    # torch.manual_seed(42)
    # torch.cuda.manual_seed(42)
    # random.seed(42)

    dir_input = ('../Data/input/')
    dataset = load_pickle(dir_input + 'graph.pickle')
    word_dict = load_pickle(dir_input + 'sequence_dict.pickle')
    n_word = len(word_dict)
    # print(n_word)

    #print(dataset.is_cuda)
    for a in dataset:
        a.to(device)

    #dataset = list(zip(graph,interactions))
    dataset = shuffle_dataset(dataset, 666)
    # split_idx = dataset.get_idx_split()
    dataset_train, dataset_ = split_dataset(dataset, 0.8)
    dataset_dev, dataset_test = split_dataset(dataset_, 0.5)

    train_loader = torch_geometric.loader.DataLoader(dataset_train,batch_size=32,shuffle=True)
    test_loader = torch_geometric.loader.DataLoader(dataset_test,batch_size=32,shuffle=True)
    dev_loader = torch_geometric.loader.DataLoader(dataset_dev,batch_size=32,shuffle=True)
    print("train dataset:{}",format(len(dataset_train)))
    print("test dataset:{}",format(len(dataset_test)))
    print("dev dataset:{}",format(len(dataset_dev)))
    print("train loader:{}",format(len(train_loader)))
    print("test loader:{}",format(len(test_loader)))
    print("dev loader:{}",format(len(dev_loader)))
   
    # train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    # valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    # if args.save_test_dir != '':
    #     testdev_loader = DataLoader(dataset[split_idx["test-dev"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    # if args.checkpoint_dir != '':
    #     os.makedirs(args.checkpoint_dir, exist_ok = True)

    # shared_params = {
    #     'num_layers': args.num_layers,
    #     'emb_dim': args.emb_dim,
    #     'drop_ratio': args.drop_ratio,
    #     'graph_pooling': args.graph_pooling
    # }

    torch.manual_seed(666)
    model = GNN().to(device)
    trainer = Trainer(model)
    tester = Tester(model)

    #  if args.gnn == 'gin':
    #     model = GNN(gnn_type = 'gin', virtual_node = False, **shared_params).to(device)
    # elif args.gnn == 'gin-virtual':
    #     model = GNN(gnn_type = 'gin', virtual_node = True, **shared_params).to(device)
    # elif args.gnn == 'gcn':
    #     model = GNN(gnn_type = 'gcn', virtual_node = False, **shared_params).to(device)
    # elif args.gnn == 'gcn-virtual':
    #     model = GNN(gnn_type = 'gcn', virtual_node = True, **shared_params).to(device)
    # else:
    #     raise ValueError('Invalid GNN type')

    # num_params = sum(p.numel() for p in model.parameters())
    # print(f'#Params: {num_params}')

    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    setting = 'test1'
    file_MAEs = '../Data/Results/output/MAEs--' + setting + '.txt'
    file_model = '../Data/Results/output/' + setting
    MAEs = ('Epoch\tTime(sec)\tRMSE_train\tR2_train\tMAE_dev\tMAE_test\tRMSE_dev\tRMSE_test\tR2_dev\tR2_test')
    with open(file_MAEs, 'w') as f:
        f.write(MAEs + '\n')

    print('Training...')
    print(MAEs)
    start = timeit.default_timer()

    for epoch in range(1,100):
        loss_train, rmse_train, r2_train = trainer.train(train_loader)
        MAE_dev, RMSE_dev, R2_dev = tester.test(dev_loader)
        MAE_test, RMSE_test, R2_test = tester.test(test_loader)

        end = timeit.default_timer()
        time = end - start

        MAEs = [epoch, time, rmse_train, r2_train, MAE_dev,
                MAE_test, RMSE_dev, RMSE_test, R2_dev, R2_test]
        tester.save_MAEs(MAEs, file_MAEs)
        tester.save_model(model, file_model)

        print('\t'.join(map(str, MAEs)))

        
    


