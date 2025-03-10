
import torch
import numpy as np
from sklearn.cluster import KMeans
import joblib
import os



def build_combined_adjacency_matrix(features, coordinates, alpha=0.5,gamma=0.1,k = 3):
    """
    Construct neighbor matrix based on coordinate and feature similarity.
    Args:
        features (torch.Tensor): Instance feature matrix with shape (N, F)
        coordinates (torch.Tensor): Instance coordinates with shape (N, 2) or (N, 3)
        alpha (float): Weighting factor balancing feature similarity and spatial distance
        k (int): Number of nearest neighbors to select for each instance
    Returns:
        torch.Tensor: Adjacency matrix with shape (N, N)
    """
    coordinates = coordinates.squeeze(0)  # (N, 2)
    num_nodes = coordinates.size(0)


    coordinates = coordinates.float()  
    
    diff = coordinates.unsqueeze(1) - coordinates.unsqueeze(0)

    distance_matrix = torch.norm(diff, dim=2)  

    distance_matrix = distance_matrix / torch.max(distance_matrix)

    spatial_similarity = torch.exp(-gamma * distance_matrix**2) 

    _, top_k_indices = torch.topk(spatial_similarity, k=k + 1, dim=1)
    adjacency_matrix = torch.zeros((num_nodes, num_nodes), device=features.device)
    for i in range(num_nodes):
        adjacency_matrix[i, top_k_indices[i, 1:]] = spatial_similarity[i, top_k_indices[i, 1:]]
    
    
    return adjacency_matrix

def save_random_group(args,test_loader): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for idx, (coords, data, label,data_dir) in enumerate (test_loader):
            # instancetoken = []

            file_path = f'{data_dir[0]}_{args.num_subbags}_random.pkl'
            if os.path.exists(file_path):
                print(f"File {file_path} already exists. Skipping...")
                continue
    
            update_coords, update_data, label = coords.to(device), data.to(device), label.to(device).long()
            grouping_instance = grouping(args.num_subbags,max_size=4300,group_size=args.group_size) 
            features_group,coords_group = grouping_instance.random_grouping(update_coords,update_data)
            graph_adjacency_matrix =  []
            for patch_step in range(0, len(features_group)):
                adjacency_matrix = build_combined_adjacency_matrix(features_group[patch_step], coords_group[patch_step], alpha=0.5,gamma=0.1,k = 3)
                graph_adjacency_matrix.append(adjacency_matrix)
            
            model_data = {
            'features_group': features_group,  
            'coords_group': coords_group,  
            'graph_adjacency_matrix': graph_adjacency_matrix, 
            'names': f'{data_dir[0]}_{args.num_subbags}_group_graoh.pkl'                  
            }
    

            joblib.dump(model_data, f'{data_dir[0]}_{args.num_subbags}_group_graoh.pkl')



def save_kmeans_features(args,test_loader): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for idx, (coords, data, label,data_dir) in enumerate (test_loader):



            file_path = f'{data_dir[0]}_{args.num_subbags}_featurekm.pkl'
            if os.path.exists(file_path):
                print(f"File {file_path} already exists. Skipping...")
                continue
    
            update_coords, update_data, label = coords.to(device), data.to(device), label.to(device).long()
            grouping_instance = grouping(args.num_subbags,max_size=4300,group_size=args.group_size) 
            features_group,coords_group = grouping_instance.embedding_grouping(update_coords,update_data)
            graph_adjacency_matrix =  []
            for patch_step in range(0, len(features_group)):
                adjacency_matrix = build_combined_adjacency_matrix(features_group[patch_step], coords_group[patch_step], alpha=0.5,gamma=0.1,k = 3)
                graph_adjacency_matrix.append(adjacency_matrix)
            
            model_data = {
            'features_group': features_group,  
            'coords_group': coords_group, 
            'graph_adjacency_matrix': graph_adjacency_matrix,  
            'names': f'{data_dir[0]}_{args.num_subbags}_group_graoh.pkl'                  
            }
    
            joblib.dump(model_data, f'{data_dir[0]}_{args.num_subbags}_group_graoh.pkl')


def save_kmeans_coord(args,test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for idx, (coords, data, label,data_dir) in enumerate (test_loader):
            # instancetoken = []

            file_path = f'{data_dir[0]}_{args.num_subbags}_group_graoh.pkl'
            if os.path.exists(file_path):
                print(f"File {file_path} already exists. Skipping...")
                continue
    
            update_coords, update_data, label = coords.to(device), data.to(device), label.to(device).long()
            grouping_instance = grouping(args.num_subbags,max_size=4300,group_size=args.group_size) 
            features_group,coords_group = grouping_instance.coords_grouping(update_coords,update_data)
            
            graph_adjacency_matrix =  []
            for patch_step in range(0, len(features_group)):
                adjacency_matrix = build_combined_adjacency_matrix(features_group[patch_step], coords_group[patch_step], alpha=0.5,gamma=0.1,k = 3)
                graph_adjacency_matrix.append(adjacency_matrix)
            
            model_data = {
            'features_group': features_group,  
            'coords_group': coords_group, 
            'graph_adjacency_matrix': graph_adjacency_matrix,  
            'names': f'{data_dir[0]}_{args.num_subbags}_group_graoh.pkl'                 
            }
    
           
            joblib.dump(model_data, f'{data_dir[0]}_{args.num_subbags}_group_graoh.pkl')

def save_seqential_coord(self, features):
        B, N, C = features.shape
        features = features.squeeze()
        indices = np.array_split(range(N),self.groups_num)
        features_group = self.make_subbags(indices,features)
        
        return features_group


def seed_torch(seed=2021):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 


def split_array(array, m):
    n = len(array)
    indices = np.random.choice(n, n, replace=False) 
    split_indices = np.array_split(indices, m)  

    result = []
    for indices in split_indices:
        result.append(array[indices])
    return result

class grouping:

    def __init__(self,groups_num,max_size=1e10,group_size = 128):
        self.groups_num = groups_num
        self.max_size = int(max_size)
        self.group_size = group_size
        self.action_std = 0.1
        
    
    def indicer(self, labels):
        indices = []
        groups_num = len(set(labels))
        for i in range(groups_num):
            temp = np.argwhere(labels==i).squeeze()
            indices.append(temp)
        return indices
    
    def make_subbags(self, idx, features):
        index = idx
        features_group = []
        for i in range(len(index)):
            member_size = (index[i].size)
            if member_size > self.max_size:  
                index[i] = np.random.choice(index[i],size=self.max_size,replace=False)
            temp = features[index[i]]
            temp = temp.unsqueeze(dim=0)  
            features_group.append(temp)
            
        return features_group
    
        
    def coords_nomlize(self, coords):
        coords = coords.squeeze()
        means = torch.mean(coords,0)
        xmean,ymean = means[0],means[1]
        stds = torch.std(coords,0)
        xstd,ystd = stds[0],stds[1]
        xcoords = (coords[:,0] - xmean)/xstd
        ycoords = (coords[:,1] - ymean)/ystd
        xcoords,ycoords = xcoords.view(xcoords.shape[0],1),ycoords.view(ycoords.shape[0],1)
        coords = torch.cat((xcoords,ycoords),dim=1)
        
        return coords
    
    
    def coords_grouping(self,coords,features,c_norm=False):
        features = features.squeeze()
        coords = coords.squeeze()
        if c_norm:
            coords = self.coords_nomlize(coords.float())
        features = features.squeeze()
        k = KMeans(n_clusters=self.groups_num, random_state=0, n_init='auto').fit(coords.cpu().numpy())
        indices = self.indicer(k.labels_)
        features_group = self.make_subbags(indices,features)
        coords_group = self.make_subbags(indices,coords)
        
        return features_group,coords_group
    
    def embedding_grouping(self,coords,features,):
        features = features.squeeze()
        k = KMeans(n_clusters=self.groups_num, random_state=0,n_init='auto').fit(features.cpu().detach().numpy())
        indices = self.indicer(k.labels_)
        features_group = self.make_subbags(indices,features)
        coords_group = self.make_subbags(indices,coords)
        return features_group,coords_group
    
    def random_grouping(self, coords,features,):
        B, N, C = features.shape
        features = features.squeeze() 
        indices = split_array(np.array(range(int(N))),self.groups_num)
        features_group = self.make_subbags(indices,features) 
        coords_group = self.make_subbags(indices,coords)
        return features_group,coords_group
        
    def seqential_grouping(self, coords,features,):
        B, N, C = features.shape
        features = features.squeeze()
        indices = np.array_split(range(N),self.groups_num)
        features_group = self.make_subbags(indices,features)
        coords_group = self.make_subbags(indices,coords)
        return features_group,coords_group

    def idx_grouping(self,idx,coords,features,):
        idx = idx.cpu().numpy()
        idx = idx.reshape(-1)
        B, N, C = features.shape
        features = features.squeeze()
        indices = self.indicer(idx)
        features_group = self.make_subbags(indices,features)
        coords_group = self.make_subbags(indices,coords)
        return features_group,coords_group
    
    
    