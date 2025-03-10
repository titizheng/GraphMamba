import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

from datasets.group_graph_core import save_kmeans_coord,save_kmeans_features ,save_seqential_coord,save_random_group
from torch.utils.data import DataLoader
from datasets.toloda_datasets import h5file_Dataset
import argparse

def DPRC_test_make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', default='TCGA',type=str) 
    parser.add_argument('--mode', default='sequence',type=str,choices=['sequence','random','kmeans_coords']) 
    parser.add_argument('--in_chans', default=512,type=int) 

    parser.add_argument('--num_subbags', default=10,type=int) 
    parser.add_argument('--embed_dim', default=512,type=int)

    parser.add_argument('--mambamil_rate',type=int, default=10, help='mambamil_rate')
    parser.add_argument('--mambamil_layer',type=int, default=1, help='mambamil_layer')
    parser.add_argument('--mambamil_type',type=str, default='BiMamba', choices= ['Mamba', 'BiMamba', 'SRMamba'], help='mambamil_type')
    parser.add_argument('--drop_out', type=float, default=0.25, help='enable dropout (p=0.25)')

    parser.add_argument('--save_dir', default='',type=str)
    parser.add_argument('--h5',default='',type=str)
    parser.add_argument('--csv', default='',type=str)
    parser.add_argument('--group_size', type=int, default=512) 

    args = parser.parse_args()

    return args



def select_main(args):
    import pandas as pd
    import shutil
    

    flod = 'flod0'
    args.h5 = '/feature/h5_files'
    args.csv = f'/five_flod_csv/patient_labels_{flod}.csv'
    args.num_subbags = 10

    test_dataset = h5file_Dataset(args.csv,args.h5,'test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    if args.mode == 'kmeans_coords':
        save_kmeans_coord(args,test_dataloader ) 
    elif args.mode == 'kmeans_features':
        save_kmeans_features(args,test_dataloader )
    elif args.mode == 'sequence':
        save_seqential_coord(args,test_dataloader )
    elif args.mode == 'random':
        save_random_group(args,test_dataloader )

    


if __name__ == "__main__":

    args = DPRC_test_make_parse()
    select_main(args)