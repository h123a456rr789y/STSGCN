from torch.utils.data import Dataset
import numpy as np
from h5py import File
import scipy.io as sio
from utils import data_utils
from matplotlib import pyplot as plt
import torch

import os 

class TotalCap3D(Dataset):

    def __init__(self, data_dir,input_n,output_n,skip_rate, split=0,inertia_thres=70):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        self.path_to_data = os.path.join(data_dir,'TotalCapture/dataset')
        self.split = split
        self.in_n = input_n
        self.out_n = output_n
        self.sample_rate = 2
        self.p3d = {}
        self.data_idx = []
        
        subs = np.array([[1, 2, 4], [3], [5]], dtype=object)
        acts = data_utils.define_actions_TotalCap(actions)


        subjs = subs[split]
        all_seqs, dim_used= data_utils.load_data_totalcap_3d(path_to_data, subjs, acts, sample_rate, input_n + output_n,test_manner)
        
        ## Select the huge movement (inertia_changes) clips for testing  
        if split==1:
            selected_clips=data_utils.select_inertia_changes_clips(all_seqs,threshold=inertia_thres)
            all_seqs=all_seqs[selected_clips,:,:]

        self.all_seqs = all_seqs
        self.dim_used = dim_used
        self.shape = all_seqs.shape


        all_seqs = all_seqs[:, :, dim_used]
        all_seqs = all_seqs.transpose(0, 2, 1)
        all_seqs = all_seqs.reshape(-1, input_n + output_n)
        all_seqs = all_seqs.transpose()

        dct_m_in, _ = data_utils.get_dct_matrix(input_n + output_n)
        dct_m_out, _ = data_utils.get_dct_matrix(input_n + output_n)
        pad_idx = np.repeat([input_n - 1], output_n)
        i_idx = np.append(np.arange(0, input_n), pad_idx)
        input_dct_seq = np.matmul(dct_m_in[0:dct_used, :], all_seqs[i_idx, :])
        input_dct_seq = input_dct_seq.transpose().reshape([-1, len(dim_used), dct_used])

        output_dct_seq = np.matmul(dct_m_out[0:dct_used, :], all_seqs)
        output_dct_seq = output_dct_seq.transpose().reshape([-1, len(dim_used), dct_used])


        self.input_dct_seq = input_dct_seq
        self.output_dct_seq = output_dct_seq

    def __len__(self):
        return np.shape(self.input_dct_seq)[0]

    def __getitem__(self, item):
        return self.input_dct_seq[item], self.output_dct_seq[item], self.all_seqs[item]
