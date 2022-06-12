from torch.utils.data import Dataset
import numpy as np
from h5py import File
import scipy.io as sio
from utils import data_utils
from matplotlib import pyplot as plt
import torch

from torch.autograd.variable import Variable
import os 

class TotalCap3D(Dataset):

    def __init__(self, data_dir,input_n,output_n,skip_rate, split=0, actions=None):
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
        seq_len = self.in_n + self.out_n

        subs = np.array([[1, 2, 4], [3], [5]], dtype=object)
        
        if actions is None:
            acts = ["acting", "freestyle", "rom", "walking"]
        else:
            acts = actions
        
        


        subs = subs[split]
        #all_seqs, dim_used= data_utils.load_data_totalcap_3d(self.path_to_data, subs, acts, self.sample_rate, input_n + output_n)
        key=0
        for subj in subs:
            for action_idx in np.arange(len(acts)):
                action = acts[action_idx]
                if(not (subj == 5) and not (subj ==4) ): # s1, s2,s3
                    for subact in [1, 2, 3]:  # subactions
                        print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))

                        filename = '{0}/s{1}/{2}{3}/gt_skel_gbl_pos.txt'.format(self.path_to_data, subj, action, subact)
                        action_sequence = data_utils.readCSVasFloat_TotalCap(filename)
                        n, d = action_sequence.shape
                        even_list = range(0, n, self.sample_rate)
                        num_frames = len(even_list)
                        the_sequence = np.array(action_sequence[even_list, :])
                        the_seq = torch.from_numpy(the_sequence).float().cuda()
                        self.p3d[key] = the_seq.view(num_frames, -1).cpu().data.numpy()

                        valid_frames = np.arange(0, num_frames - seq_len + 1, skip_rate)
                        tmp_data_idx_1 = [key] * len(valid_frames)
                        tmp_data_idx_2 = list(valid_frames)
                        self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                        key += 1

                else: ## s4 s5 
                    if(action == ("acting") or action == ("rom")):
                        print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 3))
                        filename = '{0}/s{1}/{2}{3}/gt_skel_gbl_pos.txt'.format(self.path_to_data, subj, action, 3)
                        action_sequence = data_utils.readCSVasFloat_TotalCap(filename)
                        n, d = action_sequence.shape
                        even_list = range(0, n, self.sample_rate)
                        num_frames = len(even_list)
                        the_sequence = np.array(action_sequence[even_list, :])
                        the_seq = torch.from_numpy(the_sequence).float().cuda()
                        
                        self.p3d[key] = the_seq.view(num_frames, -1).cpu().data.numpy()
                        valid_frames = np.arange(0, num_frames - seq_len + 1, skip_rate)
                        tmp_data_idx_1 = [key] * len(valid_frames)
                        tmp_data_idx_2 = list(valid_frames)
                        self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                        key += 1
                        
                    elif(action == "freestyle"):
                        for subact in [1,3]: # subactions
                            print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))
                            filename = '{0}/s{1}/{2}{3}/gt_skel_gbl_pos.txt'.format(self.path_to_data, subj, action, subact)
                            action_sequence = data_utils.readCSVasFloat_TotalCap(filename)
                            n, d = action_sequence.shape
                            even_list = range(0, n, self.sample_rate)
                            num_frames = len(even_list)
                            the_sequence = np.array(action_sequence[even_list, :])
                            the_seq = torch.from_numpy(the_sequence).float().cuda()

                            self.p3d[key] = the_seq.view(num_frames, -1).cpu().data.numpy()
                            valid_frames = np.arange(0, num_frames - seq_len + 1, skip_rate)
                            tmp_data_idx_1 = [key] * len(valid_frames)
                            tmp_data_idx_2 = list(valid_frames)
                            self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                            key += 1
                            
                    else:### (action =="walking"):
                        print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 2))
                        filename = '{0}/s{1}/{2}{3}/gt_skel_gbl_pos.txt'.format(self.path_to_data, subj, action, 2)
                        action_sequence = data_utils.readCSVasFloat_TotalCap(filename)
                        n, d = action_sequence.shape
                        even_list = range(0, n, self.sample_rate)
                        num_frames = len(even_list)
                        the_sequence = np.array(action_sequence[even_list, :])
                        the_seq = torch.from_numpy(the_sequence).float().cuda()

                        self.p3d[key] = the_seq.view(num_frames, -1).cpu().data.numpy()
                        valid_frames = np.arange(0, num_frames - seq_len + 1, skip_rate)
                        tmp_data_idx_1 = [key] * len(valid_frames)
                        tmp_data_idx_2 = list(valid_frames)
                        self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                        key += 1
                

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        return self.p3d[key][fs]
