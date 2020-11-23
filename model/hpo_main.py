__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '3.6'
__status__ = "Research"
__date__ = "1/12/2018"
__license__= "MIT License"

import torch
from torchvision import transforms
import numpy as np
import time
import glob
import random
import argparse
import h5py
import json
import torch.nn.init as init

from config import  *
from sys_utils import *
from vsum_tools import  *
from model.vasnet_model import  *
from hyperopt import fmin, atpe, space_eval, hp, STATUS_OK, Trials
import pickle

hps_args = {'root': '', 'datasets': None, 'splits': None, 'train': True, 'verbose': False, 'output_dir': 'art_hpo', 'change_point': False, 'model': 'art_hpo'}
print_pkg_versions()
hps = HParameters()
hps.load_from_args(hps_args)
print("Parameters:")
print("----------------------------------------------------------------------")
print(hps)

def weights_init(m):
    classname = m.__class__.__name__
    if classname == 'Linear':
        init.xavier_uniform_(m.weight, gain=np.sqrt(2.0))
        if m.bias is not None:
            init.constant_(m.bias, 0.1)

def parse_splits_filename(splits_filename):
    # Parse split file and count number of k_folds
    spath, sfname = os.path.split(splits_filename)
    sfname, _ = os.path.splitext(sfname)
    dataset_name = sfname.split('_')[0]  # Get dataset name e.g. tvsum
    dataset_type = sfname.split('_')[1]  # augmentation type e.g. aug

    # The keyword 'splits' is used as the filename fields terminator from historical reasons.
    if dataset_type == 'splits':
        # Split type is not present
        dataset_type = ''

    # Get number of discrete splits within each split json file
    with open(splits_filename, 'r') as sf:
        splits = json.load(sf)

    return dataset_name, dataset_type, splits

def lookup_weights_splits_file(path, dataset_name, dataset_type, split_id):
    dataset_type_str = '' if dataset_type == '' else dataset_type + '_'
    weights_filename = path + '/models/{}_{}splits_{}_*.tar.pth'.format(dataset_name, dataset_type_str, split_id)
    weights_filename = glob.glob(weights_filename)
    if len(weights_filename) == 0:
        print("Couldn't find model weights: ", weights_filename)
        return ''

    # Get the first weights file in the dir
    weights_filename = weights_filename[0]
    splits_file = path + '/splits/{}_{}splits.json'.format(dataset_name, dataset_type_str)

    return weights_filename, splits_file


class AONet:

    def __init__(self, hps: HParameters):
        self.hps = hps
        self.model = None
        self.log_file = None
        self.verbose = hps.verbose


    def fix_keys(self, keys, dataset_name = None):
        """
        :param keys:
        :return:
        """
        # dataset_name = None
        if len(self.datasets) == 1:
            dataset_name = next(iter(self.datasets))

        keys_out = []
        for key in keys:
            t = key.split('/')
            if len(t) != 2:
                assert dataset_name is not None, "ERROR dataset name in some keys is missing but there are multiple dataset {} to choose from".format(len(self.datasets))

                key_name = dataset_name+'/'+key
                keys_out.append(key_name)
            else:
                keys_out.append(key)

        return keys_out


    def load_datasets(self, datasets = None):
        """
        Loads all h5 datasets from the datasets list into a dictionary self.dataset
        referenced by their base filename
        :param datasets:  List of dataset filenames
        :return:
        """
        if datasets is None:
            datasets = self.hps.datasets

        datasets_dict = {}
        for dataset in datasets:
            _, base_filename = os.path.split(dataset)
            base_filename, _ = os.path.splitext(base_filename)
            print("Loading:", dataset)
            # dataset_name = base_filename.split('_')[2]
            # print("\tDataset name:", dataset_name)
            datasets_dict[base_filename] = h5py.File(dataset, 'r')

        self.datasets = datasets_dict
        return datasets_dict


    def load_split_file(self, splits_file):

        self.dataset_name, self.dataset_type, self.splits = parse_splits_filename(splits_file)
        n_folds = len(self.splits)
        self.split_file = splits_file
        print("Loading splits from: ",splits_file)

        return n_folds


    def select_split(self, split_id):
        print("Selecting split: ",split_id)

        self.split_id = split_id
        n_folds = len(self.splits)
        assert self.split_id < n_folds, "split_id (got {}) exceeds {}".format(self.split_id, n_folds)

        split = self.splits[self.split_id]
        #print(split)
        self.train_keys = split['train_keys']
        self.test_keys = split['test_keys']
        #print(self.hps.get_dataset_by_name(self.dataset_name))
        #print(self.hps)
        #print(self.dataset_name)
        dataset_filename = self.hps.get_dataset_by_name(self.dataset_name)[0]
        _,dataset_filename = os.path.split(dataset_filename)
        dataset_filename,_ = os.path.splitext(dataset_filename)
        self.train_keys = self.fix_keys(self.train_keys, dataset_filename)
        self.test_keys = self.fix_keys(self.test_keys, dataset_filename)
        return



    def load_model(self, model_filename):
        self.model.load_state_dict(torch.load(model_filename, map_location=lambda storage, loc: storage))
        return


    def initialize(self, cuda_device=None, drop_in=0.5, drop_out=0.5, val_q=0.06, apperture=20):
        rnd_seed = 12345
        # random.seed(rnd_seed)
        np.random.seed(rnd_seed)
        torch.manual_seed(rnd_seed)
        self.model = hps.model(drop_in=drop_in, drop_out=drop_out, val_q=val_q, apperture=apperture)
        self.model.eval()
        self.model.apply(weights_init)

        cuda_device = cuda_device or self.hps.cuda_device

        if self.hps.use_cuda:
            print("Setting CUDA device: ",cuda_device)
            torch.cuda.set_device(cuda_device)
            torch.cuda.manual_seed(rnd_seed)

        if self.hps.use_cuda:
            self.model.cuda()

        return


    def get_data(self, key):
        key_parts = key.split('/')
        assert len(key_parts) == 2, "ERROR. Wrong key name: "+key
        dataset, key = key_parts
        #print(f"self.datasets[dataset][key] = {self.datasets[dataset][key]}")
        return self.datasets[dataset][key]

    def lookup_weights_file(self, data_path):
        dataset_type_str = '' if self.dataset_type == '' else self.dataset_type + '_'
        weights_filename = data_path + '/models/{}_{}splits_{}_*.tar.pth'.format(self.dataset_name, dataset_type_str, self.split_id)
        weights_filename = glob.glob(weights_filename)
        if len(weights_filename) == 0:
            print("Couldn't find model weights: ", weights_filename)
            return ''

        # Get the first weights filename in the dir
        weights_filename = weights_filename[0]
        splits_file = data_path + '/splits/{}_{}splits.json'.format(self.dataset_name, dataset_type_str)

        return weights_filename, splits_file


    def train(self, output_dir='EX-0', lr=0.00005):

        print("Initializing VASNet model and optimizer...")
        self.model.train()

        criterion = nn.MSELoss()

        if self.hps.use_cuda:
            criterion = criterion.cuda()

        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=self.hps.l2_req)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 10, 2)
        print("Starting training...")

        max_val_fscore = 0
        max_val_fscore_epoch = 0
        train_keys = self.train_keys[:]

        for epoch in range(self.hps.epochs_max):

            print("Epoch: {0:6}".format(str(epoch)+"/"+str(self.hps.epochs_max)), end='')
            self.model.train()
            avg_loss = []
            iter_num = len(train_keys)
            random.shuffle(train_keys)
            #print("Am i get ignored?")
            for i, key in enumerate(train_keys):
                dataset = self.get_data(key)
                y, _, target = self.model.train_wrapper(self.hps, dataset)
                loss_att = 0
                #print(np.where(target>0))
                #print("y:"+str(y))
                loss = criterion(y, target)
                #print("loss:"+str(loss))
                # loss2 = y.sum()/seq_len
                loss = loss + loss_att
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step(epoch+i/iter_num)
                avg_loss.append([float(loss), float(loss_att)])

            # Evaluate test dataset
            val_fscore, video_scores = self.eval(self.test_keys)
            if max_val_fscore < val_fscore:
                max_val_fscore = val_fscore
                max_val_fscore_epoch = epoch

            avg_loss = np.array(avg_loss)
            #print("avg_loss")
            #print(avg_loss)
            print("   Train loss: {0:.05f}".format(np.mean(avg_loss[:, 0])), end='')
            print('   Test F-score avg/max:{0:0.5}/{1:0.5}'.format(float(val_fscore),float(max_val_fscore)))

            if self.verbose:
                video_scores = [["No", "Video", "F-score"]] + video_scores
                print_table(video_scores, cell_width=[3,40,8])

            # Save model weights
            path, filename = os.path.split(self.split_file)
            base_filename, _ = os.path.splitext(filename)
            path = os.path.join(output_dir, 'models_temp', base_filename+'_'+str(self.split_id))
            os.makedirs(path, exist_ok=True)
            filename = str(epoch)+'_'+str(round(val_fscore*100,3))+'.pth.tar'
            torch.save(self.model.state_dict(), os.path.join(path, filename))
        
        return max_val_fscore, max_val_fscore_epoch


    def eval(self, keys, results_filename=None):

        self.model.eval()
        summary = {}
        att_vecs = {}
        with torch.no_grad():
            for i, key in enumerate(keys):
                dataset = self.get_data(key)
                y, att_vec = self.model.eval_wrapper(self.hps, dataset)
                summary[key] = y[0].detach().cpu().numpy()
                att_vecs[key] = att_vec.detach().cpu().numpy()

        f_score, video_scores = self.eval_summary(summary, keys, metric=self.dataset_name,
                    results_filename=results_filename, att_vecs=att_vecs)

        return f_score, video_scores


    def eval_summary(self, machine_summary_activations, test_keys, results_filename=None, metric='tvsum', att_vecs=None):

        eval_metric = 'avg' if metric == 'tvsum' else 'max'
        print(f"{results_filename} will be generated")
        if results_filename is not None:
            h5_res = h5py.File(results_filename, 'w')

        fms = []
        video_scores = []
        for key_idx, key in enumerate(test_keys):
            d = self.get_data(key)
            probs = machine_summary_activations[key]

            if 'change_point' not in d:
                print("ERROR: No change points in dataset/video ",key)

            cps = d['change_point'][...]
            num_frames = d['n_frames'][()]
            nfps = d['n_frame_per_seg'][...].tolist()
            positions = d['picks'][...]
            user_summary = d['user_summary'][...]

            machine_summary = generate_summary(probs, cps, num_frames,nfps,positions,proportion = 0.15)
            fm, _, _ = evaluate_summary(machine_summary, user_summary, eval_metric)
            fms.append(fm)

            # Reporting & logging
            video_scores.append([key_idx + 1, key, "{:.1%}".format(fm)])

            if results_filename is not None:
                gt = d['gtscore'][...]
                h5_res.create_dataset(key + '/score', data=probs)
                h5_res.create_dataset(key + '/machine_summary', data=machine_summary)
                h5_res.create_dataset(key + '/gtscore', data=gt)
                h5_res.create_dataset(key + '/fm', data=fm)
                h5_res.create_dataset(key + '/picks', data=positions)

                video_name = key.split('/')[1]
                if 'video_name' in d:
                    video_name = d['video_name'][...]
                h5_res.create_dataset(key + '/video_name', data=video_name)

                if att_vecs is not None:
                    h5_res.create_dataset(key + '/att', data=att_vecs[key])

        mean_fm = np.mean(fms)

        # Reporting & logging
        if results_filename is not None:
            h5_res.close()

        return mean_fm, video_scores


#==============================================================================================



def eval_split(splits_filename, data_dir='test'):

    print("\n")
    ao = AONet(hps)
    ao.initialize()
    ao.load_datasets()
    ao.load_split_file(splits_filename)
    result_filename = splits_filename
    result_filename = result_filename.replace("splits.json","")
    val_fscores = []
    for split_id in range(len(ao.splits)):
        ao.select_split(split_id)
        weights_filename, _ = ao.lookup_weights_file(data_dir)
        print("Loading model:", weights_filename)
        ao.load_model(weights_filename)
        val_fscore,video_scores=ao.eval(ao.test_keys,data_dir+'/'+result_filename+str(split_id)+".h5")
        val_fscores.append(val_fscore)

        val_fscore_avg = np.mean(val_fscores)

        if hps.verbose:
            video_scores = [["No.", "Video", "F-score"]] + video_scores
            print_table(video_scores, cell_width=[4,45,5])

        print("Avg F-score: ", val_fscore)
        print("")

    print("Total AVG F-score: ", val_fscore_avg)
    return val_fscore_avg


def train(lr, drop_in, drop_out, val_q, apperture):
    os.makedirs(hps.output_dir, exist_ok=True)
    os.makedirs(os.path.join(hps.output_dir, 'splits'), exist_ok=True)
    os.makedirs(os.path.join(hps.output_dir, 'code'), exist_ok=True)
    os.makedirs(os.path.join(hps.output_dir, 'models'), exist_ok=True)
    os.system('cp -f splits/*.json  ' + hps.output_dir + '/splits/')
    os.system('cp *.py ' + hps.output_dir + '/code/')

    # Create a file to collect results from all splits
    f = open(hps.output_dir + '/results.txt', 'wt')

    for split_filename in hps.splits: # Normally there is only one element in list. for-loop is meaningless.
        dataset_name, dataset_type, splits = parse_splits_filename(split_filename)
        #print(f'split_filename is {split_filename}')
        #print(f'dataset_name is {dataset_name}')
        # For no augmentation use only a dataset corresponding to the split file
        datasets = None
        if dataset_type == '':
            datasets = hps.get_dataset_by_name(dataset_name)
        #print(f"{datasets} is not null")
        if datasets is None:
            datasets = hps.datasets
        #print(f"current dataset : {datasets}")
        f_avg = 0
        # n_folds = len(splits)
        n_folds = 1
        for split_id in range(n_folds):
            ao = AONet(hps) # Make model
            ao.initialize(drop_in=drop_in, drop_out=drop_out, val_q=val_q, apperture=apperture)
            ao.load_datasets(datasets=datasets)
            ao.load_split_file(splits_file=split_filename)
            ao.select_split(split_id=split_id)

            fscore, fscore_epoch = ao.train(output_dir=hps.output_dir, lr=lr)
            f_avg += fscore

            # Log F-score for this split_id
            f.write(split_filename + ', ' + str(split_id) + ', ' + str(fscore) + ', ' + str(fscore_epoch) + '\n')
            f.flush()

            # Save model with the highest F score
            _, log_file = os.path.split(split_filename)
            log_dir, _ = os.path.splitext(log_file)
            log_dir += '_' + str(split_id)
            log_file = os.path.join(hps.output_dir, 'models', log_dir) + '_' + str(fscore) + '.tar.pth'

            os.makedirs(os.path.join(hps.output_dir, 'models', ), exist_ok=True)
            os.system('mv ' + hps.output_dir + '/models_temp/' + log_dir + '/' + str(fscore_epoch) + '_*.pth.tar ' + log_file)
            os.system('rm -rf ' + hps.output_dir + '/models_temp/' + log_dir)

            print("Split: {0:}   Best F-score: {1:0.5f}   Model: {2:}".format(split_filename, fscore, log_file))

        # Write average F-score for all splits to the results.txt file
        f_avg /= n_folds
        f.write(split_filename + ', ' + str('avg') + ', ' + str(f_avg) + '\n')
        f.flush()

    f.close()
    return fscore


def objective_func(val):
    arrperture = int(val['apperture'])
    lr = 2**-int(val['lr'])
    fscore = train(lr, val['drop_in'], val['drop_out'], val['val_q'], arrperture)

    return {
        'loss' : 1 - fscore,
        'status' : STATUS_OK
    } 