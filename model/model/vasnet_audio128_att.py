__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '3.6'
__status__ = "Research"
__date__ = "1/12/2018"
__license__= "MIT License"


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from config import  *
from model.layer_norm import  *




class SelfAttention(nn.Module):

    def __init__(self, apperture=20, ignore_itself=False, input_size=1024, output_size=1024):
        super(SelfAttention, self).__init__()

        self.apperture = apperture
        self.ignore_itself = ignore_itself

        self.m = input_size
        self.output_size = output_size

        self.K = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.Q = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.V = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.output_linear = nn.Linear(in_features=self.output_size, out_features=self.m, bias=False)

        self.drop50 = nn.Dropout(0.5)



    def forward(self, x):
        n = x.shape[0]  # sequence length

        K = self.K(x)  # ENC (n x m) => (n x H) H= hidden size
        Q = self.Q(x)  # ENC (n x m) => (n x H) H= hidden size
        V = self.V(x)

        Q *= 0.06
        logits = torch.matmul(Q, K.transpose(1,0))

        if self.ignore_itself:
            # Zero the diagonal activations (a distance of each frame with itself)
            logits[torch.eye(n).byte()] = -float("Inf")

        if self.apperture > 0:
            # Set attention to zero to frames further than +/- apperture from the current one
            onesmask = torch.ones(n, n)
            trimask = torch.tril(onesmask, -self.apperture) + torch.triu(onesmask, self.apperture)
            logits[trimask == 1] = -float("Inf")

        att_weights_ = nn.functional.softmax(logits, dim=-1)
        weights = self.drop50(att_weights_)
        y = torch.matmul(V.transpose(1,0), weights).transpose(1,0)
        y = self.output_linear(y)

        return y, att_weights_



class VASNet_Audio128_Att(nn.Module):

    def __init__(self):
        super(VASNet_Audio128_Att, self).__init__()

        self.m = 1024 # cnn features size
        self.m_audio = 128 # audio features size
        self.m_total = self.m + self.m_audio

        self.att = SelfAttention(input_size=self.m, output_size=self.m)
        self.att_audio = SelfAttention(input_size=self.m_audio, output_size=self.m_audio)

        self.ka = nn.Linear(in_features=self.m_total, out_features=1024)
        self.kb = nn.Linear(in_features=self.ka.out_features, out_features=1024)
        self.kc = nn.Linear(in_features=self.kb.out_features, out_features=1024)
        self.kd = nn.Linear(in_features=self.ka.out_features, out_features=1)

        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.drop50 = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=0)
        self.layer_norm_y = LayerNorm(self.m)
        self.layer_norm_y_audio = LayerNorm(self.m_audio)
        self.layer_norm_ka = LayerNorm(self.ka.out_features)

    def train_wrapper(self, hps, dataset):
        seq = dataset['features'][...]
        seq = torch.from_numpy(seq).unsqueeze(0)

        audio_128 = dataset['audio_features_128'][...]
        audio_128 = torch.from_numpy(audio_128).unsqueeze(0)

        target = dataset['gtscore'][...]
        target = torch.from_numpy(target).unsqueeze(0)

        # Normalize frame scores
        target -= target.min()
        if target.max() != 0:
            target /= target.max()
        if hps.use_cuda:
            seq = seq.float().cuda()
            target = target.float().cuda()
            audio_128 = audio_128.float().cuda()
 
        return self.forward(seq, seq.shape[1], audio_128, audio_128.shape[1]) + (target,)

    def eval_wrapper(self, hps, dataset):
        # seq = self.dataset[key]['features'][...]
        seq = dataset['features'][...]
        seq = torch.from_numpy(seq).unsqueeze(0)

        audio_128 = dataset['audio_features_128'][...]
        audio_128 = torch.from_numpy(audio_128).unsqueeze(0)

        if hps.use_cuda:
            seq = seq.float().cuda()
            audio_128 = audio_128.float().cuda()
        
        return self.forward(seq, seq.shape[1], audio_128, audio_128.shape[1])

    def forward(self, x, seq_len, x_audio, audio_len):

        m = x.shape[2] # Feature size
        m_audio = x_audio.shape[2]

        # Place the video frames to the batch dimension to allow for batch arithm. operations.
        # Assumes input batch size = 1.
        print("x: ", x)
        x = x.view(-1, m)
        y, att_weights_ = self.att(x)
        print("y: ",y)
        print("att_weights: ", att_weights_)
        y = y + x
        y = self.drop50(y)
        y = self.layer_norm_y(y)

        x_audio = x_audio.view(-1, m_audio)
        y_audio, att_weights_audio_ = self.att_audio(x_audio)
        y_audio = y_audio + x_audio
        y_audio = self.drop50(y_audio)
        y_audio = self.layer_norm_y_audio(y_audio)

        # Concat audio attention model
        y_comb = np.concatenate((y, y_audio), 1)
        att_weights_comb_ = np.concatenate((att_weights_, att_weights_audio_), 1)

        # Frame level importance score regression
        # Two layer NN
        y_comb = self.ka(y_comb)
        y_comb = self.relu(y_comb)
        y_comb = self.drop50(y_comb)
        y_comb = self.layer_norm_ka(y_comb)
        y_comb = self.kd(y_comb)
        y_comb = self.sig(y_comb)

        y_comb = y_comb.view(1, -1)

        return y_comb, att_weights_comb_



if __name__ == "__main__":
    pass
