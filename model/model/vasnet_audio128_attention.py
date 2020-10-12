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



class VASNet_Audio128_Attention(nn.Module):

    def __init__(self):
        super(VASNet_Audio128_Attention, self).__init__()

        self.m = 1024 # cnn features size
        self.audio_size = 128
        self.audio_linear = nn.Linear(in_features=self.audio_size,out_features=512, bias=False)
        self.att = SelfAttention(input_size=self.m, output_size=self.m)
        self.att_audio = SelfAttention(apperture = 40,input_size=self.audio_size,output_size=self.audio_size)
        self.ka = nn.Linear(in_features=self.m+self.audio_size, out_features=1024)
        self.kb = nn.Linear(in_features=self.ka.out_features, out_features=1024)
        self.kc = nn.Linear(in_features=self.kb.out_features, out_features=1024)
        self.kd = nn.Linear(in_features=self.ka.out_features, out_features=1)

        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.drop50 = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=0)
        self.layer_norm_y = LayerNorm(self.m)
        self.layer_norm_ka = LayerNorm(self.ka.out_features)
        self.layer_norm_audio=LayerNorm(self.att_audio.output_size)
        self.layer_norm_audio_2 = LayerNorm(self.att_audio.output_size)

    def train_wrapper(self, hps, dataset):
        seq = dataset['features'][...]
        audio_128 = dataset['audio_features_128'][...]
        #seq = np.concatenate((seq, audio_128), 1)
        seq = torch.from_numpy(seq).unsqueeze(0)

        target = dataset['gtscore'][...]
        target = torch.from_numpy(target).unsqueeze(0)
        audio_128 = torch.from_numpy(audio_128).unsqueeze(0)
        # Normalize frame scores
        target -= target.min()
        if target.max() != 0:
            target /= target.max()
        if hps.use_cuda:
            audio_128 = audio_128.float().cuda()
            seq, target = seq.float().cuda(), target.float().cuda()
 
        return self.forward(seq,seq.shape[1],audio_128,audio_128.shape[1]) + (target,)

    def eval_wrapper(self, hps, dataset):
        # seq = self.dataset[key]['features'][...]
        seq = dataset['features'][...]
        audio_128 = dataset['audio_features_128'][...]
        #seq = np.concatenate((seq, audio_128), 1)
        seq = torch.from_numpy(seq).unsqueeze(0)
        audio_128 = torch.from_numpy(audio_128).unsqueeze(0)

        if hps.use_cuda:
            audio_128 = audio_128.float().cuda()
            seq = seq.float().cuda()
        
        return self.forward(seq, seq.shape[1],audio_128,audio_128.shape[1])

    def forward(self, x, seq_len,audio,audio_len):

        m = x.shape[2] # Feature size
        #aud = self.audio_linear(audio)
        #aud = aud.view(-1,audio_len)
        #aud = self.layer_norm_audio(aud)
        aud = audio.view(audio_len,-1)
        aud = self.layer_norm_audio_2(aud)
        #print(f'aud : {aud.data.cpu().numpy()[0]}')
        #print(f'x : {x.data.cpu().numpy()[0]}')
        # Place the video frames to the batch dimension to allow for batch arithm. operations.
        # Assumes input batch size = 1.
        x = x.view(-1, m)
        #x = self.layer_norm_y(x)
        #print(x.shape)
        #print(aud.shape)
        #x = torch.cat((x,aud),dim=1)
        y, att_weights_ = self.att(x)
        aud_y, att_weights_aud = self.att_audio(aud)
        #print(f'y from attention :{y.data.cpu().numpy()[0]}')
        y = y + x
        y = self.drop50(y)
        y = self.layer_norm_y(y)
        #print(f'aud_y after attention : {aud_y.data.cpu().numpy()[0]}')
        aud_y = aud_y + aud
        aud_y = self.drop50(aud_y)
        aud_y = self.layer_norm_audio(aud_y)
        # Frame level importance score regression
        # Two layer NN
        #print(f'aud_y after layer_norm : {aud_y.data.cpu().numpy()[0]}')
        #print(f'y after batch norm : {y.data.cpu().numpy()[0]}')
        y = torch.cat((y,aud_y),dim=1)
        y = self.ka(y)
        y = self.relu(y)
        y = self.drop50(y)
        y = self.layer_norm_ka(y)
        #print(f'y after mlp : {y.data.cpu().numpy()[0]}')
        y = self.kd(y)
        #print(f'y after other mlp : {y.data.cpu().numpy()[0]}')
        y = self.sig(y)
        #print(f'y after sigmoid : {y.data.cpu().numpy()[0]}')
        y = y.view(1, -1)

        return y, att_weights_



if __name__ == "__main__":
    pass
