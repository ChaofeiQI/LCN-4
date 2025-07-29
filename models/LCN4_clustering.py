import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import register

@register('constellation-CFC')
class FeatureClustering_Layer(nn.Module):
    def __init__(self, K=1.0, num_clusters=2, fix_init=False, channels=64, V_count_init=1.0,**kwargs):
        super(FeatureClustering_Layer, self).__init__()
        self.K = K
        self.num_clusters = num_clusters    
        self.fix_init = fix_init
        self.channels = channels
        
        if self.fix_init:
            self.r = np.random.RandomState(1)
            V_init = torch.tensor(self.r.randn(self.num_clusters, self.channels)).float()
        else:
            V_init = torch.randn(self.num_clusters, self.channels)
        V_init = F.normalize(V_init, dim=-1) 

        self.register_buffer('V_buffer', V_init)  
        self.register_buffer('V_count_buffer', V_count_init * torch.ones(self.num_clusters, dtype=torch.float64)) 

    def compute_dist(self, U, V): 
        V = F.normalize(V, dim=-1)
        UV_dist = U.mm(V.transpose(0, 1))
        
        return UV_dist


    def mik_compute(self,UV_dist):
        Coeff = F.softmax(self.K * UV_dist, dim=-1) 
        
        return Coeff 

    def forward(self, x, shape={}):
        
        V    = self.V_buffer.detach().clone()
        V_count = self.V_count_buffer.detach().clone()
        
        assert len(x.shape) == 2
        U = x
        N, C = U.shape
        
        U = F.normalize(U, dim=-1)
        UV_dist = self.compute_dist(U, V)
        
        if self.training:            
            Coeff = self.mik_compute(UV_dist)                        
            cur_V = Coeff.transpose(0,1).mm(U)/Coeff.sum(0).view(-1,1)
            delta_count = Coeff.sum(0).double()                     
            V_count += delta_count                                   
            alpha_vec = (delta_count / V_count).float().view(-1, 1) 
            V = (1-alpha_vec)*V + alpha_vec*cur_V                   
            
            self.V_buffer.copy_(V.detach().clone()) 
            self.V_count_buffer.copy_(V_count.detach().clone())     

        return UV_dist           


class MeanShiftClusteringLayer(nn.Module):
    def __init__(self, bandwidth=1.0, channels=64, num_clusters=2):
        super(MeanShiftClusteringLayer, self).__init__()
        self.bandwidth = bandwidth
        self.channels = channels
        self.num_clusters = num_clusters
        self.cluster_centers = None

    def forward(self, x, shape={}):
        assert len(x.shape) == 2        
        x_np = x.detach().cpu().numpy()
        ms = MeanShift(bandwidth=self.bandwidth)
        ms.fit(x_np)
        labels = ms.labels_
        self.cluster_centers = ms.cluster_centers_

        labels_tensor = torch.from_numpy(labels).to(x.device)
        cluster_centers_tensor = torch.from_numpy(self.cluster_centers).to(x.device)

        return cluster_centers_tensor