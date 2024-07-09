import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from  .base import register, make
import copy

class MHAtt(nn.Module):
    def __init__(self, embedding_size = 256, multi_head=1, dropout_ratio = 0.1):
        super(MHAtt, self).__init__()
        self.multi_head = multi_head
        self.multi_head_size = int(embedding_size/multi_head)
        self.embedding_size = embedding_size
        self.linear_v = nn.Linear(embedding_size, embedding_size)
        self.linear_k = nn.Linear(embedding_size, embedding_size)
        self.linear_q = nn.Linear(embedding_size, embedding_size)
        self.linear_merge = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout_ratio)

    def att(self, value, key, query, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None: scores = scores.masked_fill(mask, -1e9)
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)
        return torch.matmul(att_map, value)
    
    def forward(self, q, k, v):
        B = q.shape[0]
        v = self.linear_v(v).view(
            B,
            -1,
            self.multi_head,
            self.multi_head_size
            ).transpose(1, 2)
        k = self.linear_k(k).view(
            B,
            -1,
            self.multi_head,
            self.multi_head_size
            ).transpose(1, 2)
        q = self.linear_q(q).view(
            B,
            -1,
            self.multi_head,
            self.multi_head_size
            ).transpose(1, 2)
        
        atted = self.att(v, k, q)
        atted = atted.transpose(1, 2).contiguous().view(
            B,
            -1,
            self.embedding_size
            )
        atted = self.linear_merge(atted)
        return atted

class SA(nn.Module):
    def __init__(self, dropout_ratio = 0.1, multi_head = 1, embedding_size = 256, pre_normalize = False):
        super(SA, self).__init__()
        self.mhatt = MHAtt(embedding_size = embedding_size, multi_head = multi_head)
        self.embedding_size = embedding_size
        self.dropout = nn.Dropout(dropout_ratio)
        self.dropout1 = nn.Dropout(dropout_ratio)
        self.dropout2 = nn.Dropout(dropout_ratio)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.linear1 = nn.Linear(embedding_size, embedding_size)
        self.linear2 = nn.Linear(embedding_size, embedding_size)
        self.pre_normalize = pre_normalize
        
    def forward(self, x, shape = {}, fourier_pos=None):
        if self.pre_normalize:
            print('pre_normallize:') 
            return self.forward_pre(x, shape, fourier_pos)
        return self.forward_post(x, shape, fourier_pos)
        
    def forward_pre(self, x, shape = {}, fourier_pos = None) :
        v = x.view(shape['B'], -1, self.embedding_size)
        v2 = self.norm1(v)   
        q = self.with_pos_embed(v2.view(shape['B'], shape['H'], shape['W'],-1), pos)
        q = k = self.with_pos_embed(q, fourier_pos)
        q, k = q.view(shape['B'], -1, self.embedding_size),k.view(shape['B'], -1, self.embedding_size) 
        
        v = v + self.dropout1(self.mhatt(q, k , v2))
        v2 = self.norm2(v)
        v2 = self.linear2(self.dropout(F.relu(self.linear1(v2))))
        v = v + self.dropout2(v2)
        return v
    
    def with_pos_embed(self, x, pos = None):
        return x if pos is None else x + pos
        
    def forward_post(self, x, shape = {}, fourier_pos=None):
        if fourier_pos is not None: q = k = self.with_pos_embed(x, fourier_pos)
        else: q = k = x
        q, k = q.view(shape['B'], -1, self.embedding_size), k.view(shape['B'], -1, self.embedding_size)
        v = x.view(shape['B'], -1, self.embedding_size)
        atted = self.mhatt(q, k, v)              
        x = self.norm1(v + self.dropout1(atted)) 
        x2 = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.norm2(x + self.dropout2(x2))    
        x = x.view(shape['B'],shape['H'], shape['W'],-1)
        return x

def _get_clones(module,num_layers):
        return nn.ModuleList([copy.deepcopy(module) for i in range(num_layers)])


@register('nsq_pe')
class GridPositionEmbeddings(nn.Module):
    def __init__(self, num_pos_feats=32):
        super().__init__()
        self.num_pos_feats = num_pos_feats

    def forward(self, x):
        B, _, H, W = x.shape
        y_embed = torch.linspace(-1, 1, steps=H, dtype=torch.float32, device=x.device) 
        x_embed = torch.linspace(-1, 1, steps=W, dtype=torch.float32, device=x.device) 
        
        y_embed = y_embed.view(1, 1, H, 1)
        x_embed = x_embed.view(1, 1, 1, W)

        y_embed = y_embed.expand(B, 1, H, W)  
        x_embed = x_embed.expand(B, 1, H, W)  

        pos_x = torch.cat((x_embed, y_embed), dim=1)       
        pos_x = pos_x.repeat(1, self.num_pos_feats, 1, 1)
        pos_x = pos_x.transpose(1, 3).contiguous()
        return pos_x

@register('fourier_pe')
class PositionEmbeddingFourier(nn.Module):
    def __init__(self, num_pos_feats=32, temperature=10000, normalize=False, scale=None, amp=1.0):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.amp = amp
        if scale is not None and normalize is False: 
            raise ValueError("If the scale parameter is passed in, normalize must be True")
        if scale is None: scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        B, _, H, W = x.shape
        mask = torch.ones(B, H, W, dtype=torch.float32, device=x.device)
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        '''
        dim_t: tensor([1.0000e+00, 1.0000e+00, 1.3335e+00, 1.3335e+00, 1.7783e+00, 1.7783e+00,
        2.3714e+00, 2.3714e+00, 3.1623e+00, 3.1623e+00, 4.2170e+00, 4.2170e+00,
        5.6234e+00, 5.6234e+00, 7.4989e+00, 7.4989e+00, 1.0000e+01, 1.0000e+01,
        1.3335e+01, 1.3335e+01, 1.7783e+01, 1.7783e+01, 2.3714e+01, 2.3714e+01,
        3.1623e+01, 3.1623e+01, 4.2170e+01, 4.2170e+01, 5.6234e+01, 5.6234e+01,
        7.4989e+01, 7.4989e+01, 1.0000e+02, 1.0000e+02, 1.3335e+02, 1.3335e+02,
        1.7783e+02, 1.7783e+02, 2.3714e+02, 2.3714e+02, 3.1623e+02, 3.1623e+02,
        4.2170e+02, 4.2170e+02, 5.6234e+02, 5.6234e+02, 7.4989e+02, 7.4989e+02,
        1.0000e+03, 1.0000e+03, 1.3335e+03, 1.3335e+03, 1.7783e+03, 1.7783e+03,
        2.3714e+03, 2.3714e+03, 3.1623e+03, 3.1623e+03, 4.2170e+03, 4.2170e+03,
        5.6234e+03, 5.6234e+03, 7.4989e+03, 7.4989e+03], device='cuda:0')
        '''
        freqs = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        freqs = 1.0 / torch.pow(10000.0, (2 * (freqs // 2)) / self.num_pos_feats)
        '''
        freqs: tensor([1.0000e+00, 1.0000e+00, 7.4989e-01, 7.4989e-01, 5.6234e-01, 5.6234e-01,
        4.2170e-01, 4.2170e-01, 3.1623e-01, 3.1623e-01, 2.3714e-01, 2.3714e-01,
        1.7783e-01, 1.7783e-01, 1.3335e-01, 1.3335e-01, 1.0000e-01, 1.0000e-01,
        7.4989e-02, 7.4989e-02, 5.6234e-02, 5.6234e-02, 4.2170e-02, 4.2170e-02,
        3.1623e-02, 3.1623e-02, 2.3714e-02, 2.3714e-02, 1.7783e-02, 1.7783e-02,
        1.3335e-02, 1.3335e-02, 1.0000e-02, 1.0000e-02, 7.4989e-03, 7.4989e-03,
        5.6234e-03, 5.6234e-03, 4.2170e-03, 4.2170e-03, 3.1623e-03, 3.1623e-03,
        2.3714e-03, 2.3714e-03, 1.7783e-03, 1.7783e-03, 1.3335e-03, 1.3335e-03,
        1.0000e-03, 1.0000e-03, 7.4989e-04, 7.4989e-04, 5.6234e-04, 5.6234e-04,
        4.2170e-04, 4.2170e-04, 3.1623e-04, 3.1623e-04, 2.3714e-04, 2.3714e-04,
        1.7783e-04, 1.7783e-04, 1.3335e-04, 1.3335e-04], device='cuda:0')
        '''
        pos_x = x_embed[:, :, :, None] * freqs
        pos_y = y_embed[:, :, :, None] * freqs

        pos_x = self.amp * torch.cat((pos_x.sin(), pos_x.cos()), dim=-1)[:, :, :, :32].flatten(3)
        pos_y = self.amp * torch.cat((pos_y.sin(), pos_y.cos()), dim=-1)[:, :, :, :32].flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3)
        
        return pos

# Self-Attention Positional Embedding
@register('self-attention-position-Embed')    
class Transformer_Layer(nn.Module):
    def __init__(self, dropout_ratio = 0.1, multi_head = 1, embedding_size = 256, pre_normalize=\
                 False, num_self_attention_layers = 1,**kwargs):
        super(Transformer_Layer, self).__init__()
        
        self.layers = _get_clones(SA(dropout_ratio = dropout_ratio,multi_head = multi_head, embedding_size = embedding_size, pre_normalize = pre_normalize), num_layers = num_self_attention_layers)
    
    def forward(self, x, shape = {}, fourier_pos=None):
        output = x
        
        for layer in self.layers:
            output = layer(output, shape = shape, fourier_pos=fourier_pos)

        return output


# Location Aware Feature Clustering Module(LAFCM)
@register('constellation-LAFCM')
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_feat_cluster=False, use_self_attention=False, self_attention_kwargs={}, feat_cluster_kwargs={}):
        super(ConvBlock, self).__init__()
        self.use_feat_cluster = use_feat_cluster                
        self.use_self_attention = use_self_attention            
        self.self_attention_kwargs = self_attention_kwargs      
        self.feat_cluster_kwargs = feat_cluster_kwargs         
        self.feat_cluster_kwargs['channels'] = out_channels
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.pe1 = make('nsq_pe', num_pos_feats=out_channels//2)

        if self.use_feat_cluster: self.feat_cluster = make('constellation-CFC',**self.feat_cluster_kwargs)

        if self.use_self_attention:
            self.pe2 = make(self.self_attention_kwargs['positional_encoding'], num_pos_feats=self_attention_kwargs['embedding_size'])
            self.transformer = make('self-attention-position-Embed', **self.self_attention_kwargs)

        self.merge = nn.Conv2d(out_channels + self.feat_cluster_kwargs['num_clusters'], out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x, sideout=False):

        B, _, H, W = x.shape
        shape = {}
        shape['B'], shape['H'], shape['W'] = B, H, W
        sideout_dict = {}
        
        out_conv = self.conv(x)
        
        nsq_position = self.pe1(out_conv)
        encode_conv = out_conv + nsq_position.permute(0,3,1,2).contiguous()   
        
        if self.use_feat_cluster or self.use_self_attention:

            feature_sideout = encode_conv

            if self.use_feat_cluster:
                out_conv_reshape = out_conv.permute(0,2,3,1).contiguous().view(B*H*W, -1) 
                UV_dist = self.feat_cluster(out_conv_reshape, shape) 
                feature_sideout = UV_dist.view(B,H,W,-1).permute(0,3,1,2).contiguous() 
                
            if self.use_self_attention:
                pos = None
                if self.self_attention_kwargs['positional_encoding'] == 'fourier_pe': 
                    pos = self.pe2(feature_sideout)

                selfatt_sideout = self.transformer(feature_sideout.permute(0,2,3,1).contiguous(),shape, pos)
                feature_sideout = selfatt_sideout.view(B,H,W,-1).permute(0,3,1,2).contiguous()
            
            out_cat = torch.cat([encode_conv, feature_sideout], dim=1)
            out = self.merge(out_cat)
        
        else: out = out_conv
        
        out = self.bn(out)
        out = self.relu(out)
        out = self.maxpool(out)

        if sideout: return out, sideout_dict
        else: return out


@register('convnet4-featcluster-minibatch-sideout-classifier-ybranch')
class ConvNet4FeatCluster(nn.Module):
    def __init__(self, x_dim=3, h_dim=64, z_dim=64, stem_use_feat_cluster_list=[],
                 branch1_use_feat_cluster_list=[False, False, False, False],
                 branch2_use_feat_cluster_list=[False, False, False, False],
                 stem_use_self_attention_list=[],
                 branch1_use_self_attention_list =[False, False, False, False],
                 branch2_use_self_attention_list=[False, False, False, False],
                 self_attention_kwargs={}, feat_cluster_kwargs={}, y_branch_stage=0):
        super().__init__()
        channels = [h_dim, h_dim, z_dim, z_dim]# 64, 64, 64, 64
        self.n = len(channels)

        # Prepare for arguments
        def create_list(start_stage, end_stage, use_feat_cluster_list, use_self_attention_list):
            num_blocks = end_stage - start_stage + 1
            return nn.ModuleList(make('constellation-LAFCM', in_channels = in_channels, out_channels = out_channels,
                            use_feat_cluster = use_feat_cluster, use_self_attention = use_self_attention,
                            self_attention_kwargs = self_attention_kwargs, feat_cluster_kwargs = feat_cluster_kwargs)
                            for in_channels,out_channels,use_feat_cluster,use_self_attention,self_attention_kwargs,feat_cluster_kwargs in 
                            zip(([3]+channels[:-1])[start_stage:end_stage+1], channels[start_stage:end_stage+1], 
                                use_feat_cluster_list[start_stage:end_stage+1], use_self_attention_list[start_stage:end_stage+1],
                                [self_attention_kwargs]*num_blocks, [feat_cluster_kwargs]*num_blocks))                     
                                 
        self.stem =    create_list(0,      y_branch_stage-1, stem_use_feat_cluster_list + branch1_use_feat_cluster_list, stem_use_self_attention_list+branch1_use_self_attention_list)  # Note: Actually only stem kwargs are used. The unused branch arguments can be either from branch1 or branch2. Here we use branch1.
        self.branch1 = create_list(y_branch_stage, self.n-1, stem_use_feat_cluster_list + branch1_use_feat_cluster_list, stem_use_self_attention_list+branch1_use_self_attention_list)  # Note: Actually only branch1 kwargs are used.
        self.branch2 = create_list(y_branch_stage, self.n-1, stem_use_feat_cluster_list + branch2_use_feat_cluster_list, stem_use_self_attention_list+branch2_use_self_attention_list)  # Note: Actually only branch2 kwargs are used.
        
        self.out_dim = channels[3]
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, sideout=False, branch=-1):
        def sideout_func(x, sideout, attr_name):
            blocks = getattr(self, attr_name) 
            if sideout:
                sideout_dict = {}
                for i in range(len(blocks)):
                    x, s = blocks[i](x, sideout=True)
                    for layer_name, layer in s.items():
                        sideout_dict["{}.{}.{}".format(attr_name, i, layer_name)] = layer
                return x, sideout_dict
            else:
                for i in range(len(blocks)):
                    x = blocks[i](x)
                return x

        if branch == 1: branch_attr_name = "branch1"
        elif branch == 2: branch_attr_name = "branch2"
        else: raise ValueError()

        if sideout:
            sideout_dict = {}
            x, s_stem = sideout_func(x, sideout=True, attr_name="stem")
            x, s_branch = sideout_func(x, sideout=True, attr_name=branch_attr_name)
            sideout_dict.update(s_stem)
            sideout_dict.update(s_branch)
            sideout_dict['before_avgpool'] = x
        else:
            x = sideout_func(x, sideout=False, attr_name="stem")
            x = sideout_func(x, sideout=False, attr_name=branch_attr_name)

        # Feature average pooling 
        x = x.mean(-1).mean(-1)   
        
        # Return if enable side output.
        if sideout: return x, sideout_dict
        else: return x


@register('convnet4-featcluster-minibatch-sideout-classifier-ybranch-param-reduced')
def conv4_ybranch(**kwargs):
    return ConvNet4FeatCluster(x_dim=3, h_dim=64, z_dim=64,**kwargs)
