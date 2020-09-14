import torch
import torch.nn as nn
import torchvision.models as backbone_
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign
from collections import OrderedDict
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from loss import *
import pdb


class Net_Basic(nn.Module):
    def __init__(self,margin=0.2):
        super(Net_Basic, self).__init__()
        self.backbone = backbone_.vgg16(pretrained=True).features
        # self.backbone = backbone_.inception_v3(pretrained=True)  # sain
        self.RoI = MultiScaleRoIAlign(featmap_names=[0], output_size=7, sampling_ratio=2)
        self.W_t = nn.Linear(512, 128, bias=False)
        self.W_v = nn.Linear(512, 128, bias=False)
        self.F_v = nn.Linear(512,512)
        self.F_t = nn.Linear(512, 512)

        self.W_combine = nn.Linear(1024, 512)
        self.triplet = TripletLoss(margin=margin)

    def extract_feature(self, x, bb_box):
        VGG_Tensor = OrderedDict()
        VGG_Tensor[0] = self.backbone(x)
        image_sizes_batch = [(256, 256) for i in range(x.shape[0])]
        output = self.RoI(VGG_Tensor, bb_box, image_sizes_batch)
        output_batch = output.split([y.shape[0] for y in bb_box], dim=0)
        return output_batch

    def CAMP_Interact(self, V, T): #V = anchor, T = Sample

        V_emb = self.W_v(V)  # N1 x 128
        T_emb = self.W_t(T)  # N2 x 128

        A = torch.matmul(T_emb, V_emb.t())  # N2 x N1

        A_v = F.softmax(A / math.sqrt(128), dim=-1)  # N2 x N1
        A_t = F.softmax(A.t() / math.sqrt(128), dim=-1)  # N1 x N2

        V_agg = torch.matmul(A_v, V)  # N2 x N1  *  N1 x 512   = N2 x 512
        T_agg = torch.matmul(A_t, T)  # N1 x N2  *  N2 x 512   = N1 x 512

        G_v = torch.sigmoid(V * T_agg)  # Element wise multiplication
        V_comb = F.relu(self.F_v(G_v * (V + T_agg))) + V  # N1 x 512   V_comb = updated V

        G_t = torch.sigmoid(T * V_agg)
        T_comb = F.relu(self.F_t(G_t * (T + V_agg))) + T  # N2 x 512

        V_final = V_comb.max(0)[0]
        T_final = T_comb.max(0)[0]
	
        return F.normalize(V_final.unsqueeze(0)), F.normalize(T_final.unsqueeze(0))
        # return V_comb, T_comb

    def Gumbel_Tree(self, V_Feature, Training = True):
        num_nodes = V_Feature.shape[0]
        if num_nodes == 1 : return V_Feature

        # Fuse two nodes and number of nodes will be reduced by one

        Correlation_Matrix = torch.matmul(V_Feature, V_Feature.t())
        Correlation_Vector = Correlation_Matrix[torch.triu(torch.ones_like(Correlation_Matrix), diagonal=1) == 1].unsqueeze(0)
        Correlation_Indices = torch.triu(torch.ones_like(Correlation_Matrix), diagonal=1).nonzero()

        left_node = torch.repeat_interleave(V_Feature, torch.arange(start=num_nodes-1, end=-1, step=-1, device='cuda'), dim=0)
        right_node = []
        for i in range(1, num_nodes):
            right_node.extend(V_Feature[i:])
        node_concat = torch.cat((left_node, torch.stack(right_node)), dim=1)

        node_rest = []
        # stores the unchanged nodes for each pair of selected nodes.
        # basically correlation indices store the nodes which are to be merged.
        # hence, node rest [i] will store the rest of the nodes uninvolved during merging of index[i]'s merging
        for index in Correlation_Indices:
            node_rest.append(torch.cat((V_Feature[:index[0]],
                                        V_Feature[index[0]+1:index[1]],
                                        V_Feature[index[1]+1:]), dim=0))
        node_rest = torch.stack(node_rest)

        if Training:
            select_mask = st_gumbel_softmax(logits=Correlation_Vector, temperature=1.0, mask=None)
        else:
            select_mask = greedy_select(logits=Correlation_Vector, mask=None)

        node_concatV = (node_concat * select_mask.squeeze(0).unsqueeze(-1).expand_as(node_concat)).sum(0)
        node_concatV = self.W_combine(node_concatV)

        node_rest = (node_rest * select_mask.squeeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(node_rest)).sum(0)
        V_Feature = torch.cat((node_rest, node_concatV.unsqueeze(0)), dim=0)

        return V_Feature

    def Hierarchy_Implcit(self, V, T, Training=True): #V = anchor, T = Sample

        while V.shape[0] > 1 or T.shape[0] > 1:
	    # Step 1::  Cross-modal attention
            V, T = self.CAMP_Interact(V, T)

	    # Step 2::  Hierarchical Fusion
            V = self.Gumbel_Tree(V_Feature=V, Training=Training)  # 2 nodes fused to 1 in one branch
            T = self.Gumbel_Tree(V_Feature=T, Training=Training)  # 2 nodes fused to 1 in the other branch

        return V.squeeze(0), T.squeeze(0)
        # return F.normalize(V).squeeze(0), F.normalize(T).squeeze(0)

    def Train(self, sketch, sketch_bbox, positive, pos_bbox, negative, negative_bbox):

        sketch_F_matrix_batch = self.extract_feature(sketch, sketch_bbox)
        pos_F_matrix_batch = self.extract_feature(positive, pos_bbox)
        neg_F_matrix_batch = self.extract_feature(negative, negative_bbox)

        anchor_Batch, sample_batch = [], []
        for sketch_F_matrix,  pos_F_matrix in zip(sketch_F_matrix_batch, pos_F_matrix_batch):
            sketch_F_matrix = F.adaptive_max_pool2d(sketch_F_matrix, (1, 1)).view(-1, 512)
            pos_F_matrix = F.adaptive_max_pool2d(pos_F_matrix, (1, 1)).view(-1, 512)
            sketch_F_pos, pos_F = self.Hierarchy_Implcit(sketch_F_matrix, pos_F_matrix)
            anchor_Batch.append(sketch_F_pos)
            sample_batch.append(pos_F)

        for sketch_F_matrix, neg_F_matrix in zip(sketch_F_matrix_batch,neg_F_matrix_batch):
            sketch_F_matrix = F.adaptive_max_pool2d(sketch_F_matrix, (1, 1)).view(-1, 512)
            neg_F_matrix = F.adaptive_max_pool2d(neg_F_matrix, (1, 1)).view(-1, 512)
            sketch_F_neg, neg_F = self.Hierarchy_Implcit(sketch_F_matrix, neg_F_matrix)
            anchor_Batch.append(sketch_F_neg)
            sample_batch.append(neg_F)
        loss = self.triplet(torch.stack(anchor_Batch[:positive.shape[0]]), torch.stack(anchor_Batch[positive.shape[0]:]),
                            torch.stack(sample_batch[:positive.shape[0]]), torch.stack(sample_batch[positive.shape[0]:]))

        return loss

    def Test(self, sketch, sketch_bbox, positive, pos_bbox):
        sketch_F_matrix_batch = self.extract_feature(sketch, sketch_bbox)
        pos_F_matrix_batch = self.extract_feature(positive, pos_bbox)
        anchor_Batch, sample_batch = [], []
        for sketch_F_matrix,  pos_F_matrix in zip(sketch_F_matrix_batch, pos_F_matrix_batch):
            sketch_F_matrix = F.adaptive_max_pool2d(sketch_F_matrix, (1, 1)).view(-1, 512)
            pos_F_matrix = F.adaptive_max_pool2d(pos_F_matrix, (1, 1)).view(-1, 512)
            anchor_Batch.append(sketch_F_matrix)
            sample_batch.append(pos_F_matrix)
        return anchor_Batch, sample_batch


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor_P, anchor_N, positive, negative, size_average=True):
        distance_positive = (anchor_P - positive).pow(2).sum(1)  
        distance_negative = (anchor_N - negative).pow(2).sum(1)  
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()




if __name__ == "__main__":
    model = Net_Basic()
    print('loaded')
    print(model)

    embedding = model(torch.randn(10,3,299,299))
    print(embedding.shape)
    for p in model.parameters():
        print(p.requires_grad, p.shape)









