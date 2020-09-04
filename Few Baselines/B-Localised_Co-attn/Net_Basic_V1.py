import torch
import torch.nn as nn
import torchvision.models as backbone_
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign
from collections import OrderedDict
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net_Basic(nn.Module):
    def __init__(self,margin=0.2):
        super(Net_Basic, self).__init__()
        self.backbone = backbone_.vgg16(pretrained=True).features
        self.RoI = MultiScaleRoIAlign(featmap_names=[0], output_size=7, sampling_ratio=2)
        self.W_t = nn.Linear(512, 128, bias=False)
        self.W_v = nn.Linear(512, 128, bias=False)
        self.F_v = nn.Linear(512,512)
        self.F_t = nn.Linear(512, 512)
        self.F_gv = nn.Linear(1024, 512)
        self.F_gt = nn.Linear(1024, 512)
        self.F_gtant = nn.Linear(1024, 512)
        self.F_gtanv = nn.Linear(1024, 512)
        self.triplet = TripletLoss(margin=margin)

    def extract_feature(self, x, bb_box):
        VGG_Tensor = OrderedDict()
        VGG_Tensor[0] = self.backbone(x)
        image_sizes_batch = [(256, 256) for i in range(x.shape[0])]
        output = self.RoI(VGG_Tensor, bb_box, image_sizes_batch)
        output_batch = output.split([y.shape[0] for y in bb_box], dim=0)
        return output_batch

    # def CAMP_Interact(self, V, T): #V = anchor, T = Sample  ; Ayan
    #
    #     V_emb = self.W_v(V)  # N1 x 128
    #     T_emb = self.W_t(T)  # N2 x 128
    #
    #     A = torch.matmul(T_emb, V_emb.t())  # N2 x N1
    #
    #     A_v = F.softmax(A / math.sqrt(128), dim=-1)  # N2 x N1
    #     A_t = F.softmax(A.t() / math.sqrt(128), dim=-1)  # N1 x N2
    #
    #     V_agg = torch.matmul(A_v, V)  # N2 x N1  *  N1 x 512   = N2 x 512
    #     T_agg = torch.matmul(A_t, T)  # N1 x N2  *  N2 x 512   = N1 x 512
    #
    #     G_v = torch.sigmoid(V * T_agg)  # Element wise multiplication
    #     V_comb = F.relu(self.F_v(G_v * (V + T_agg))) + V  # N1 x 512   V_comb = updated V
    #
    #     G_t = torch.sigmoid(T * V_agg)
    #     T_comb = F.relu(self.F_t(G_t * (T + V_agg))) + T  # N2 x 512
    #
    #     V_final = V_comb.max(0)[0]
    #     T_final = T_comb.max(0)[0]
    #
    #     return F.normalize(V_final.unsqueeze(0)).squeeze(0), F.normalize(T_final.unsqueeze(0)).squeeze(0)
    #

    def CAMP_Interact(self, V, T): #V = anchor, T = Sample  ; Ayan

        V_emb = self.W_v(V)  # N1 x 128
        T_emb = self.W_t(T)  # N2 x 128

        A = torch.matmul(T_emb, V_emb.t())  # N2 x N1

        A_v = F.softmax(A / math.sqrt(128), dim=-1)  # N2 x N1
        A_t = F.softmax(A.t() / math.sqrt(128), dim=-1)  # N1 x N2

        V_agg = torch.matmul(A_v, V)  # N2 x N1  *  N1 x 512   = N2 x 512
        T_agg = torch.matmul(A_t, T)  # N1 x N2  *  N2 x 512   = N1 x 512

        # G_v1 = torch.sigmoid(V * T_agg)  # Element wise multiplication
        G_v = torch.sigmoid(self.F_gv(torch.cat((V,T_agg),dim=1)))
        # VT_agg = torch.tanh(self.F_gtanv(torch.cat((V,T_agg),dim=1)))
        V_comb = F.relu(self.F_v(G_v * (V + T_agg))) + V  # N1 x 512   V_comb = updated V
        # V_comb = (G_v * (V + T_agg)) + V  # N1 x 512   V_comb = updated V -- not helpful
        # V_comb = F.relu(self.F_v(G_v * (VT_agg))) + V  # N1 x 512   V_comb = updated V

        # G_t1 = torch.sigmoid(T * V_agg)
        G_t = torch.sigmoid(self.F_gt(torch.cat((T, V_agg), dim=1)))
        # TV_agg = torch.tanh(self.F_gtant(torch.cat((T, V_agg), dim=1)))
        T_comb = F.relu(self.F_t(G_t * (T + V_agg))) + T  # N2 x 512
        # T_comb = G_t * (T + V_agg) + T  # N2 x 512    -- not helpful
        # print ('hi')
        V_final = V_comb.max(0)[0]
        T_final = T_comb.max(0)[0]

        return F.normalize(V_final.unsqueeze(0)).squeeze(0), F.normalize(T_final.unsqueeze(0)).squeeze(0)


    def Train(self, sketch, sketch_bbox, positive, pos_bbox, negative, negative_bbox):

        sketch_F_matrix_batch = self.extract_feature(sketch, sketch_bbox)
        pos_F_matrix_batch = self.extract_feature(positive, pos_bbox)
        neg_F_matrix_batch = self.extract_feature(negative, negative_bbox)

        anchor_Batch, sample_batch = [], []
        for sketch_F_matrix,  pos_F_matrix in zip(sketch_F_matrix_batch, pos_F_matrix_batch):
            sketch_F_matrix = F.adaptive_max_pool2d(sketch_F_matrix, (1, 1)).view(-1, 512)
            pos_F_matrix = F.adaptive_max_pool2d(pos_F_matrix, (1, 1)).view(-1, 512)
            sketch_F_pos, pos_F = self.CAMP_Interact(sketch_F_matrix, pos_F_matrix)
            anchor_Batch.append(sketch_F_pos)
            sample_batch.append(pos_F)

        for sketch_F_matrix, neg_F_matrix in zip(sketch_F_matrix_batch,neg_F_matrix_batch):
            sketch_F_matrix = F.adaptive_max_pool2d(sketch_F_matrix, (1, 1)).view(-1, 512)
            neg_F_matrix = F.adaptive_max_pool2d(neg_F_matrix, (1, 1)).view(-1, 512)
            sketch_F_neg, neg_F = self.CAMP_Interact(sketch_F_matrix, neg_F_matrix)
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
        distance_positive = (anchor_P - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor_N - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


if __name__ == "__main__":
    model = Net_Basic()
    print('loaded')
    print(model)

    embedding = model(torch.randn(10,3,299,299))
    print(embedding.shape)
    #for p in model.parameters():
    #    print(p.requires_grad, p.shape)









