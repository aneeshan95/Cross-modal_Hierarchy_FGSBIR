import torch
import torch.nn as nn
import torchvision.models as backbone_
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import math
import torch.nn.modules.container as container
import copy

class Joint_embdedding(nn.Module):
    def __init__(self):
        super(Joint_embdedding, self).__init__()
        self.raster_emb = Net_Basic()
        self.vector_emb = Vector_embedding()
        self.linear_embed = nn.Sequential(nn.Linear(768+2048, 512), nn.ReLU(True),nn.Dropout(0.5), nn.Linear(512, 64))



    def forward(self, raster_sketch, vector_sketch, seq_len):
        vector_emb = self.vector_emb(vector_sketch, seq_len)
        # taking 2048 version
        _, raster_emb = self.raster_emb(raster_sketch)

        return F.normalize(self.linear_embed(torch.cat((vector_emb, raster_emb), dim=-1)))


class Vector_embedding(nn.Module):
    def __init__(self, inp_dim=3, hidden_size=768, atten_heads=4, transformer_num_layers=2):
        super(Vector_embedding, self).__init__()

        self.emb_stroke = nn.Linear(inp_dim, hidden_size)  # Embedding Layer
        self.pos_encoder_stroke = PositionalEncoding(hidden_size)

        encoder_layers_stroke = nn.TransformerEncoderLayer(hidden_size,
                                                           atten_heads, hidden_size)
        encoder_norm_stroke = torch.nn.LayerNorm(hidden_size)
        self.encoder_stroke = TransformerEncoderUS(
            encoder_layers_stroke, transformer_num_layers, encoder_norm_stroke)
        self.linear_embed = nn.Linear(768, 64)

    def forward(self, x, seq_len):
        x = x.type(torch.float32).to(device)
        x = x.permute(1, 0, 2)  # Shape: max_len, batch, 3

        src_key_pad_mask = torch.zeros(
            (x.shape[1], x.shape[0]), dtype=torch.bool)
        for i_k, seq in enumerate(seq_len):
            src_key_pad_mask[i_k, seq:] = True

        x = self.emb_stroke(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.encoder_stroke(self.pos_encoder_stroke(x),
                                src_key_padding_mask=src_key_pad_mask.to(device))

        x_hidden = []
        for i_k, seq in enumerate(seq_len):
            x_hidden.append(torch.max(x[:seq, i_k, :], dim=0)[0])
        x_hidden = torch.stack(x_hidden, dim=0)

        return x_hidden


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoderUS(nn.Module):


    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoderUS, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        r"""Pass the input through the endocder layers in turn.

        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output


def _get_clones(module, N):
    return container.ModuleList([copy.deepcopy(module) for i in range(N)])
    # return container.ModuleList([copy.copy(module) for i in range(N)])


class Vector_embedding2(nn.Module):
    def __init__(self):
        super(Vector_embedding2, self).__init__()
        self.LSTM_encoder = nn.LSTM(3, 512,
                                    num_layers=3,
                                    dropout=0.5,
                                    batch_first=True, bidirectional=True)
        self.linear_embedding = nn.Linear(1024, 768)

    def forward(self, x, seq_len):

        batch_size = x.shape[0]
        x = pack_padded_sequence(x.to(device), seq_len.to(device),
                batch_first=True, enforce_sorted=False)

        _, (x_hidden, _) = self.LSTM_encoder(x.float())
        x_hidden = x_hidden.view(3,2,batch_size,512)[-1].permute(1,0,2).reshape(batch_size, -1)
        embedding  = self.linear_embedding(x_hidden)

        return F.normalize(embedding)

class Net_Basic(nn.Module):
    def __init__(self):
        super(Net_Basic, self).__init__()
        backbone = backbone_.inception_v3(pretrained=True)

        #self.backbone.aux_logits = False
        self.Conv2d_1a_3x3 = backbone.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = backbone.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = backbone.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = backbone.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = backbone.Conv2d_4a_3x3
        self.Mixed_5b = backbone.Mixed_5b
        self.Mixed_5c = backbone.Mixed_5c
        self.Mixed_5d = backbone.Mixed_5d
        self.Mixed_6a = backbone.Mixed_6a
        self.Mixed_6b = backbone.Mixed_6b
        self.Mixed_6c = backbone.Mixed_6c
        self.Mixed_6d = backbone.Mixed_6d
        self.Mixed_6e = backbone.Mixed_6e

        self.Mixed_7a = backbone.Mixed_7a
        self.Mixed_7b = backbone.Mixed_7b
        self.Mixed_7c = backbone.Mixed_7c
        self.fc = nn.Linear(2048, 64)

    def forward(self, x):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = x.view(x.size(0), -1) #N x 2048
        x = F.normalize(x) #N x 2048
        embedding = self.fc(x)
        embedding = F.normalize(embedding)
        return embedding, x

if __name__ == "__main__":
    model = Net_Basic()
    print('loaded')
    print(model)

    embedding = model(torch.randn(10,3,299,299))
    print(embedding.shape)









