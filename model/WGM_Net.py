import torch
import torch.nn as nn
from model.WGM_Cell import WGM_Cell
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
import pywt
import math
from model.MambaBlock import build_mamba_block


def disentangle(x, w, j=1):
    # 调整输入张量 x 的维度顺序，从 [S, T, N, D] 变为 [S, D, N, T]
    # pywt.wavedec 函数通常期望在最后一个维度上进行分解操作，所以需要调整输入张量的维度顺序，使得时间维度 T 位于最后，以便 pywt.wavedec 能够正确地对时间序列进行小波分解。
    x = x.permute(0, 3, 2, 1)  # [S,D,N,T]
    # x_np = x.cpu().numpy()
    x_np = x.detach().cpu().numpy()
    # 使用 pywt.wavedec 函数对 x_np 进行小波分解，得到小波系数列表 coef
    coef = pywt.wavedec(x_np, w, level=j)
    # 初始化低频系数列表 coefl
    coefl = [coef[0]]
    # 循环 len(coef)-1 次，将 None 添加到 coefl 列表中，用于后续重构低频信号
    for i in range(len(coef) - 1):
        coefl.append(None)
    # 初始化高频系数列表 coefh，将 None 作为第一个元素
    coefh = [None]
    for i in range(len(coef) - 1):
        coefh.append(coef[i + 1])
    # 使用 pywt.waverec 函数对低频系数和高频系数列表 coefl和coefh进行小波重构，得到低频和高频信号的 NumPy 数组 xl_np,xh_np
    xl_np = pywt.waverec(coefl, w)
    xh_np = pywt.waverec(coefh, w)
    # NumPy数组转回Tensor
    # 同时调整维度顺序，从 [S, D, N, T] 变回 [S, T, N, D]
    xl = torch.from_numpy(xl_np).to(x.device).permute(0, 3, 2, 1)
    xh = torch.from_numpy(xh_np).to(x.device).permute(0, 3, 2, 1)
    return xl, xh


class WGM_Encoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim, num_layers=1):
        super(WGM_Encoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.WGM_cells = nn.ModuleList()
        self.WGM_cells.append(WGM_Cell(node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim))
        for _ in range(1, num_layers):
            self.WGM_cells.append(WGM_Cell(node_num, dim_out, dim_out, cheb_k, embed_dim, time_dim))

    def forward(self, x, init_state, node_embeddings):
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]  # x=[batch,steps,nodes,input_dim]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]  # state=[batch,steps,nodes,input_dim]
            inner_states = []
            for t in range(seq_length):
                state = self.WGM_cells[i](current_inputs[:, t, :, :], state, [node_embeddings[0][:, t, :],
                                                                             node_embeddings[
                                                                                 1]])  # state=[batch,steps,nodes,input_dim]
                # state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state,[node_embeddings[0], node_embeddings[1]])
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)

        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.WGM_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)  # (num_layers, B, N, hidden_dim)


class WGM_Decoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim, num_layers=1):
        super(WGM_Decoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.WGM_cells = nn.ModuleList()
        self.WGM_cells.append(WGM_Cell(node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim))
        for _ in range(1, num_layers):
            self.WGM_cells.append(WGM_Cell(node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim))

    def forward(self, xt, init_state, node_embeddings):
        # xt: (B, N, D)
        # init_state: (num_layers, B, N, hidden_dim)
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.WGM_cells[i](current_inputs, init_state[i], [node_embeddings[0], node_embeddings[1]])
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden


# =============================
# 方案A：直接用 POI 初始化 node_embeddings
# =============================
# 新增：MLP 将 POI(6维) -> embed_dim
class POIEncoder(nn.Module):
    def __init__(self, poi_dim, embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6, 64), nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128), nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, embed_dim)
        )

    def forward(self, poi):
        return self.mlp(poi)  # (307, embed_dim)



class WGM_Net(nn.Module):
    def __init__(self, args):
        super(WGM_Net, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.use_D = args.use_day
        self.use_W = args.use_week

        # 1. 加载 POI 数据
        # ----------------------
        poi = np.load('/root/WGM_Net/data/PEMS08/PEMS08_poi_6.npy')  # (307,6)
        poi = torch.tensor(poi, dtype=torch.float32)
        # 2. MLP -> embed_dim
        # ----------------------
        self.poi_encoder = POIEncoder(poi_dim=poi.shape[1], embed_dim=args.embed_dim)
        poi_emb = self.poi_encoder(poi)  # (307, embed_dim)

        self.rand_node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim))
        self.gate_mlp = nn.Sequential(
            nn.Linear(args.embed_dim * 2, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
        self.gate_mlp[-2].bias.data.fill_(math.log(0.3 / (1 - 0.3)))

        gate_input = torch.cat([self.rand_node_embeddings, poi_emb], dim=1)

        gate = self.gate_mlp(gate_input).expand(-1, args.embed_dim)

        combined = gate * self.rand_node_embeddings + (1 - gate) * poi_emb

        self.node_embeddings = nn.Parameter(combined, requires_grad=True)

        # 可学习融合系数（初始均分，最终归一化）
        self.alpha1 = nn.Parameter(torch.tensor(0.25))
        self.alpha2 = nn.Parameter(torch.tensor(0.25))
        self.alpha3 = nn.Parameter(torch.tensor(0.25))
        self.alpha4 = nn.Parameter(torch.tensor(0.25))

        self.dropout = nn.Dropout(p=0.15)
        self.default_graph = args.default_graph

        self.T_i_D_emb = nn.Parameter(torch.empty(288, args.time_dim))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, args.time_dim))
        # 增加节假日定义
        self.H_i_emb = nn.Parameter(torch.empty(3, args.time_dim))  # 0=工作日, 1=周末, 2=节假日
        nn.init.xavier_uniform_(self.H_i_emb)
        # self.embedding_proj = nn.Linear(3 * args.time_dim, args.time_dim)

        # 定义编码器
        self.encoder_xl1 = WGM_Encoder(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                      args.embed_dim, args.time_dim, args.num_layers)
        self.encoder_xh1 = WGM_Encoder(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                      args.embed_dim, args.time_dim, args.num_layers)
        self.encoder_xl2 = WGM_Encoder(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                      args.embed_dim, args.time_dim, args.num_layers)
        self.encoder_xh2 = WGM_Encoder(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                      args.embed_dim, args.time_dim, args.num_layers)
        # self.decoder = WGM_Decoder(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
        #                           args.embed_dim, args.time_dim, args.num_layers)
        # predictor
        self.proj = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim, bias=True))
        self.end_conv_xl1 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.end_conv_xh1 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.end_conv_xl2 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.end_conv_xh2 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        # self.end_conv2 = nn.Conv2d(12, 12, kernel_size=(1, 1), bias=True)

        # 可学习融合尺度（把注意力结果作为对线性融合的增强项）
        self.attn_scale = nn.Parameter(torch.tensor(0.5))

        if args.type == 'P':
            self.TA = TransformAttentionModel(self.hidden_dim, args.time_dim, args.embed_dim)
            self.decoder = Parallel_decoder(args)

    def forward(self, source, traget=None, batches_seen=None):
        # source: B, T_1, N, D
        # target: B, T_2, N, D

        t_i_d_data1 = source[..., 0, -3]
        t_i_d_data2 = traget[..., 0, -3]
        # T_i_D_emb = self.T_i_D_emb[(t_i_d_data[:, -1, :] * 288).type(torch.LongTensor)]
        T_i_D_emb1 = self.T_i_D_emb[(t_i_d_data1 * 288).type(torch.LongTensor)]  # [B, T_src, time_dim]
        T_i_D_emb2 = self.T_i_D_emb[(t_i_d_data2 * 288).type(torch.LongTensor)]  # [B, T_tar, time_dim]

        d_i_w_data1 = source[..., 0, -2]
        d_i_w_data2 = traget[..., 0, -2]
        # D_i_W_emb = self.D_i_W_emb[(d_i_w_data[:, -1, :]).type(torch.LongTensor)]
        D_i_W_emb1 = self.D_i_W_emb[(d_i_w_data1).type(torch.LongTensor)]  # [B, T_src, time_dim]
        D_i_W_emb2 = self.D_i_W_emb[(d_i_w_data2).type(torch.LongTensor)]  # [B, T_tar, time_dim]
        # ========== 节假日索引 (倒数第1个) ==========
        h_i_data1 = source[..., 0, -1]
        h_i_data2 = traget[..., 0, -1]
        H_i_emb1 = self.H_i_emb[(h_i_data1).type(torch.LongTensor)]  # [B, T_src, time_dim]
        H_i_emb2 = self.H_i_emb[(h_i_data2).type(torch.LongTensor)]  # [B, T_tar, time_dim]
        # 将三个嵌入特征分步相乘
        node_embedding1 = T_i_D_emb1 * D_i_W_emb1 * H_i_emb1
        node_embedding2 = T_i_D_emb2 * D_i_W_emb2 * H_i_emb2
        # node_embedding1 = self.embedding_proj(node_embedding1)  # (B, T, N, time_dim)
        # node_embedding2 = self.embedding_proj(node_embedding2)

        en_node_embeddings = [node_embedding1, self.node_embeddings]

        # source = source[..., :self.input_dim].unsqueeze(-1)
        # source = source[..., :self.input_dim]
        B, T, N, _ = source.shape
        flow = source[..., 0].unsqueeze(-1)  # [B,T,N,1]

        xl1, xh1 = disentangle(flow, 'bior3.5', j=1)
        xl2, xh2 = disentangle(flow, 'haar', j=1)

        # 提取其他辅助特征（如 速度 占有率等）
        # x_aux = source[..., 1:]  # 除了第0维以外的所有特征

        inp_xl1 = xl1
        inp_xh1 = xh1
        inp_xl2 = xl2
        inp_xh2 = xh2

        # ===== 编码器执行函数 =====
        def run_path(encoder, input_tensor):
            init_state = encoder.init_hidden(input_tensor.shape[0])
            state, h_n = encoder(input_tensor, init_state, en_node_embeddings)
            return h_n  # (num_layers, B, N, hidden_dim)

        # ===== 四分支输出 =====
        h_xl1 = run_path(self.encoder_xl1, inp_xl1)
        h_xh1 = run_path(self.encoder_xh1, inp_xh1)
        h_xl2 = run_path(self.encoder_xl2, inp_xl2)
        h_xh2 = run_path(self.encoder_xh2, inp_xh2)

        # ===== 融合（归一化） =====
        alpha_total = self.alpha1 + self.alpha2 + self.alpha3 + self.alpha4
        a1 = self.alpha1 / alpha_total
        a2 = self.alpha2 / alpha_total
        a3 = self.alpha3 / alpha_total
        a4 = self.alpha4 / alpha_total
        # linear_conv = a1 * out_xl1 + a2 * out_xh1 + a3 * out_xl2 + a4 * out_xh2  # (B, C, N, 1)

        # 每个 h_x* : (num_layers, B, N, hidden_dim)
        h_n = (a1 * h_xl1[-1] + a2 * h_xh1[-1] + a3 * h_xl2[-1] + a4 * h_xh2[-1]).unsqueeze(0)  # (1, B, N, hidden_dim)

        # reshape conv outputs -> (B, N, T, D), T = horizon, D = output_dim

        output = self.decoder(
            source=source,
            traget=traget,
            h_n=h_n,
            node_embedding1=node_embedding1,
            node_embedding2=node_embedding2,
            node_embeddings=self.node_embeddings,
            batches_seen=batches_seen
        )

        return output


class Parallel_decoder(nn.Module):
    def __init__(self, args=None):
        super(Parallel_decoder, self).__init__()
        self.TA = TransformAttentionModel(args.rnn_units, args.time_dim, args.embed_dim)
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.decoder = WGM_Decoder(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                  args.embed_dim, args.time_dim, args.num_layers)
        # self.proj = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim, bias=True))
        self.dropout = nn.Dropout(p=0.)
        # self.proj = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim, bias=True))
        self.weights = nn.Parameter(torch.FloatTensor(self.horizon, self.hidden_dim, self.output_dim))
        self.bias = nn.Parameter(torch.FloatTensor(self.horizon, self.output_dim))
        # ===== 输出端 Mamba =====
        self.out_mamba = build_mamba_block(
            dim=self.output_dim,  # ⚠️ 注意这里是 output_dim
            use_standard=True,
            d_state=16,
            d_conv=4,
            expand=2
        )
        self.out_dropout = nn.Dropout(0.1)
        # 可学习残差权重（强烈建议）
        self.out_scale = nn.Parameter(torch.tensor(0.5))
        # self.end_conv = nn.Conv2d(args.horizon, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def forward(self, source, traget, h_n, node_embedding1, node_embedding2, node_embeddings, batches_seen):
        h_n = h_n[0]
        h_n = h_n.unsqueeze(1)
        de_input = self.TA(h_n, node_embedding1[:, -1, :].unsqueeze(1), node_embedding2).flatten(0, 1)
        # de_input = h_n.expand(-1,self.horizon,-1,-1).flatten(0, 1)
        # output = self.proj(self.dropout(de_input))
        node_embedding2 = node_embedding2.flatten(0, 1)
        # return output

        go = torch.zeros((source.shape[0] * self.horizon, self.num_node, self.output_dim), device=source.device)

        state, ht_list = self.decoder(go, [de_input], [node_embedding2, node_embeddings])  # Decoder

        # go = self.proj(self.dropout(state))
        # output = go.reshape(source.shape[0],self.horizon,self.num_node,self.output_dim)

        # state = state.reshape(source.shape[0], self.horizon, self.num_node, self.hidden_dim)
        # output = torch.matmul(self.dropout(state), self.weights) + self.bias.unsqueeze(dim=-2)
        B = source.shape[0]

        state = state.reshape(B, self.horizon, self.num_node, self.hidden_dim)

        output = torch.matmul(self.dropout(state), self.weights) + self.bias.unsqueeze(dim=-2)
        # output: (B, H, N, output_dim)

        # output: (B, H, N, D)

        out_seq = output  # 不 reshape ❗

        out_seq = out_seq + self.out_scale * self.out_dropout(self.out_mamba(out_seq))

        output = out_seq
        # output = self.end_conv(self.dropout(state)).reshape(source.shape[0],self.horizon,self.num_node,self.output_dim)

        return output



class TransformAttentionModel(torch.nn.Module):
    def __init__(self, hidden_dim, time_dim, embed_dim):
        super(TransformAttentionModel, self).__init__()
        self.fc_Q = torch.nn.Linear(time_dim + hidden_dim, hidden_dim)
        self.fc_K = torch.nn.Linear(time_dim + hidden_dim, hidden_dim)
        self.fc_V = torch.nn.Linear(time_dim + hidden_dim, hidden_dim)
        self.fc24 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc25 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, 2, hidden_dim, hidden_dim))
        self.weights = nn.Parameter(torch.FloatTensor(2, hidden_dim, hidden_dim))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, hidden_dim))
        self.bias = nn.Parameter(torch.FloatTensor(hidden_dim))
        self.d = 8
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_dim)

    def forward(self, X, STE_P, STE_Q):
        STE_Q = STE_Q.unsqueeze(2)
        STE_P = STE_P.unsqueeze(2)
        query = F.relu(
            self.fc_Q(torch.cat((STE_Q.expand(-1, -1, X.shape[2], -1), X.expand(-1, STE_Q.shape[1], -1, -1)), dim=-1)))
        key = F.relu(self.fc_K(torch.cat((STE_P.expand(-1, -1, X.shape[2], -1), X), dim=-1)))
        value = F.relu(self.fc_V(torch.cat((STE_P.expand(-1, -1, X.shape[2], -1), X), dim=-1)))

        query = torch.cat(torch.split(query, int(query.shape[-1] / self.d), dim=-1), dim=0)
        key = torch.cat(torch.split(key, int(key.shape[-1] / self.d), dim=-1), dim=0)
        value = torch.cat(torch.split(value, int(value.shape[-1] / self.d), dim=-1), dim=0)
        query = torch.transpose(query, 2, 1)  # [K * batch_size, num_nodes, num_steps, d]
        key = torch.transpose(torch.transpose(key, 1, 2), 2, 3)  # [K * batch_size, num_nodes, d, num_steps]
        value = torch.transpose(value, 2, 1)

        attention = torch.matmul(query, key)  # [K * batch_size, num_nodes, num_steps, num_steps]
        # attention /= (self.d ** 0.5)
        attention = torch.softmax(attention, dim=-2)

        output = torch.matmul(attention, value)
        output = torch.transpose(output, 2, 1)
        output = torch.cat(torch.split(output, output.shape[0] // self.d, dim=0), dim=-1)
        output = torch.stack((X.expand(-1, output.shape[1], -1, -1), output), dim=3)
        # output = torch.einsum('btnki,nkio->btno', output, weights) + bias  # b, N, dim_out

        output = torch.einsum('btnki,kio->btno', output, self.weights) + self.bias  # b, N, dim_out

        return output


