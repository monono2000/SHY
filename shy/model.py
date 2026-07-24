import torch
import numpy as np
import torch.nn as nn
from torch_scatter import scatter
from torch.nn import functional as F

if __package__:
    from .layers import *
else:
    from layers import *


def _visit_len_to_int(visit_len):
    if torch.is_tensor(visit_len):
        return int(visit_len.item())
    return int(visit_len)


def relaxed_bernoulli_straight_through(probs, temperature):
    temperature = torch.as_tensor(temperature, dtype=probs.dtype, device=probs.device)
    relaxed = torch.distributions.RelaxedBernoulli(temperature=temperature, probs=probs).rsample()
    hard = (relaxed >= 0.5).to(relaxed.dtype)
    return hard + relaxed - relaxed.detach()


def sanitize_probabilities_if_needed(probs):
    if torch.isfinite(probs).all():
        return probs
    return torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0).clamp(1e-6, 1 - 1e-6)


# Hierarchical embedding initialization module.
# return: (code_num, single_dim * code_levels.shape[1]).
class HierarchicalEmbedding(nn.Module):
    def __init__(self, code_levels, code_num_in_levels, code_dims):
        super(HierarchicalEmbedding, self).__init__()
        self.level_num = len(code_num_in_levels)
        code_levels = code_levels.to(torch.long)
        if torch.any(code_levels <= 0):
            raise ValueError("code_levels must be strictly positive before embedding lookup.")
        self.register_buffer('code_level_indices', code_levels - 1)
        self.level_embeddings = nn.ModuleList([nn.Embedding(code_num, code_dim) for level, (code_num, code_dim) in enumerate(zip(code_num_in_levels, code_dims))])

    def forward(self):
        embeddings = [self.level_embeddings[level](self.code_level_indices[:, level]) for level in range(self.level_num)]
        embeddings = torch.cat(embeddings, dim=1)
        return embeddings


# Hypergraph neural network module (personalized embeddings).
class HGNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, nhead, dropout_p, hgnn_model, device):
        super(HGNN, self).__init__()
        self.nlayer = nlayer
        self.HGNN_model = hgnn_model
        if hgnn_model == 'UniGINConv':
            self.convs = nn.ModuleList(
                [UniGINConv(nfeat, nhid, heads=nhead, dropout=0.)] +
                [UniGINConv(nhid * nhead, nhid, heads=nhead, dropout=0.) for _ in range(self.nlayer - 1)]
            )
            if self.nlayer > 0:
                self.conv_out = UniGINConv(nhid * nhead, nclass, heads=1, dropout=0.)
            else:
                self.conv_out = UniGINConv(nfeat, nclass, heads=1, dropout=0.)
        elif hgnn_model == 'UniSAGEConv':
            self.convs = nn.ModuleList(
                [UniSAGEConv(nfeat, nhid, heads=nhead, dropout=0.)] +
                [UniSAGEConv(nhid * nhead, nhid, heads=nhead, dropout=0.) for _ in range(self.nlayer - 1)]
            )
            if self.nlayer > 0:
                self.conv_out = UniSAGEConv(nhid * nhead, nclass, heads=1, dropout=0.)
            else:
                self.conv_out = UniSAGEConv(nfeat, nclass, heads=1, dropout=0.)
        elif hgnn_model == 'UniGATConv':
            self.convs = nn.ModuleList(
                [UniGATConv(nfeat, nhid, heads=nhead, dropout=0.)] +
                [UniGATConv(nhid * nhead, nhid, heads=nhead, dropout=0.) for _ in range(self.nlayer - 1)]
            )
            if self.nlayer > 0:
                self.conv_out = UniGATConv(nhid * nhead, nclass, heads=1, dropout=0.)
            else:
                self.conv_out = UniGATConv(nfeat, nclass, heads=1, dropout=0.)
        elif hgnn_model == 'UniGCNConv':
            self.convs = nn.ModuleList(
                [UniGCNConv(nfeat, nhid, heads=nhead, dropout=0.)] +
                [UniGCNConv(nhid * nhead, nhid, heads=nhead, dropout=0.) for _ in range(self.nlayer - 1)]
            )
            if self.nlayer > 0:
                self.conv_out = UniGCNConv(nhid * nhead, nclass, heads=1, dropout=0.)
            else:
                self.conv_out = UniGCNConv(nfeat, nclass, heads=1, dropout=0.)
        elif hgnn_model == 'UniGCNIIConv':
            self.prelude = nn.Linear(nfeat, nhid)
            self.convs = nn.ModuleList(
                [UniGCNIIConv(nhid, nhid, heads=nhead, dropout=0.)] +
                [UniGCNIIConv(nhid, nhid, heads=nhead, dropout=0.) for _ in range(self.nlayer - 1)]
            )
            if self.nlayer > 0:
                self.conv_out = UniGCNIIConv(nhid, nhid, heads=1, dropout=0.)
                self.postlude = nn.Linear(nhid, nclass)
            else:
                self.conv_out = UniGCNIIConv(nfeat, nfeat, heads=1, dropout=0.)
                self.postlude = nn.Linear(nfeat, nclass)
        elif hgnn_model == 'AllDeepSets':
            self.convs = nn.ModuleList(
                [AllSet(nfeat, nhid, heads=nhead, aggr='add', PMA=False, device=device, dropout=dropout_p)] +
                [AllSet(nhid, nhid, heads=nhead, aggr='add', PMA=False, device=device, dropout=dropout_p) for _ in range(self.nlayer - 1)]
            )
            if self.nlayer > 0:
                self.conv_out = AllSet(nhid, nclass, heads=nhead, aggr='add', PMA=False, device=device, dropout=dropout_p)
            else:
                self.conv_out = AllSet(nfeat, nclass, heads=nhead, aggr='add', PMA=False, device=device, dropout=dropout_p)
        elif hgnn_model == 'AllSetTransformer':
            self.convs = nn.ModuleList(
                [AllSet(nfeat, nhid, heads=nhead, aggr='mean', PMA=True, device=device, dropout=dropout_p)] +
                [AllSet(nhid, nhid, heads=nhead, aggr='mean', PMA=True, device=device, dropout=dropout_p) for _ in range(self.nlayer - 1)]
            )
            if self.nlayer > 0:
                self.conv_out = AllSet(nhid, nclass, heads=nhead, aggr='mean', PMA=True, device=device, dropout=dropout_p)
            else:
                self.conv_out = AllSet(nfeat, nclass, heads=nhead, aggr='mean', PMA=True, device=device, dropout=dropout_p)
        elif hgnn_model == 'HyperGCNConv':
            self.convs = nn.ModuleList(
                [HyperGCNConv(nfeat, nhid, True, device, dropout_p)] +
                [HyperGCNConv(nhid, nhid, True, device, dropout_p) for _ in range(self.nlayer - 1)]
            )
            if self.nlayer > 0:
                self.conv_out = HyperGCNConv(nhid, nclass, True, device, dropout_p)
            else:
                self.conv_out = HyperGCNConv(nfeat, nclass, True, device, dropout_p)
        else:
            print("Error: no selected hypergraph neural network model.")
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, X, V, E, H):
        if self.HGNN_model == "UniGCNConv":
            if self.nlayer > 0:
                for conv in self.convs:
                    X = conv(X, V, E, H)
                    X = self.act(X)
                    X = self.dropout(X)
            X = self.conv_out(X, V, E, H)
        elif self.HGNN_model == "UniGCNIIConv":
            if self.nlayer > 0:
                X = F.relu(self.prelude(X))
                X0 = X
                for conv in self.convs:
                    X = conv(X, V, E, X0, H)
                    X = self.act(X)
                    X = self.dropout(X)
                X = self.conv_out(X, V, E, X0, H)
                X = self.postlude(X)
            else:
                X = self.conv_out(X, V, E, X, H)
                X = self.postlude(X)
        else:
            if self.nlayer > 0:
                for conv in self.convs:
                    X = conv(X, V, E)
                    X = self.act(X)
                    X = self.dropout(X)
            X = self.conv_out(X, V, E)
        return F.leaky_relu(X)


# Obtain incident masks with probabilities (for sampling nodes).
class HSL_Layer_Part1(nn.Module):
    def __init__(self, emb_dim, edge_chunk_size=16):
        super(HSL_Layer_Part1, self).__init__()
        self.sample_MLP_1 = nn.Linear(emb_dim * 2, 256)
        self.act = nn.ReLU()
        self.sample_MLP_2 = nn.Linear(256, 1)
        self.edge_chunk_size = edge_chunk_size

    def forward(self, X, eX):
        hidden_dim = self.sample_MLP_1.out_features
        node_dim = X.shape[-1]
        weight = self.sample_MLP_1.weight
        bias = self.sample_MLP_1.bias
        node_term = F.linear(X, weight[:, :node_dim], bias)
        edge_term = F.linear(eX, weight[:, node_dim:], None)

        incident_mask_chunks = []
        for chunk_start in range(0, eX.shape[0], self.edge_chunk_size):
            chunk_end = min(chunk_start + self.edge_chunk_size, eX.shape[0])
            pair_hidden = node_term.unsqueeze(1) + edge_term[chunk_start:chunk_end].unsqueeze(0)
            pair_hidden = self.act(pair_hidden.reshape(-1, hidden_dim))
            pair_scores = self.sample_MLP_2(pair_hidden).view(X.shape[0], chunk_end - chunk_start)
            incident_mask_chunks.append(pair_scores)

        incident_mask_prob = torch.sigmoid(torch.cat(incident_mask_chunks, dim=1))
        incident_mask_prob = sanitize_probabilities_if_needed(incident_mask_prob)
        if len(incident_mask_prob.shape) == 1:
            incident_mask_prob = incident_mask_prob.unsqueeze(1)
        return incident_mask_prob


# Add false-negative nodes and sample nodes.
class HSL_Layer_Part2(nn.Module):
    def __init__(self, n_c, emb_dim, add_ratio, temperature):
        super(HSL_Layer_Part2, self).__init__()
        self.cos_weight = nn.Parameter(torch.randn(n_c, emb_dim))
        self.add_ratio = add_ratio
        self.temperature = temperature

    def forward(self, X, H, V, E, eX, incident_mask_prob):
        # First add false-negative nodes.
        node_fc = F.normalize(X.unsqueeze(1) * self.cos_weight.unsqueeze(0), p=2, dim=-1)
        edge_fc = F.normalize(eX.unsqueeze(1) * self.cos_weight.unsqueeze(0), p=2, dim=-1)
        S = torch.einsum('ncd,mcd->nm', node_fc, edge_fc) / self.cos_weight.shape[0]
        S[V, E] = -1e30
        topk_size = int(self.add_ratio * E.shape[0])
        if topk_size > 0:
            _, i = torch.topk(S.flatten(), topk_size, sorted=False)
            row = torch.div(i, S.shape[1], rounding_mode="floor")
            col = i % S.shape[1]
            # H can be a view into the padded batch tensor, so keep mutations off the
            # original storage that is reused by reconstruction loss.
            enriched_H = H.clone()
            enriched_H[row, col] = 1.0
        else:
            enriched_H = H
        # Node sampling.
        incident_mask_prob = sanitize_probabilities_if_needed(incident_mask_prob)
        incident_mask = relaxed_bernoulli_straight_through(incident_mask_prob, self.temperature)
        return enriched_H * incident_mask


# Phenotype representation module (aggregating node embeddings to get hypergraph embedding).
class HypergraphEmbeddingAggregator(nn.Module):
    def __init__(self, in_channel, hid_channel):
        super(HypergraphEmbeddingAggregator, self).__init__()
        self.temporal_edge_aggregator = nn.GRU(in_channel, hid_channel, 1)
        self.attention_context = nn.Linear(hid_channel, 1, bias=False)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, X, H):
        if H.dim() == 3:
            visit_emb = torch.matmul(H.transpose(1, 2).to(torch.float32), X).permute(1, 0, 2)
            hidden_states, _ = self.temporal_edge_aggregator(visit_emb)
            alpha = self.softmax(torch.squeeze(self.attention_context(hidden_states), -1))
            return torch.sum(alpha.unsqueeze(-1) * hidden_states, 0)
        visit_emb = torch.matmul(H.T.to(torch.float32), X)
        hidden_states, _ = self.temporal_edge_aggregator(visit_emb)
        alpha = self.softmax(torch.squeeze(self.attention_context(hidden_states), 1))
        hg_emb = torch.sum(alpha.unsqueeze(-1) * hidden_states, 0)
        return hg_emb


class HSLEncoder(nn.Module):
    def __init__(self, code_dims, HGNN_dim, after_HGNN_dim, HGNN_layer_num, nhead, num_TP, temperature, add_ratio, n_c, hid_state_dim, dropout, HGNN_model, device):
        super(HSLEncoder, self).__init__()
        self.HGNN_layer_num = HGNN_layer_num
        if HGNN_layer_num >= 0:
            self.firstHGNN = HGNN(sum(code_dims), HGNN_dim, after_HGNN_dim, HGNN_layer_num, nhead, dropout, HGNN_model, device)
        else:
            self.NoneHGNN = nn.Linear(sum(code_dims), after_HGNN_dim)
        self.num_TP = num_TP
        self.HSL_P1_combo = nn.ModuleList([HSL_Layer_Part1(after_HGNN_dim) for i in range(num_TP)])
        self.HSL_P2_combo = nn.ModuleList([HSL_Layer_Part2(n_c, after_HGNN_dim, addr, temp) for temp, addr in zip(temperature, add_ratio)])
        self.hyperG_emb_aggregator = HypergraphEmbeddingAggregator(after_HGNN_dim, hid_state_dim)

    def _batched_incident_mask_probs(self, X_1, eX):
        node_dim = X_1.shape[-1]
        edge_chunk_size = self.HSL_P1_combo[0].edge_chunk_size
        sample_mlp_1_weight = torch.stack([layer.sample_MLP_1.weight for layer in self.HSL_P1_combo], dim=0)
        sample_mlp_1_bias = torch.stack([layer.sample_MLP_1.bias for layer in self.HSL_P1_combo], dim=0)
        sample_mlp_2_weight = torch.stack([layer.sample_MLP_2.weight.squeeze(0) for layer in self.HSL_P1_combo], dim=0)
        sample_mlp_2_bias = torch.stack([layer.sample_MLP_2.bias.squeeze(0) for layer in self.HSL_P1_combo], dim=0)
        hidden_dim = sample_mlp_1_weight.shape[1]

        node_term = torch.einsum('nd,khd->knh', X_1, sample_mlp_1_weight[:, :, :node_dim]) + sample_mlp_1_bias.unsqueeze(1)
        edge_term = torch.einsum('md,khd->kmh', eX, sample_mlp_1_weight[:, :, node_dim:])

        incident_mask_chunks = []
        for chunk_start in range(0, eX.shape[0], edge_chunk_size):
            chunk_end = min(chunk_start + edge_chunk_size, eX.shape[0])
            pair_hidden = node_term.unsqueeze(2) + edge_term[:, None, chunk_start:chunk_end, :]
            pair_hidden = torch.relu(pair_hidden.reshape(self.num_TP, -1, hidden_dim))
            pair_scores = torch.einsum('ksh,kh->ks', pair_hidden, sample_mlp_2_weight) + sample_mlp_2_bias.unsqueeze(1)
            incident_mask_chunks.append(pair_scores.view(self.num_TP, X_1.shape[0], chunk_end - chunk_start))

        incident_mask_prob = torch.sigmoid(torch.cat(incident_mask_chunks, dim=2))
        return sanitize_probabilities_if_needed(incident_mask_prob)

    def _batched_sample_temporal_phenotypes(self, X_1, H, V, E, eX, incident_mask_probs):
        cosine_weight = torch.stack([layer.cos_weight for layer in self.HSL_P2_combo], dim=0)
        node_fc = F.normalize(X_1.unsqueeze(0).unsqueeze(2) * cosine_weight.unsqueeze(1), p=2, dim=-1)
        edge_fc = F.normalize(eX.unsqueeze(0).unsqueeze(2) * cosine_weight.unsqueeze(1), p=2, dim=-1)
        similarity = torch.einsum('kncd,kmcd->knm', node_fc, edge_fc) / cosine_weight.shape[1]
        similarity[:, V, E] = -1e30

        topk_sizes = [int(layer.add_ratio * E.shape[0]) for layer in self.HSL_P2_combo]
        enriched_H = H.unsqueeze(0).expand(self.num_TP, -1, -1)
        if any(topk_size > 0 for topk_size in topk_sizes):
            enriched_H = enriched_H.clone()
            for k, topk_size in enumerate(topk_sizes):
                if topk_size <= 0:
                    continue
                _, flat_idx = torch.topk(similarity[k].reshape(-1), topk_size, sorted=False)
                row = torch.div(flat_idx, similarity.shape[2], rounding_mode="floor")
                col = flat_idx % similarity.shape[2]
                enriched_H[k, row, col] = 1.0

        temperatures = incident_mask_probs.new_tensor([float(layer.temperature) for layer in self.HSL_P2_combo]).view(self.num_TP, 1, 1)
        incident_mask = relaxed_bernoulli_straight_through(incident_mask_probs, temperatures)
        return enriched_H * incident_mask

    def forward(self, X, H):
        # First update hypergraph embeddings via message passing.
        V, E = torch.nonzero(H, as_tuple=True)
        if self.HGNN_layer_num >= 0:
            X_1 = self.firstHGNN(X, V, E, H)
        else:
            X_1 = F.leaky_relu(self.NoneHGNN(X))
        eX = scatter(X_1[V], E, dim=0, reduce='mean')
        # Split into different channels; each channel extracts one temporal phenotype and gets the embedding.
        if self.num_TP > 1:
            incident_mask_probs = self._batched_incident_mask_probs(X_1, eX)
            TPs = self._batched_sample_temporal_phenotypes(X_1, H, V, E, eX, incident_mask_probs)
            latent_TPs = self.hyperG_emb_aggregator(X_1, TPs)
        else:
            incident_mask_probs = self.HSL_P1_combo[0](X_1, eX)
            TPs = self.HSL_P2_combo[0](X_1, H, V, E, eX, incident_mask_probs)
            latent_TPs = self.hyperG_emb_aggregator(X_1, TPs)
        return TPs, latent_TPs, incident_mask_probs


# Input reconstruction module.
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden, X):
        output = F.relu(torch.mm(input.unsqueeze(0), X))
        output, hidden = self.gru(output, hidden)
        output = self.sigmoid(self.out(output[0]))
        return output, hidden



class HSL_Decoder(nn.Module):
    def __init__(self, latent_TP_dim, num_TP, proj_dim, code_num, device):
        super(HSL_Decoder, self).__init__()
        self.to_context = nn.Linear(latent_TP_dim * num_TP, proj_dim)
        self.reconstruct_net = DecoderRNN(proj_dim, code_num)
        self.code_num = code_num

    def forward(self, latent_TP, visit_len, H, X):
        # Combine latent_TP through concatenation and calculate context vector.
        decoder_hidden = self.to_context(torch.reshape(latent_TP, (-1,))).view(1, -1)
        # Decode and map to the target dimension.
        target_tensor = H.T
        decoder_input = torch.zeros(self.code_num, device=X.device, dtype=target_tensor.dtype)
        reconstructed_visits = []
        for di in range(visit_len):
            output, decoder_hidden = self.reconstruct_net(decoder_input, decoder_hidden, X)
            reconstructed_visits.append(output.squeeze(0))
            decoder_input = target_tensor[di]
        return torch.stack(reconstructed_visits, dim=1)


# Prediction module.
class FinalClassifier(nn.Module):
    def __init__(self, in_channel, code_num, key_dim, SA_head, num_TP):
        super(FinalClassifier, self).__init__()
        self.num_TP = num_TP
        if num_TP > 1:
            self.w_key = nn.Linear(in_channel, key_dim)
            self.w_query = nn.Linear(in_channel, key_dim)
            self.w_value = nn.Linear(in_channel, key_dim)
            self.multihead_attn = nn.MultiheadAttention(key_dim, SA_head)
            self.tp_attention = nn.Linear(key_dim, 1, bias=False)
        self.classifier = nn.Linear(in_channel, code_num)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, latent_tp):
        if self.num_TP > 1:
            keys = self.w_key(latent_tp)
            querys = self.w_query(latent_tp)
            values = self.w_value(latent_tp)
            sa_output, _ = self.multihead_attn(querys, keys, values, need_weights=False)
            alpha = self.softmax(torch.squeeze(self.tp_attention(sa_output), -1))
            separate_pred = self.softmax(self.classifier(latent_tp))
            final_pred = torch.sum(separate_pred * torch.unsqueeze(alpha, -1).expand(-1, -1, separate_pred.shape[-1]), -2)
            return final_pred, alpha
        else:
            final_pred = self.softmax(self.classifier(latent_tp))
            return final_pred, latent_tp.new_ones(latent_tp.shape[0], 1)


class SHy(nn.Module):
    def __init__(self, code_levels, single_dim, HGNN_dim, after_HGNN_dim, HGNN_layer_num, nhead, num_TP, temperature, add_ratio, n_c, hid_state_dim, dropout, key_dim, SA_head, HGNN_model, device):
        super(SHy, self).__init__()
        # Hierarchical embedding for medical codes.
        code_num_in_levels = (np.max(code_levels, axis=0)).tolist()
        code_levels = torch.as_tensor(code_levels, dtype=torch.long)
        code_dims = [single_dim] * code_levels.shape[1]
        self.hier_embed_layer = HierarchicalEmbedding(code_levels, code_num_in_levels, code_dims)
        # Extract multiple temporal phenotypes via HSL.
        self.encoder = HSLEncoder(code_dims, HGNN_dim, after_HGNN_dim, HGNN_layer_num, nhead, num_TP, temperature, add_ratio, n_c, hid_state_dim, dropout, HGNN_model, device)
        # Calculate reconstruction loss to ensure fidelity.
        self.decoder = HSL_Decoder(hid_state_dim, num_TP, sum(code_dims), code_levels.shape[0], device)
        # Final classifier.
        self.fclf = FinalClassifier(hid_state_dim, code_levels.shape[0], key_dim, SA_head, num_TP)

    def forward(self, Hs, visit_lens):
        # Hierarchical embedding for medical codes.
        X = self.hier_embed_layer()
        # Obtain multiple temporal phenotypes (and latent representations) via HSL & Decoder.
        tp_list = []; latent_tp_list = []; recon_H_list = []
        for i in range(len(Hs)):
            visit_len = _visit_len_to_int(visit_lens[i])
            patient_h = Hs[i][:, :visit_len]
            tp, latent_tp, _ = self.encoder(X, patient_h)
            tp_list.append(tp)
            latent_tp_list.append(latent_tp)
            recon_H_list.append(self.decoder(latent_tp, visit_len, patient_h, X))
        # Classify based on the temporal phenotype embeddings.
        pred, alphas = self.fclf(torch.stack(latent_tp_list))
        return pred, tp_list, recon_H_list, alphas

