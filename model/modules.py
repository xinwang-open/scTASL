import torch
import torch.distributions as D
import torch.nn.functional as F
from torch import nn

EPS = 1e-7
TOTAL_COUNT = 1e4


class Feature_wise_gating(nn.Module):
    """
    Lightweight self-attention operating along the feature dimension.
    Produces a gating vector to reweight hidden features.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = hidden_dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.softmax(q * k * self.scale, dim=-1)

        return scores * v


class GraphConv(nn.Module):
    r"""
    Graph convolution (propagation only)
    """

    def forward(self, input: torch.Tensor, eidx: torch.Tensor, enorm: torch.Tensor) -> torch.Tensor:
        r"""
        Forward propagation

        Parameters
        ----------
        input
            Input data (:math:`n_{vertices} \times n_{features}`)
        eidx
            Vertex indices of edges (:math:`2 \times n_{edges}`)
        enorm
            Normalized weight of edges (:math:`n_{edges}`)


        Returns
        -------
        result
            Graph convolution result (:math:`n_{vertices} \times n_{features}`)
        """
        sidx, tidx = eidx  # source index and target index
        message = input[sidx] * enorm.unsqueeze(1)  # n_edges * n_features
        res = torch.zeros_like(input)
        tidx = tidx.unsqueeze(1).expand_as(message)  # n_edges * n_features
        res.scatter_add_(0, tidx, message)
        return res


class GraphEncoder(nn.Module):
    r"""
    Graph encoder producing a diagonal Gaussian D.Normal(loc, scale) for each vertex.

    Architecture
    ------------
    - Learnable node embeddings: vrepr (vnum x out_dim)
    - Two residual blocks:
        * LayerNorm -> PReLU -> GraphConv -> Dropout -> Residual
    - Linear projections to (mu, sigma)
    """

    def __init__(self, vnum: int, out_dim: int) -> None:
        super().__init__()
        self.vnum = vnum
        self.out_dim = out_dim

        # Node embeddings with Xavier init: (vnum, out_dim)
        self.vrepr = nn.Parameter(torch.empty(vnum, out_dim))
        nn.init.xavier_uniform_(self.vrepr)

        # GraphConv layers (assumed signature: conv(x, eidx, enorm) -> (V, C))
        self.conv1 = GraphConv()
        self.conv2 = GraphConv()

        # Normalization, activation, dropout
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.act = nn.PReLU()
        self.drop = nn.Dropout(p=0.2)

        # Projection to Gaussian parameters
        self.loc = nn.Linear(out_dim, out_dim)
        self.std_lin = nn.Linear(out_dim, out_dim)

        # Scale clipping range
        self.scale_min = 1e-4
        self.scale_max = 10.0

    def forward(self, eidx: torch.Tensor, enorm: torch.Tensor) -> D.Normal:
        r"""
        Parameters
        ----------
        eidx : (2, n_edges) LongTensor
            Edge indices [source_idx, target_idx].
        enorm : (n_edges,) Tensor
            Normalized edge weights.

        Returns
        -------
        dist : torch.distributions.Normal
            Per-vertex Normal(loc, scale), shape (vnum, out_dim).
        """
        # Initial node features from learnable embeddings: (V, C)
        x = self.vrepr

        # ----- Block 1 -----
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h, eidx, enorm)
        h = self.drop(h)
        x = x + h  # residual connection

        # ----- Block 2 -----
        h = self.norm2(x)
        h = self.act(h)
        h = self.conv2(h, eidx, enorm)
        h = self.drop(h)
        x = x + h  # residual connection

        # Gaussian parameters
        loc = self.loc(x)  # (V, C)
        pre_scale = self.std_lin(x)  # (V, C)
        scale = F.softplus(pre_scale) + EPS
        scale = torch.clamp(scale, min=self.scale_min, max=self.scale_max)

        return D.Normal(loc, scale)


class GraphDecoder(nn.Module):
    r"""
    Graph decoder using inner product of node embeddings.

    Given node latents v and edge index eidx, produces a Bernoulli
    distribution per edge.
    """

    def forward(self, v: torch.Tensor, eidx: torch.Tensor) -> D.Bernoulli:
        r"""
        Parameters
        ----------
        v : (n_vertices, dim) Tensor
            Node latent representations.
        eidx : (2, n_edges) LongTensor
            Edge indices [source_idx, target_idx].

        Returns
        -------
        dist : torch.distributions.Bernoulli
            Bernoulli over edges with logits of shape (n_edges,).
        """
        sidx, tidx = eidx  # source and target index, (n_edges,)
        logits = (v[sidx] * v[tidx]).sum(dim=1)  # inner product per edge
        return D.Bernoulli(logits=logits)


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        emb_size: int,
        output_dim: int,
        dropout_rate: float = 0.0,
        use_library_norm: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.emb_size = emb_size
        self.num_topics = output_dim
        self.TOTAL_COUNT = 1e4
        self.use_library_norm = use_library_norm

        # -------- Block 1: Linear -> PReLU -> BN -> Dropout --------
        self.fc1 = nn.Linear(input_dim, emb_size)
        self.act1 = nn.PReLU()
        self.bn1 = nn.BatchNorm1d(emb_size)
        self.drop1 = nn.Dropout(p=dropout_rate)

        # -------- Block 2: Linear -> PReLU -> BN -> Dropout --------
        self.fc2 = nn.Linear(emb_size, emb_size)
        self.act2 = nn.PReLU()
        self.bn2 = nn.BatchNorm1d(emb_size)
        self.drop2 = nn.Dropout(p=dropout_rate)

        # -------- Lightweight feature-wise gating (self-attention-like) --------
        self.self_attn1 = Feature_wise_gating(emb_size)
        self.attn_norm1 = nn.LayerNorm(emb_size)
        self.self_attn2 = Feature_wise_gating(emb_size)
        self.attn_norm2 = nn.LayerNorm(emb_size)

        # -------- Posterior heads: map hidden -> (mu, sigma) --------
        self.mu = nn.Linear(emb_size, output_dim, bias=False)
        self.sigma = nn.Linear(emb_size, output_dim, bias=False)

    def compute_l(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute library size per sample (sum over features).
        Returns:
            l: [B, 1], clamped to avoid division by zero.
        """
        return x.sum(dim=1, keepdim=True).clamp_min(EPS)

    def normalize(self, x: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        """
        Library-size normalization + log1p transform.
        x_norm = log( 1 + x * (TOTAL_COUNT / l) )
        """
        return (x * (self.TOTAL_COUNT / l)).log1p()

    def preprocess(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.use_library_norm:
            l = self.compute_l(x)
            h = self.normalize(x, l)
            return h, l
        else:
            h = x
            return h, None

    def encode_base(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input into a hidden representation (before posterior heads).

        Returns:
            h: [B, emb_size]  hidden features (e.g., 256-d)
            l: [B, 1]         library size used for normalization
        """
        # l = self.compute_l(x)
        # h = self.normalize(x, l)
        h, l = self.preprocess(x)

        # ----- Block 1 -----
        h = self.drop1(self.bn1(self.act1(self.fc1(h))))
        h = self.attn_norm1(h + self.self_attn1(h))  # residual + LayerNorm

        # ----- Block 2 -----
        h = self.drop2(self.bn2(self.act2(self.fc2(h))))
        h = self.attn_norm2(h + self.self_attn2(h))  # residual + LayerNorm

        return h, l

    def posterior(self, h: torch.Tensor) -> D.Normal:
        """
        Build diagonal Gaussian posterior q(z|x) from hidden features.

        Args:
            h: [B, emb_size]

        Returns:
            dist: Normal(mu, sigma) with mu/sigma shape [B, output_dim]
        """
        mu = self.mu(h)
        sigma = F.softplus(self.sigma(h)) + EPS
        return D.Normal(mu, sigma)

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        """
        Forward pass.

        Args:
            x: [B, input_dim] raw counts/features
            return_hidden: if True, also return hidden features h

        Returns:
            dist: Normal posterior q(z|x)
            l:    library size [B, 1]
            h:    hidden features [B, emb_size] (only if return_hidden=True)
        """
        h, l = self.encode_base(x)
        dist = self.posterior(h)

        if return_hidden:
            return dist, l, h
        return dist, l


class Decoder(nn.Module):
    """Negative Binomial decoder used for VAE reconstruction."""

    def __init__(
        self,
        input_dim: int,
        emb_size: int,
        output_dim: int,
        n_batches,
        use_library_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.emb_size = emb_size
        self.output_dim = output_dim
        self.use_library_norm = use_library_norm

        self.layers = nn.Sequential(
            nn.Linear(input_dim, emb_size),
            nn.BatchNorm1d(emb_size),
            nn.PReLU(),
            nn.Linear(emb_size, emb_size),
            nn.BatchNorm1d(emb_size),
            nn.PReLU(),
        )
        self.output_layer = nn.Linear(emb_size, output_dim)
        self.log_theta = nn.Parameter(torch.zeros(n_batches, output_dim))

    def forward(self, x: torch.Tensor, b: torch.Tensor, l: torch.Tensor | None = None):
        x = self.layers(x)
        x = self.output_layer(x)

        if self.use_library_norm:
            mu = F.softmax(x, dim=1) * l
        else:
            mu = F.softplus(x) + EPS

        log_theta = self.log_theta[b]

        return D.NegativeBinomial(log_theta.exp(), logits=(mu + EPS).log() - log_theta)


class Decoder_g(nn.Module):
    """
    Negative Binomial decoder with residual local attention over features.
    out_dim: number of genes/peaks; n_batches: batch-effect conditioning size.
    """

    def __init__(self, out_dim: int, n_batches: int = 1) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.n_batches = n_batches

        self.scale_lin = nn.Parameter(torch.zeros(n_batches, out_dim))
        self.bias = nn.Parameter(torch.zeros(n_batches, out_dim))
        self.log_theta = nn.Parameter(torch.zeros(n_batches, out_dim))

        self.local_attn = nn.Conv1d(
            in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False
        )
        self.res_norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        c_hidden: torch.Tensor,  # [B, d_c]
        f_hidden: torch.Tensor,  # [G, d_c]
        b: torch.Tensor,  # [B] LongTensor, batch id
        l: torch.Tensor,  # [B] or [B,1], library size
    ):
        H = c_hidden @ f_hidden.t()  # [B, G]

        H_local = self.local_attn(H.unsqueeze(1)).squeeze(1)  # [B, G]
        H = H + F.relu(H_local)
        H = self.res_norm(H)  # [B, G]

        scale = F.softplus(self.scale_lin[b])  # [B, G] > 0
        logit_mu = scale * H + self.bias[b]  # [B, G]

        l = l.reshape(-1, 1)  # [B, 1]
        comp = F.softmax(logit_mu, dim=1)  # [B, G]
        mu = comp * l  # [B, G]

        log_theta = self.log_theta[b]  # [B, G]
        theta = log_theta.exp()  # [B, G] > 0

        return D.NegativeBinomial(theta, logits=(mu + EPS).log() - log_theta)


class NaiveAffineTransform(nn.Module):
    def __init__(self, input_dim, z_dim, affine_num, reverse=False) -> None:
        super().__init__()
        self.input_dim = input_dim

        # affine matrix init with identity affine
        direction = -1 if reverse else 1
        self.affine_matrix = nn.Parameter(
            torch.stack(
                [torch.randn(input_dim, input_dim).flatten() * direction for _ in range(affine_num)]
            )
        )
        self.affine_offset = nn.Parameter(torch.randn(affine_num, input_dim))

        # regressor for the affine transform selection
        self.fc_loc = nn.Sequential(
            nn.Linear(input_dim, z_dim), nn.ReLU(True), nn.Linear(z_dim, affine_num)
        )

    def forward(self, x):
        soft_idx = F.softmax(self.fc_loc(x), dim=-1)
        affine_matrix = torch.mm(soft_idx, self.affine_matrix)
        affine_matrix = affine_matrix.view(-1, self.input_dim, self.input_dim)  # [b, d, d]
        affine_offset = torch.mm(soft_idx, self.affine_offset)
        affine_offset = affine_offset.unsqueeze(-1)  # [b, d, 1]

        output = x.unsqueeze(-1)
        output = torch.bmm(affine_matrix, output) + affine_offset
        output = output.squeeze(-1)
        return output


class Predictor(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.2) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        self.output_dim = output_dim

        self.w_p = nn.Sequential(
            nn.Linear(self.input_dim, 2 * self.input_dim),
            nn.BatchNorm1d(2 * self.input_dim),
            nn.PReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(2 * self.input_dim, self.output_dim),
        )

    def forward(self, x: torch.Tensor):
        x = self.w_p(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim, output_dim, n_batches, dropout_rate=0.2) -> None:
        super().__init__()
        self.n_batches = n_batches

        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        self.output_dim = output_dim
        ptr_dim = self.input_dim + self.n_batches

        self.w_d = nn.Sequential(
            nn.Linear(ptr_dim, 256),
            nn.PReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, self.output_dim),
        )

    def forward(self, x: torch.Tensor, b: torch.Tensor):
        if self.n_batches:
            b_one_hot = F.one_hot(b, num_classes=self.n_batches)
            x = torch.cat([x, b_one_hot], dim=1)

        x = self.w_d(x)

        return x


class Prior(torch.nn.Module):
    """Diagonal Gaussian prior."""

    def __init__(self, loc: float = 0.0, std: float = 1.0) -> None:
        super().__init__()
        loc = torch.as_tensor(loc, dtype=torch.get_default_dtype())
        std = torch.as_tensor(std, dtype=torch.get_default_dtype())
        self.register_buffer("loc", loc)
        self.register_buffer("std", std)

    def forward(self) -> D.Normal:
        return D.Normal(self.loc, self.std)
