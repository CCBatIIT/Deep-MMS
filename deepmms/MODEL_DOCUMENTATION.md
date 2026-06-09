# Deep-MMS Model Documentation

## Notation

| Symbol | Meaning |
|---|---|
| $x \in \mathbb{R}^{B \times D}$ | Input batch; $B$ = batch size, $D = N \cdot 3$ = flattened coordinates |
| $N$ | Number of heavy atoms |
| $\mathbf{r}_i \in \mathbb{R}^3$ | Position of atom $i$ |
| $d_{ij} = \|\mathbf{r}_i - \mathbf{r}_j\|$ | Pairwise distance |
| $d$ | Embedding / hidden dimension |
| $L$ | Number of hidden layers / blocks |
| $K$ | Latent dimensionality |
| $\hat{x}$ | Reconstructed coordinates |
| $\sigma(\cdot)$ | Sigmoid; $\text{SiLU}(x) = x \sigma(x)$; $\text{GELU}$ as standard |
| $\text{LN}$ | Layer normalisation |
| $\text{sg}(\cdot)$ | Stop-gradient (no backprop through) |
| $\text{MLP}_d^L$ | $L$-layer MLP with hidden width $d$ and ReLU activations |
| $W, b$ | Learnable weight matrices and biases |

All models share the same interface:

```
encode(x)          →  (μ_z, log σ²_z)        shape (B, K) each
decode(z)          →  x̂                        shape (B, D)
__call__(x, z_rng) →  (x̂,  μ_z,  log σ²_z)
```

Models that lack a stochastic encoder (VQ-VAE, Flow, MAE) return zeros for
$\log \sigma^2_z$ so the trainer interface is uniform.

The reparameterisation trick is used by all VAE-family models:

$$z = \mu_z + \sigma_z \odot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)$$

---

## 1. BatchNorm\_VAE  ·  `train.py`

**Design intent.** Symmetric square MLP. All hidden layers have the same width
as the input ($d = D$), making each Dense layer a full square matrix
transformation — the closest learned approximation to an invertible linear map.
Dropout and optional BatchNorm provide regularisation.

### Encoder

$$h^{(0)} = x$$

$$h^{(\ell)} = \text{Dropout}\!\left(\text{(BN)}\left(\text{ReLU}\!\left(W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}\right)\right)\right), \quad \ell = 1 \ldots L$$

$$\mu_z = W_\mu h^{(L)} + b_\mu, \qquad \log\sigma^2_z = W_\sigma h^{(L)} + b_\sigma$$

### Decoder (mirrors encoder in reverse)

$$g^{(0)} = z \in \mathbb{R}^K$$

$$g^{(\ell)} = \text{Dropout}\!\left(\text{(BN)}\left(\text{ReLU}\!\left(W^{(L+1-\ell)} g^{(\ell-1)} + b^{(L+1-\ell)}\right)\right)\right)$$

$$\hat{x} = W_{\text{out}} g^{(L)} + b_{\text{out}}$$

### Loss

$$\mathcal{L} = \log\!\sqrt{\frac{1}{B}\sum_i \text{RMSD}(x_i, \hat{x}_i)^2}$$

### Critical difference from other models

No inductive bias toward molecular geometry. Every atom coordinate is treated
as an independent feature; atom $i$ influences the reconstruction of atom $j$
only through the weight matrices, with no explicit encoding of physical
proximity.

**Use when:** establishing a baseline or when training speed is the priority.

**Key parameters:** `dropout_rates` (length = $L$), `is_batchnorm`

---

## 2. BetaVAE  ·  `train_beta_vae.py`

**Design intent.** Identical MLP architecture to BatchNorm\_VAE with one
change: an explicit KL penalty weighted by $\beta > 1$ is added to the loss.
This information-bottleneck pressure encourages the $K$ latent dimensions to
become statistically independent and individually interpretable.

### Encoder / Decoder

Identical to BatchNorm\_VAE.

### Loss

$$\mathcal{L}_\beta = \log\sqrt{\overline{\text{RMSD}^2}} + \beta \cdot \mathcal{L}_{\text{KL}}$$

$$\mathcal{L}_{\text{KL}} = \text{KL}\!\left[q(z|x) \,\|\, \mathcal{N}(0,I)\right] = -\tfrac{1}{2}\sum_{k=1}^{K}\!\left(1 + \log\sigma^2_{z,k} - \mu^2_{z,k} - \sigma^2_{z,k}\right)$$

### Critical difference

The additional $\beta \mathcal{L}_{\text{KL}}$ term creates information pressure.
As $\beta \to \infty$ the model is forced to discard all but the most critical
$K$ bits of information; as $\beta \to 0$ it reduces to BatchNorm\_VAE.
Disentangled latents mean that sweeping $z_k$ independently (perturbation
analysis) changes one physical mode without perturbing others.

**Use when:** the goal is interpretable latent dimensions — each latent should
correspond to a single conformational mode (helix opening, loop displacement,
etc.).

**Key parameters:** `beta` (default 4.0); larger $\beta$ = more disentanglement
but higher reconstruction error.

---

## 3. VQVAE  ·  `train_vq_vae.py`

**Design intent.** Replace the continuous Gaussian posterior with a discrete
codebook. Each encoded frame is assigned to the nearest of $C$ learned
prototype vectors. The discrete bottleneck forces the model to partition
conformation space into a finite vocabulary of states.

### Encoder

Same MLP as BatchNorm\_VAE $\to$ continuous pre-quantised representation:

$$z_e = \text{MLP}(x) \in \mathbb{R}^K$$

### Vector Quantisation

$$k^* = \arg\min_{k \in \{1,\ldots,C\}} \|z_e - e_k\|^2$$

$$z_q = e_{k^*} \in \mathbb{R}^K$$

Straight-through gradient (gradient passes through quantisation unchanged):

$$z_{\text{st}} = z_e + \text{sg}(z_q - z_e)$$

### Decoder

$$\hat{x} = \text{MLP}(z_{\text{st}})$$

### Loss

$$\mathcal{L}_{\text{VQ}} = \log\sqrt{\overline{\text{RMSD}^2}} + \underbrace{\|\text{sg}(z_e) - z_q\|^2}_{\text{codebook loss}} + \lambda\underbrace{\|z_e - \text{sg}(z_q)\|^2}_{\text{commitment loss}}$$

The codebook loss moves $e_{k^*}$ toward the encoder output; the commitment
loss (weight $\lambda$, controlled by `vq_commitment_weight`, default 1.0) keeps the encoder output close to the
codebook.

### Critical difference

The latent is **discrete**: each frame maps to an integer index $k^* \in
\{1,\ldots,C\}$. Reconstruction is exact given the codebook entry. There is no
posterior to sample from — instead you can decode any codebook entry to generate
a canonical conformational state. The codebook size $C$ acts as a
macro-state count analogous to the number of clusters in KMeans.

**Use when:** you want the model to discover and name a finite set of
conformational states automatically, or when downstream clustering analysis is
the goal.

**Key parameters:** `codebook_size` (default 512); `vq_commitment_weight`
(default 1.0 in trainer JSON).

---

## 4. EquivariantVAE  ·  `train_equivariant.py`

**Design intent.** The encoder uses only SE(3)-invariant pairwise distances as
input features (SchNet-style), so the latent representation is immune to rigid
body rotation and translation. The decoder predicts Cartesian coordinates
directly — valid because training data is pre-superposed.

### Encoder

Reshape $x$ to atom positions $\{\mathbf{r}_i\}_{i=1}^N$.

**Pairwise distances:**

$$d_{ij} = \|\mathbf{r}_i - \mathbf{r}_j\| \in \mathbb{R}$$

**Radial basis function expansion** ($n_{\text{rbf}}$ Gaussians with centres
$\mu_k$ uniformly spaced from 0 to $r_c$):

$$\psi_k(d_{ij}) = \exp\!\left(-\gamma_k (d_{ij} - \mu_k)^2\right) \cdot u(d_{ij})$$

where $u(r) = 1 - \tfrac{1}{2}(r/r_c)^2$ for $r < r_c$ is a smooth envelope.

**Initial atom features:**

$$\mathbf{h}_i^{(0)} = \mathbf{0} \in \mathbb{R}^d$$

**Interaction layer** (repeated $n_{\text{int}}$ times):

$$\mathbf{m}_{ij} = W_f \,\boldsymbol{\psi}(d_{ij}) \odot \mathbf{h}_j \qquad (\text{filter network})$$

$$\mathbf{h}_i^{(\ell+1)} = \mathbf{h}_i^{(\ell)} + \sum_{j \neq i} \mathbf{m}_{ij}$$

**Global readout:**

$$\mathbf{h}_{\text{global}} = \frac{1}{N} \sum_{i=1}^N \mathbf{h}_i^{(n_{\text{int}})}$$

$$\mu_z = W_\mu \mathbf{h}_{\text{global}}, \qquad \log\sigma^2_z = W_\sigma \mathbf{h}_{\text{global}}$$

### Decoder

Standard MLP from $z \in \mathbb{R}^K$ to $\hat{x} \in \mathbb{R}^D$.

### Loss

Standard RMSD loss (same as BatchNorm\_VAE).

### Critical difference

The encoder **cannot see individual coordinates** — only pairwise distances.
This means the latent encodes internal geometry (bond lengths, angles,
inter-atomic contacts) but not global orientation. For pre-aligned data this
removes wasted representational capacity. Atom–atom coupling is explicitly
modelled through the message-passing sum, unlike MLPs where coupling emerges
only through matrix products.

**Use when:** you want physically motivated invariance and explicit modelling of
atomic proximity, particularly for proteins where contact patterns drive
function.

**Key parameters:** `embed_dim`, `n_rbf` (default 32), `cutoff_dist` (default
1.0 nm), `n_interactions` (default 3).

---

## 5. PerceiverVAE  ·  `train_perceiver.py`

**Design intent.** Reduce the $\mathcal{O}(N^2)$ attention cost of
TransformerVAE to $\mathcal{O}(N \cdot M)$ by introducing $M \ll N$ learned
latent query vectors that cross-attend to the atom tokens.

### Encoder

**Atom embedding:**

$$\mathbf{a}_i = W_a \mathbf{r}_i + \mathbf{p}_i \in \mathbb{R}^d$$

where $\mathbf{p}_i$ is a learnable positional encoding for atom $i$.

**Latent array** (learned, shared across the batch):

$$Q \in \mathbb{R}^{M \times d}, \quad M \ll N$$

**Cross-attention** (latent queries attend to atom keys/values):

$$Q' = Q + \text{MHA}\!\left(Q,\; K = A,\; V = A\right) \quad \text{cost: } \mathcal{O}(M \cdot N)$$

$$Q'' = Q' + \text{MHA}(Q', Q', Q') + \text{FFN}(Q'') \quad \text{cost: } \mathcal{O}(M^2)$$

Repeated for $L$ cross + self-attention blocks.

**Readout:**

$$\mathbf{h} = \frac{1}{M}\sum_{m=1}^M Q''_m$$

$$\mu_z = W_\mu \mathbf{h}, \qquad \log\sigma^2_z = W_\sigma \mathbf{h}$$

### Decoder

$$\mathbf{q}_i = W_z z + \mathbf{p}_i \quad \text{(atom queries from latent)}$$

$$\hat{\mathbf{r}}_i = \text{TransformerBlocks}(\mathbf{q})[i] \cdot W_{\text{out}}$$

$$\hat{x} = \text{flatten}(\hat{\mathbf{r}}_1, \ldots, \hat{\mathbf{r}}_N)$$

### Loss

Standard RMSD loss.

### Critical difference

Same expressive encoder as TransformerVAE but at **linear cost in $N$**
(cross-attention is $\mathcal{O}(MN)$ not $\mathcal{O}(N^2)$). For HIV1p
($N=1599$), this reduces attention operations from $\sim$2.5M to $\sim$100K
per layer with $M=64$. The latent queries learn which atom subsets to
summarise.

**Use when:** $N$ is large (BR, HIV1p) and TransformerVAE is too slow or
GPU-memory-limited.

**Key parameters:** `embed_dim`, `num_heads` (default 4), `n_latent_queries`
(default 64), `ffn_mult` (default 4.0).

---

## 6. HierarchicalVAE  ·  `train_hierarchical.py`

**Design intent.** Decompose the latent into two levels corresponding to
different timescales: a small **global** latent $z_1$ encoding slow, large-scale
motions and a larger **local** latent $z_2$ encoding fast, small-scale
fluctuations conditioned on $z_1$.

### Encoder

**Shared backbone:**

$$\mathbf{h} = \text{MLP}_d^L(x)$$

**Level 1 (global):**

$$\mu_{z_1} = W_{\mu_1} \mathbf{h}, \qquad \log\sigma^2_{z_1} = W_{\sigma_1} \mathbf{h}$$

$$z_1 \sim q(z_1|x) = \mathcal{N}(\mu_{z_1}, \sigma^2_{z_1} I), \qquad \dim(z_1) = K_1 = \lfloor K/4 \rfloor$$

**Level 2 (local, conditioned on $z_1$):**

$$\mathbf{h}_2 = \text{MLP}([\mathbf{h},\; z_1])$$

$$\mu_{z_2} = W_{\mu_2} \mathbf{h}_2, \qquad \log\sigma^2_{z_2} = W_{\sigma_2} \mathbf{h}_2$$

$$z_2 \sim q(z_2|x, z_1), \qquad \dim(z_2) = K - K_1$$

### Decoder

$$\hat{x} = \text{MLP}([z_1,\; z_2]) \in \mathbb{R}^D$$

### Priors

$$p(z_1) = \mathcal{N}(0, I)$$

$$p(z_2|z_1) = \mathcal{N}(\mu_{\text{prior}}(z_1),\; I) \qquad \text{(learned prior network)}$$

### Loss

$$\mathcal{L}_{\text{HVAE}} = \log\sqrt{\overline{\text{RMSD}^2}} + \text{KL}[q(z_1|x) \,\|\, p(z_1)] + \text{KL}[q(z_2|x,z_1) \,\|\, p(z_2|z_1)]$$

### Critical difference

The latent has **structure**: $z_1$ is forced to capture the minimum
information needed to describe the slow modes, and $z_2$ captures residual
fast fluctuations. Decoding from $z_1$ alone (ignoring $z_2$) gives the
time-averaged structure. This is physically natural for MD — domain motions
(ms timescale) are encoded in $z_1$; local fluctuations (ps timescale) in $z_2$.

**Use when:** the system has known multi-scale dynamics (large domains + local
loops, e.g. BPTI, HIV1p).

---

## 7. SE3TransformerVAE  ·  `train_se3.py`

**Design intent.** PaiNN-style equivariant message passing. Each atom carries
two types of features: **scalar** features $\mathbf{s}_i \in \mathbb{R}^{d_s}$
(SE(3)-invariant) and **vector** features $\mathbf{v}_i \in \mathbb{R}^{d_v
\times 3}$ (SE(3)-equivariant — they rotate with the molecule). The
vector channel encodes directional information without requiring explicit
spherical harmonics.

### Encoder

**Initialisation:**

$$\mathbf{s}_i^{(0)} = W_0 \mathbf{1} \in \mathbb{R}^{d_s}, \qquad \mathbf{v}_i^{(0)} = \mathbf{0} \in \mathbb{R}^{d_v \times 3}$$

**Message functions** (one per layer $\ell$):

$$\phi_s(d_{ij}, \mathbf{s}_j) = \text{MLP}([d_{ij}, \mathbf{s}_j]) \to (\mathbf{m}_s, \mathbf{m}_v) \in \mathbb{R}^{d_s} \times \mathbb{R}^{d_v}$$

**Scalar update** (invariant aggregation):

$$\mathbf{s}_i^{(\ell+1)} = \mathbf{s}_i^{(\ell)} + \sum_{j \neq i} \mathbf{m}_s(d_{ij}, \mathbf{s}_j^{(\ell)})$$

**Vector update** (equivariant aggregation via direction):

$$\hat{\mathbf{r}}_{ij} = \frac{\mathbf{r}_j - \mathbf{r}_i}{d_{ij} + \epsilon} \in \mathbb{R}^3$$

$$\mathbf{v}_i^{(\ell+1)} = \mathbf{v}_i^{(\ell)} + \sum_{j \neq i} \mathbf{m}_v(d_{ij}, \mathbf{s}_j^{(\ell)}) \cdot \hat{\mathbf{r}}_{ij}$$

If the molecule rotates by $R$: $\hat{\mathbf{r}}_{ij} \to R\hat{\mathbf{r}}_{ij}$, so
$\mathbf{v}_i \to R\mathbf{v}_i$ — equivariant by construction.

**Global readout** (scalar features only, invariant):

$$\mathbf{h}_{\text{global}} = \frac{1}{N}\sum_i \mathbf{s}_i^{(L)}$$

$$\mu_z = W_\mu \mathbf{h}_{\text{global}}, \qquad \log\sigma^2_z = W_\sigma \mathbf{h}_{\text{global}}$$

### Decoder

Standard MLP.

### Loss

Standard RMSD loss.

### Critical difference

Like EquivariantVAE, the latent is SE(3)-invariant. Unlike EquivariantVAE, the
model maintains a separate **vector feature channel** that explicitly carries
directional information throughout message passing — an atom's features encode
not just *how much* it interacts with neighbours but *in which direction*. This
gives richer geometric representation at the cost of the additional
$d_v$-dimensional vector per atom per layer.

**Use when:** you need the strongest possible geometric inductive bias and can
afford the extra parameters ($d_v$ vector channels).

**Key parameters:** `d_scalar` (default 64), `d_vector` (default 16),
`n_mp_layers` (default 3), `cutoff_dist` (default 1.0 nm).

---

## 8. TransformerVAE  ·  `train_transformer.py`

**Design intent.** Treat each atom as a sequence token. Multi-head
self-attention allows every atom to attend to every other atom in a single
layer, capturing long-range correlations that would require many MLP layers to
learn implicitly.

### Encoder

**Atom tokenisation:**

$$\mathbf{a}_i = W_a \mathbf{r}_i + \mathbf{p}_i \in \mathbb{R}^d, \qquad i = 1\ldots N$$

where $\mathbf{p}_i$ is a learnable per-atom positional encoding.

**Self-attention block** (pre-norm, $H$ heads, repeated $L$ times):

$$\text{head}_h = \text{softmax}\!\left(\frac{QW_h^Q (KW_h^K)^\top}{\sqrt{d/H}}\right) VW_h^V$$

$$\text{MHA}(A) = \text{concat}(\text{head}_1,\ldots,\text{head}_H) W^O + A \quad \text{(residual)}$$

$$A' = A + \text{FFN}(\text{LN}(\text{MHA}(\text{LN}(A)))) \qquad \text{cost per block: } \mathcal{O}(N^2 d)$$

**Global readout:**

$$\mu_z = W_\mu \left(\frac{1}{N}\sum_i A'_i\right), \qquad \log\sigma^2_z = W_\sigma \left(\frac{1}{N}\sum_i A'_i\right)$$

### Decoder

$$\mathbf{q}_i = (W_z z + \mathbf{p}_i), \quad \hat{\mathbf{r}}_i = W_{\text{out}} \text{TransformerBlocks}(\mathbf{q})[i]$$

### Loss

Standard RMSD loss.

### Critical difference

The **attention pattern is learned jointly** over all atom pairs: the model
decides which atom–atom interactions are most informative. Compared to
EquivariantVAE, distances are not explicitly computed — the model must discover
geometric relationships from raw coordinates. Compared to PerceiverVAE, this
has $\mathcal{O}(N^2)$ complexity but no fixed-size bottleneck in the attention.

**Use when:** $N$ is moderate (OX, DA, CR) and you want maximum expressivity.

**Key parameters:** `embed_dim`, `num_heads` (default 4), `ffn_mult` (default
4.0).

---

## 9. MambaVAE  ·  `train_mamba.py`

**Design intent.** Replace $\mathcal{O}(N^2)$ attention with a selective state
space model that processes atoms as a 1D sequence at $\mathcal{O}(N)$ cost.
The **selective scan** allows the model to decide, for each atom, which
information from preceding atoms to carry forward and which to forget.

### Selective SSM (one Mamba block)

**Input projection:**

$$u_t = \text{Dense}(a_t) \in \mathbb{R}^{d_{\text{inner}}}, \quad d_{\text{inner}} = \lfloor \alpha \cdot d \rfloor$$

**Input-dependent (selective) parameters:**

$$\Delta_t = \text{softplus}(W_\Delta a_t), \quad B_t = W_B a_t, \quad C_t = W_C a_t$$

**Discretised state matrices:**

$$\bar{A}_t = \exp(\Delta_t \odot A), \qquad \bar{B}_t = \Delta_t \odot B_t$$

where $A \in \mathbb{R}^{d_{\text{state}}}$ is a learnable diagonal state
matrix initialised negative.

**Parallel selective scan** (via associative scan):

$$\text{operator: } (A_1, b_1) \oplus (A_2, b_2) = (A_1 A_2,\; A_1 b_2 + b_1)$$

$$h_t = \bar{A}_t \odot h_{t-1} + \bar{B}_t \odot u_t \qquad \text{(resolved in parallel)}$$

$$y_t = C_t \cdot h_t \in \mathbb{R}^d$$

**Residual:** output = $y_t + a_t$.

### Encoder

$$A = \{a_i\}_{i=1}^N = \text{AtomEmbed}(x) \quad \xrightarrow{\text{Mamba} \times L} \quad A' \quad \xrightarrow{\text{mean pool}} \quad \mathbf{h}$$

$$\mu_z = W_\mu \mathbf{h}, \qquad \log\sigma^2_z = W_\sigma \mathbf{h}$$

### Decoder

$z \to \text{tile}(N) \to \text{Mamba blocks} \to W_{\text{out}} \to \hat{x}$

### Loss

Standard RMSD loss.

### Critical difference

The SSM recurrence has **asymmetric causality**: atom $i$ can only see atoms
$1, \ldots, i-1$ directly (in the forward pass). Unlike attention, there is no
all-pairs interaction. The selective gate $\Delta_t$ determines how much
history is retained at each position — the model learns to reset its memory
when moving between structurally distinct regions (e.g. secondary structure
boundaries). At $\mathcal{O}(N)$ cost this is the most scalable sequence model.

**Use when:** $N$ is very large (HIV1p, future full-atom trajectories) and
memory is the constraint.

**Key parameters:** `embed_dim`, `d_state` (default 16), `d_inner_mult`
(default 2.0).

---

## 10. RealNVPFlow  ·  `train_flow.py`

**Design intent.** Learn an exact, invertible bijection $f: \mathbb{R}^D \to
\mathbb{R}^D$ such that $f(x) = z \sim \mathcal{N}(0,I)$. No stochastic
encoder; reconstruction is exact up to floating-point precision.

### Affine Coupling Layer

Split $x = [x_1; x_2]$ at dimension $\lfloor D/2 \rfloor$:

$$(\mathbf{s}, \mathbf{t}) = \text{MLP}(x_1) \quad \text{(scale and translate)}$$

$$y_1 = x_1, \qquad y_2 = x_2 \odot \exp(\mathbf{s}) + \mathbf{t}$$

$$\log \left|\det \frac{\partial y}{\partial x}\right| = \sum_k s_k$$

Alternate which half is transformed each layer; compose $L$ layers.

### Actnorm (after each coupling layer)

$$y = (x - \mathbf{b}) \oslash \mathbf{s}$$

$$\log|\det J| = -\sum_k \log|s_k|$$

where $\mathbf{b}, \mathbf{s}$ are data-initialised per-channel scale/bias
(makes activations unit-mean, unit-variance on the first batch).

### Forward (encoding) and inverse (decoding)

$$x \xrightarrow{f_1 \circ \cdots \circ f_L} z = f(x), \qquad z \xrightarrow{f_L^{-1} \circ \cdots \circ f_1^{-1}} \hat{x} = f^{-1}(z)$$

The exact inverse exists by construction: $f^{-1}(f(x)) = x$.

### Loss (negative log-likelihood)

$$\mathcal{L}_{\text{NLL}} = \frac{1}{2}\|z\|^2 - \sum_{\ell=1}^{L} \log\left|\det J_{f_\ell}\right|$$

### Critical difference

**No information is discarded in the encoder.** The latent $z$ has the same
dimension as $x$ ($K_{\text{reported}} \leq D$, only leading coordinates are
exported). Reconstruction is exact: $\hat{x} = f^{-1}(f(x)) = x$. The
NLL loss is a principled probability density, unlike the RMSD-based VAE losses.
Because the log-likelihood is tractable, you can compute exact free energies
$G = -k_BT \ln p(x)$ from the learned density.

**Use when:** you need exact reconstructions, density estimation, or free
energy calculations from the latent.

**Key parameters:** `n_coupling_layers` (default 8, set via JSON).

---

## 11. MaskedAutoencoder  ·  `train_mae.py`

**Design intent.** Teach the encoder to build a holistic representation of the
full molecular structure by forcing it to reconstruct coordinates of randomly
masked atoms from the unmasked subset alone.

### Training forward pass

**Random masking** (mask ratio $\rho = 0.75$):

$$\mathcal{V} \subset \{1,\ldots,N\}, \quad |\mathcal{V}| = \lfloor(1-\rho)N\rfloor \quad \text{(visible atoms)}$$

**Encoder** (ViT-style, operates on visible atoms only):

$$\mathbf{a}_i = W_a \mathbf{r}_i + \mathbf{p}_i \quad \forall i \in \mathcal{V}$$

$$\{\mathbf{a}'_i\}_{i \in \mathcal{V}} = \text{TransformerBlocks}\!\left(\{\mathbf{a}_i\}_{i \in \mathcal{V}}\right)$$

$$\mathbf{h} = \frac{1}{|\mathcal{V}|}\sum_{i \in \mathcal{V}} \mathbf{a}'_i$$

$$\mu_z = W_\mu \mathbf{h}, \qquad \log\sigma^2_z = W_\sigma \mathbf{h}$$

**Decoder** (operates on all $N$ atom positions):

$$\mathbf{q}_i = \begin{cases} W_{\text{enc}} \mathbf{a}'_i + \mathbf{p}_i & i \in \mathcal{V} \\ \mathbf{m} + \mathbf{p}_i & i \notin \mathcal{V} \end{cases}$$

where $\mathbf{m}$ is a shared learnable mask token.

$$\hat{\mathbf{r}}_i = W_{\text{out}} \text{TransformerBlocks}(\mathbf{q})[i]$$

### Training loss (masked atoms only)

$$\mathcal{L}_{\text{MAE}} = \frac{1}{|\mathcal{M}|} \sum_{i \notin \mathcal{V}} \|\mathbf{r}_i - \hat{\mathbf{r}}_i\|^2 \qquad \mathcal{M} = \{1,\ldots,N\} \setminus \mathcal{V}$$

### Inference

No masking applied; all atoms are passed through the encoder, RMSD computed
over all atoms.

### Critical difference

The model is **forced to understand molecular structure holistically** — it
cannot copy coordinates from the encoder input. To reconstruct a masked atom's
position, it must infer it from the spatial relationships of visible atoms.
Unlike the VAE family, the latent is not Gaussian — it is a compressed summary
of the partial observation $\mathcal{V}$.

**Use when:** you want strong structural representations without committing to a
specific latent dimensionality during training; the 75% masking pressure is
stronger than KL regularisation for learning compact encodings.

**Key parameters:** `embed_dim`, `mask_ratio` (default 0.75), `num_heads`
(default 4).

---

## 12. NEATAutoencoder + NEATTrainer  ·  `train_neat.py`

**Design intent.** Start with a minimal tanh MLP and grow it when training
stalls. Weights are optimised by **OpenAI Evolution Strategies** rather than
backpropagation, which is immune to vanishing gradients and requires no
differentiable loss.

### Architecture

Same symmetric encoder–decoder as BatchNorm\_VAE but:

$$h^{(\ell)} = \tanh\!\left(W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}\right)$$

Initial topology: $d_0 = \min(64, D)$, $L_0 = n_{\text{start layers}}$.

### Topology growth

When $\max F[-W:] - \max F[:-W] < \tau$:

$$\text{append layer: } W^{(L+1)} \sim \mathcal{N}(0, 0.01^2), \quad L \leftarrow L+1$$

New weights initialise near zero so network output is almost unchanged
(minimal structural innovation).

### Weight optimisation (OpenES, one generation)

$$\theta_k = \theta + \sigma\varepsilon_k, \quad \varepsilon_k \sim \mathcal{N}(0,I), \quad k = 1\ldots N_{\text{pop}} \quad \text{(antithetic pairs)}$$

$$F_k = -\overline{\text{RMSD}}(f_{\theta_k}), \qquad \tilde{F}_k = \frac{\text{rank}(F_k)}{N_{\text{pop}} - 1} - 0.5 \in [-0.5, 0.5]$$

$$\hat{g} = -\frac{1}{N_{\text{pop}} \sigma} \sum_{k=1}^{N_{\text{pop}}} \tilde{F}_k \varepsilon_k \qquad \text{(rank-normalised gradient estimate)}$$

$$\theta \leftarrow \theta + \text{Adam}(\hat{g})$$

### Critical difference

**No backpropagation.** The loss landscape is evaluated through perturbations;
the gradient is estimated from fitness differences across the population. This
makes the method applicable to non-differentiable loss functions (e.g. cluster
purity) and avoids vanishing/exploding gradients in deep networks. The topology
grows automatically rather than being fixed before training.

**Use when:** you want architecture search without specifying depth in advance,
or when the loss function is not differentiable.

**Key parameters:** `neat_start_dim`, `neat_start_layers`, `es_population`
(default 50), `es_sigma` (default 0.05), `es_lr` (default 0.01),
`neat_plateau_window` (default 200), `neat_plateau_thr` (default 0.005).

---

## 13. KANVAE  ·  `train_kan.py`

**Design intent.** Replace every Dense layer with a KAN layer where each
edge carries a learnable univariate B-spline function. Instead of fixed
activation functions on nodes, the network learns the shape of each
input–output relationship on edges.

### KAN layer

For input $x \in \mathbb{R}^{n_{\text{in}}}$, output $y \in
\mathbb{R}^{n_{\text{out}}}$:

**B-spline basis** (order $p$, grid $\{t_k\}_{k=0}^{G}$ uniform on $[-3,3]$):

$$B_{k,0}(x) = \mathbf{1}[t_k \leq x < t_{k+1}]$$

$$B_{k,p}(x) = \frac{x - t_k}{t_{k+p} - t_k} B_{k,p-1}(x) + \frac{t_{k+p+1} - x}{t_{k+p+1} - t_{k+1}} B_{k+1,p-1}(x)$$

**Edge activation** (one set of coefficients $\mathbf{c}_{ij}$ per edge):

$$\psi_{ij}(x_i) = \sum_{k=1}^{G+p} c_{ijk} B_{k,p}(x_i)$$

**Layer output:**

$$y_j = \sum_{i=1}^{n_{\text{in}}} \psi_{ij}(x_i) + w_j \cdot \text{SiLU}(x_j) \cdot \mathbf{1}[n_{\text{in}} = n_{\text{out}}]$$

The SiLU residual applies only when input and output dimensions match.

### Encoder / Decoder

Stack $L$ KAN layers in the encoder (widths from `hidden_layers`) with two
final Dense heads for $\mu_z$ and $\log\sigma^2_z$. Mirror in the decoder.

### Loss

Standard RMSD loss.

### Critical difference

The **activation function is data-driven and per-edge**, not fixed globally.
The spline coefficients can represent any smooth univariate function, allowing
the network to automatically discover the right nonlinearity for each
input–output pair. Unlike ReLU/GELU which apply the same transformation to all
edges in a layer, each KAN edge has independent behaviour. This may allow more
compact representations when the true input–output relationships are simple
(e.g. near-linear or single-peaked) but may overfit on small datasets.

**Use when:** exploring what nonlinearities are actually needed for molecular
coordinate compression; useful for post-hoc analysis of which atom–coordinate
pairs have complex vs. simple relationships.

**Key parameters:** `kan_n_grid` (default 5), `kan_order` (default 3),
`embed_dim` for layer width.

---

## Summary comparison

| Model | Encoder structure | Latent type | Atom coupling | Complexity | Loss |
|---|---|---|---|---|---|
| BatchNorm\_VAE | Square MLP | Gaussian | Implicit (weights) | $\mathcal{O}(D^2 L)$ | log RMSD |
| BetaVAE | Square MLP | Disentangled Gaussian | Implicit | $\mathcal{O}(D^2 L)$ | log RMSD + β KL |
| VQVAE | MLP + codebook | Discrete index | Implicit | $\mathcal{O}(D^2 L + KC)$ | RMSD + VQ losses |
| EquivariantVAE | SchNet distances | Gaussian (invariant) | Explicit (message passing) | $\mathcal{O}(N^2 d L)$ | log RMSD |
| PerceiverVAE | Cross-attention | Gaussian | Explicit (cross-attn) | $\mathcal{O}(NMd L)$ | log RMSD |
| HierarchicalVAE | Branched MLP | Two-level Gaussian | Implicit | $\mathcal{O}(D^2 L)$ | RMSD + KL(z1) + KL(z2) |
| SE3TransformerVAE | PaiNN MP | Gaussian (equivariant) | Explicit (direction-aware) | $\mathcal{O}(N^2 d L)$ | log RMSD |
| TransformerVAE | Self-attention | Gaussian | Explicit (self-attn) | $\mathcal{O}(N^2 d L)$ | log RMSD |
| MambaVAE | Selective SSM | Gaussian | Explicit (sequential) | $\mathcal{O}(N d L)$ | log RMSD |
| RealNVPFlow | Coupling layers | Exact (same dim) | Implicit (alternating halves) | $\mathcal{O}(DL)$ | NLL |
| MaskedAutoencoder | Partial self-attn | Gaussian | Explicit (self-attn) | $\mathcal{O}((1-\rho)^2 N^2 d)$ | Masked RMSD |
| NEATAutoencoder | Growing tanh MLP | Gaussian | Implicit | $\mathcal{O}(d^2 L)$, $L$ grows | ES fitness |
| KANVAE | B-spline MLP | Gaussian | Implicit (per-edge) | $\mathcal{O}(d^2 G L)$ | log RMSD |
