# Diamond
Title: Error-controlled non-additive interaction discovery in machine learning models
Venue: Nature Machine Intelligence

## Problem
find (unadditive) feature interactions and control FDR
## Why existing methods are not enough
- Traditional attribution / pairwise ranking methods may highlight pairs with strong marginal effects rather than true non-additive interactions.
- Ranking alone does not provide a principled threshold.
- Therefore FDR cannot be reliably controlled.
## Main pipeline
1.knockoff generation
2.non-additivity distillation
3.interaction selection / FDR control

## Knockoff generation
$$ \mathbf{X} = \{x_i\}_{i=1}^n \in \mathbb{R}^{n \times p} $$
$$ \mathbf{\tilde{X} }= \{x_i\}_{i=1}^n \in \mathbb{R}^{n \times p} $$
$$ \mathbf{Y} = \{y_i\}_{i=1}^n \in \mathbb{R}^{n \times 1} $$
- $ (X,\tilde{X}) \stackrel{d}{=} (X, \tilde{X})_{\text{swap}(j)}$ $\stackrel{d}{=}$ means equality in distribution
- knockoffs are independent of the response given the original features$\tilde{X} \bot Y|X$
- If $X_j$ is really useful, it should be more important than $\tilde{X}_j$. if $X_j$ is not useful, it should perform about the same as $\tilde{X}_j$.
$X_{original}$: n × p+$X_{knockoff}$: n × p = $X_{concat}$: n × 2p
#### Gaussian setting
In the Gaussian setting, ${\bf{X}} \approx {\mathcal{N}}(0,\varSigma )$,with cov matrix$\varSigma \in {{\mathbb{R}}}^{p\times p}$.

Conditional distribution is $\tilde{{\bf{X}}}| {\bf{X}} \approx N\left({\bf{X}}-{\rm{diag}}\{{\bf{s}}\}{\varSigma }^{-1}{\bf{X}},2{\rm{diag}}\{{\bf{s}}\}-{\rm{diag}}\{{\bf{s}}\}{\varSigma }^{-1}{\rm{diag}}\{{\bf{s}}\}\right)$. where ${\rm{diag}}\{{\bf{s}}\}$

Joint Distribution is 
$({\bf{X}},\tilde{{\bf{X}}}) \approx {\mathcal{N}}\left(\left(\begin{array}{l}{\bf{0}}\\ {\bf{0}}\end{array}\right),\left(\begin{array}{ll}\qquad\varSigma\qquad\;\varSigma -{\rm{diag}}\{{\bf{s}}\}\\ \varSigma -{\rm{diag}}\{{\bf{s}}\}\qquad\varSigma \end{array}\right)\right)$ 
The conditional covariance matrix is diagonal with strictly positive diagonal elements, making it positive definite.
``` python
class GaussianKnockoffs:
    """
    Class GaussianKnockoffs
    Knockoffs for a multivariate Gaussian model
    """

    def __init__(self, Sigma, method="equi", mu=[], tol=1e-3):
        """
        Constructor
        :param model  : A multivariate Gaussian model object containing the covariance matrix
        :param method : Specifies how to determine the free parameters of Gaussian knockoffs.
                        Allowed values: "equi", "sdp" (default "equi")
        :return:
        """

        if len(mu) == 0:
            self.mu = np.zeros((Sigma.shape[0],))
        else:
            self.mu = mu
        self.p = len(self.mu)
        self.Sigma = Sigma
        self.method = method
        # 初始化参数
        # Initialize Gaussian knockoffs by computing either SDP or min(Eigs)

        if self.method == "equi":
            lambda_min = linalg.eigh(self.Sigma, eigvals_only=True, eigvals=(0, 0))[0]
            s = min(1, 2 * (lambda_min - tol))
            self.Ds = np.diag([s] * self.Sigma.shape[0])
        elif self.method == "sdp": 
            self.Ds = np.diag(solve_sdp(self.Sigma, tol=tol))# diag(s)
        else:
            raise ValueError("Invalid Gaussian knockoff type: " + self.method)
        self.SigmaInvDs = linalg.lstsq(self.Sigma, self.Ds)[0]#Σ^{-1} D
        self.V = 2.0 * self.Ds - np.dot(self.Ds, self.SigmaInvDs)#公式中的2{\rm{diag}}\{{\bf{s}}\}-{\rm{diag}}\{{\bf{s}}\}{\varSigma }^{-1}{\rm{diag}}\{{\bf{s}}\}
        self.LV = np.linalg.cholesky(self.V + 1e-10 * np.eye(self.p))
        if linalg.eigh(self.V, eigvals_only=True, eigvals=(0, 0))[0] <= tol:
            warnings.warn(
                "Warning...........\
            The conditional covariance matrix for knockoffs is not positive definite. \
            Knockoffs will not have any power."
            )

    def generate(self, X):
        """
        Generate knockoffs for the multivariate Gaussian model
        :param X: A matrix of observations (n x p)
        :return: A matrix of knockoff variables (n x p)
        """
        n, p = X.shape
        muTilde = X - np.dot(X - np.tile(self.mu, (n, 1)), self.SigmaInvDs)
        N = np.random.normal(size=muTilde.shape)
        return muTilde + np.dot(N, self.LV.T)
```
Compared to equi, SDP is typically:more refined, closer to the optimal solution, slower to compute, more dependent on solver stability.
**SDP**
Semidefinite Programming
The semidefinite programming method solves an optimization problem using CVXOPT.
$minimize\ c^Tx$ and $F(x) \succeq0$
```
maximize    Σ s_j#让sj尽量大
subject to  2Σ ≽ diag(s) + δI # 半正定约束
            0 ≤ s ≤ 1
```
$D_s = diag(s_1,s_2,...,s_p)$

**equi**
equicorrelated
$D_s = sI$
#### Not in Gaussian setting
Not in Gaussian setting, Knockoff Machine Handles data from non-Gaussian distributions. It uses a six-layer fully connected network to learn a mapping that generates knockoffs from the original features. 
``` python
x_cat = concat(x, noise)  # Shape: (batch_size, 2p)
x_cat[:, 0::2] = x
x_cat[:, 1::2] = noise # 奇数列是noise
```
##### loss
$loss = \gamma \cdot MMD_{full}+\gamma \cdot MMD_{swap}+\lambda \cdot moments+\delta \cdot corr$
Minimize MMD
``` python
        # Combine the loss functionsD_
        loss = (
            self.GAMMA * mmd_full
            + self.GAMMA * mmd_swap
            + self.LAMBDA * loss_moments
            + self.DELTA * loss_corr
        )
```
**MMD = Maximum Mean Discrepancy**
- mmd_full: Distance between $(X,\tilde{X})$ and $(\tilde{X},X)$
  Its goal is to force the model to satisfy the exchangeability property
- mmd_swap: Distance between $(X,\tilde{X})$ and $(X,\tilde{X})_{swap}$
  the original features with the knockoff features on the selected dimensions, ensuring consistency is maintained even after random local swaps.

**loss_moments**
It approximates the statistical moment relationship between the original features and the knockoffs, specifically:Mean Variance Covariance
- $D_{meam} = X.mean(0)-Xk.mean(0)$
- $\left\| \varSigma -\varSigma_{kk}\right\|^2+\left\|Mask\odot (\Sigma - \Sigma_{xk})\right\|_F^2$ The first term is the difference between the covariance of the knockoff and the original covariance.
The second term is the difference between the cross-covariance of the original and the knockoff and that of a specific target structure. A mask is a matrix with the same shape as the original matrix, whose elements are typically: 1 Keep this element; 0 Discard this element
- $corr\_XXk = (X_{scaled} \cdot Xk_{scaled}).mean(0) $ then $(corr_{XXk}-target\_corr)^2$
  $\mathrm{corr}_{X\tilde X}^{(j)} = \frac{1}{n}\sum_{i=1}^{n} X^{\mathrm{scaled}}_{ij}\,\tilde X^{\mathrm{scaled}}_{ij} \qquad j=1,\dots,p$
  
#### GAM
knockoffgan(gen_KnockoffGAN)

``` python
knockoff_func, release_memory = gen_KnockoffGAN.KnockoffGAN(X, "Uniform", seed=seed)#均匀分布
X_knockoff = knockoff_func(X)
release_memory()
```
#### Deep learning-based generation
deepknockoffs(gen_DeepKnockoffs)

``` python
X_knockoff = gen_DeepKnockoffs.DeepKnockoffs(X, batch_size=512, test_size=0, nepochs=10, epoch_length=100)#是为了Maximum training data？
```
batch size of 512 samples,10 epochs and each length = 100
#### vaeknockoff(gen_VAEKnockoff)

``` python
X_knockoff = gen_VAEKnockoff.train_vae_knockoff(X, n_epoch=100, mb_size=256)
```
#### Knockoffsdiag
gen_Knockoffsdiag
#### GAM VS Deep learning VS VAE VS Knockoffsdiag
| Characteristic | KnockoffGAN | DeepKnockoffs | VAEKnockoff | Knockoffsdiag |
| --- | --- | --- | --- | --- |
| Algorithmic Basis（算法基础） | GAN（生成对抗网络） | Deep Learning（深度学习） | VAE（变分自编码器） | Sequential Sampling（序贯采样） |
| Training Required（是否需要训练） | Yes（是） | Yes（是） | Yes（是） | No (statistical)（否，基于统计方法） |
| GPU Compatible（GPU 兼容性） | Unknown（未知） | Yes (implicit)（是，隐式支持） | Yes (implicit)（是，隐式支持） | No（否） |
| Memory Management（内存管理） | Manual (release_memory)（手动，需调用release_memory参数） | Automatic（自动） | Automatic（自动） | Automatic（自动） |
| Parallelization（并行化支持） | No（不支持） | Batch-based（基于批次） | Batch-based（基于批次） | Thread-based (n_jobs)（基于线程，需设置n_jobs参数） |
| Return Type（返回类型） | Function + cleanup（函数对象 + 需手动清理） | Direct array（直接返回数组） | Direct array（直接返回数组） | Direct array（直接返回数组） |
| Seed Parameter（随机种子参数） | Explicit（显式支持） | None visible（无可见参数） | None visible（无可见参数） | Explicit（显式支持） |
| Distribution Assumption（分布假设） | Configurable ("Uniform")（可配置，默认均匀分布） | Data-driven（数据驱动，无强假设） | Data-driven（数据驱动，无强假设） | Diagonal covariance（对角协方差假设） |
| Invocation Pattern（调用模式） | Two-step（两步式） | Single-step（单步式） | Single-step（单步式） | Single-step（单步式） |
#### invalid knockoffs
## Non-additivity distillation
If distillation don't exist, FDR can not be controled.
${{\bf{E}}}^{{\rm{2D}}}={\left[{e}_{ij}\right]}_{i,\;j = 1}^{2p}\in {{\mathbb{R}}}^{2p\times 2p}$
$e_{ij}$ denotes the raw interaction importance between the \(i\)-th and \(j\)-th features among the \(2p\) original-and-knockoff features. This matrix serves as the basis for non-additivity distillation.
Many interaction measures favor pairs with strong marginal effects, even when no true non-additive interaction exists.
${e}_{ij}={s}_{ij}+{g}_{i}({e}_{i})+{g}_{j}({e}_{j})+b({I}_{ij})+{\varepsilon }_{ij}$
${e}_{ij}$ pairwise importance between i and j.
${e}_{i}$ importance of i.
${e}_{j}$ importance of j.
$b({I}_{ij})$ Feature-specific bias.$l_{ij} \in \{0,1\}^{2p}$ one-hot Vector
${\varepsilon }_{ij}$ random noise
$\mathop{\min}\limits_{b,{g}_{1},{g}_{2},\cdots}\mathop{\sum}\limits_{i < j}{w}_{ij}\cdot {\left\Vert {e}_{ij}-{g}_{i}({e}_{i})-{g}_{j}({e}_{j})-b({I}_{ij})\right\Vert }^{2}$ 
#### Why distillation is needed
Without non-additivity distillation, raw interaction scores may be dominated by marginal or main effects, making the comparison between original and knockoff interactions unreliable and thereby undermining effective FDR control.

#### pair categorization and attr_onehot / pair_onehot
- Metadata
- Ground Truth
  true_indices:Ground Truth is a list of indices for the truly important individual features
  true_pairs:Feature interation
- Categorizations
  false_pairs original_knockoff_pairs knockoff_knockoff_pairs
``` python 
true_indices = sorted(import_gt)
true_pairs = [(i, j) for features in inter_gt # e.g.inter_gt = [{0,1}, {2,3,4}]
              for i, j in itertools.combinations(sorted(list(features)), 2)]  #i,j range from [0,feature_num-1] 
false_pairs = [(i, j) for i, j in itertools.product(range(num_features), range(num_features)) 
               if i < j and (i, j) not in true_pairs]# false interation pairs where no true interaction exists between the original features
original_knockoff_pairs = [(i, j) for i, j in 
                           itertools.product(range(num_features), range(num_features, 2*num_features)) 
                           if i < j and i != j - num_features]# Generate pairs of original features and their corresponding knockoff features, but exclude each original feature from being paired with its own knockoff
knockoff_knockoff_pairs = [(i, j) for i, j in 
                           itertools.product(range(num_features, 2*num_features), 
                                            range(num_features, 2*num_features)) # Generate all unordered pairs of Knockoff features
                           if i < j]
```

``` python
results_to_save = {
    'seed': seed,
    'func_num': func_num,  # or 'dataset': dataset
    'true_indices': true_indices,
    'true_pairs': true_pairs,
    'false_pairs': false_pairs,
    'original_knockoff_pairs': original_knockoff_pairs,
    'knockoff_knockoff_pairs': knockoff_knockoff_pairs,
    'interactions': interactions.tolist(),  # (2p, 2p) matrix
    'attributions': attributions.tolist()   # (2p,) vector
}
```
``` python
pred = model.predict(X_test)#obs = 原始 interaction 分数 pred = GAM 认为“按背景信息来说你应该有的 interaction 分数”
residual = np.abs(obs - pred)#这个 pair 的 interaction 分数有多“超出背景预期”
round_interactions_df['obs'] = obs
round_interactions_df['pred'] = pred
round_interactions_df['calibrated_interaction'] = residual
```
**DeepPINK and other models(KAN XGBoost / LightGBM Factorization Machine (FM)) output **attributions(n,2p)** individual feature importance scores and **interactions(n,2p,2p)** pairwise feature interaction matrix**
``` python
attributions = explainer.attributions(test_dataset, baseline=baseline, use_expectation=use_expectation)  
interactions = explainer.interactions(test_dataset, baseline=baseline, use_expectation=use_expectation)
```
absolute values focus on the magnitude of the effect, appropriate for feature selection
(real.py)
``` python
'pair_onehot': np.eye(len(attributions))[i] + np.eye(len(attributions))[j]# one-hot 方式标记出该交互对涉及的两个特征
'attr_onehot': np.eye(len(attributions))[i] * attributions[i] + np.eye(len(attributions))[j] * attributions[j]#从attributions得出每个特征的重要程度得分
```
#### logistic regression for propensity scores
The probability that a pair belongs to the Original rather than the Knockoff, given the attribution-related covariates.
``` python
X_train = attr_onehot #pair 中两个特征的 attribution 编码
y_train = round_interactions_df['type'].apply(lambda x: 1 if x == 'Original' else 0).values
model = LogisticRegression().fit(X_train, y_train)
propensity_scores = model.predict_proba(X_train)[:, 1]
```
logistic regression = softmax regression(k = 2)
$P(y=1|x) = \frac{1}{1 + e^{-(w^T x + b)}}$
Among these Sigmoid function $\sigma(z) = \frac{1}{1+e^{-z}}$,Map the linear combination $w^Tx+b$ in to the interval(0,1)
#### IPTW and propensity-weighted GAM
**IPTW Calculation**
Stabilized Inverse Probability of Treatment Weights
$weight = \frac{p(select)}{p}$ or $\frac{1}{p}$
$weight = \frac{p(select)}{p}$

If we take $A = 1$as the original group, $A = 0$ as the knockoff group,**stablilized IPTM** is generally written as:

$w_i = \begin{cases}
\frac{P(A=1)}{e(X_i)} & A_i = 1 \\[6pt]
\frac{P(A=0)}{1-e(X_i)} & A_i = 0
\end{cases}$
``` python
round_interactions_df['IPTW'] = round_interactions_df.apply(
    lambda row: stabilized_factor / row['propensity_score'] if row['type'] == 'Original'
    else (1 - stabilized_factor) / (1 - row['propensity_score']),
    axis=1
)
```
stabilized_factor = np.mean(y_train) corresponds to 
$P(A=1)$
1 - stabilized_factor corresponds to 
$P(A=0)$
1 / IPTW
``` python
model.fit(X_train, y_train, weights=1 / round_interactions_df['IPTW'].values) 
```
***the code first computes stabilized propensity-based weights and then uses their inverse in the GAM fitting step for Ignoring outliers***
  
**propensity-weighted GAM**
GAM (Generalized Additive Model), which can flexibly fit nonlinear relationships
Propensity-weighted GAM calibration means first estimating the probability that an interaction pair is an original pair rather than a knockoff pair given its attribution-related covariates, then fitting a weighted generalized additive model (GAM) to explain the systematic component of the interaction score, and finally using the absolute residual as the calibrated interaction score.

VS Linear Regression$Y = \beta_0+\beta_1X_1+\beta_2X_2+\cdot\cdot\cdot+\beta_pX_p+\epsilon$
in GAM, $Y =  \beta_0+f_1(X_1)+f_2(X_2)+\cdot\cdot\cdot+f_p(X_p)+\epsilon$
no penalties

### model Architectures
##### **DeepPINK** (MLP CNN FFTtransformer)
DeepPINK: reproducible feature selection in deep **neural networks**.
Advances in Neural Information Processing Systems (NeurIPS) 8676-8686, 2018.
DeepPINK achieves distillation of non-additive effects through its unique Z-weight mechanism on MLP 1DCNN FFTtransformer (**backbone**)
``` python
self.Z_weight = nn.Parameter(torch.ones(2 * p))
```
``` python
def __init__(
        self, 
        p,  
        model_type, #可选mlp cnn transformer
        use_Z_weight=True,  #是否使用可学习的 Z 权重（大小为 2p）。若为 True，则创建一个 nn.Parameter 初始化为全1；若为 False，则后续不使用 Z 权重
        normalize_Z_weight=False, #是否对 Z 权重进行归一化（将前 p 个和后 p 个绝对值归一化到和为1）
        *args,
        **kwargs
    ):
```
**MLP(multilayer perceptron)**
Z_weight 
``` python
def _fetch_Z_weight(self):
    Z = self.Z_weight
    if self.normalize_Z_weight:
        normalizer = torch.abs(self.Z_weight[:self.p]) + \
                    torch.abs(self.Z_weight[self.p:])
        Z = torch.cat([torch.abs(self.Z_weight[:self.p]) / normalizer, 
            torch.abs(self.Z_weight[self.p:]) / normalizer], dim=0)
    return Z # Knockoff weight: |Z[i+p]| / (|Z[i]| + |Z[i+p]|)Original weight: |Z[i]| / (|Z[i]| + |Z[i+p]|)
```
model-specific interaction importance(**only MLP**)
``` python
 def _get_W(self): # 遍历 MLP 的所有线性层，将权重矩阵相乘（从输入层到输出层），得到近似的全局权重矩阵 W（形状为（输入维度, 1））
            if self.model_type == "mlp":
                with torch.no_grad():
                    # Calculate weights from MLP
                    layers = list(self.mlp.named_children())
                    W = None
                    for layer in layers:
                        if isinstance(layer[1], torch.nn.Linear):
                            weight = layer[1].weight.cpu().detach().numpy().T
                            W = weight if W is None else np.dot(W, weight)
                    W = W.squeeze(-1)
                    return W
```
Forward Pass with MLP
``` python
# src/models/DeepPink.py:323-324
if self.model_type == "mlp":
    X = self.mlp(X)
```
**CNN1D**
``` python
def compute_fc_input_dim(self, input_dim):
    # 生成模拟输入：1个样本，1个通道，长度为input_dim的序列
    x = torch.randn(1, 1, input_dim)
    # 模拟前向传播的卷积/池化过程
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.pool(x)
    # 展平：把 (1, 64, 序列长度) 变成 (1, 64×序列长度)
    x = x.view(x.size(0), -1)
    # 返回展平后的特征数（比如input_dim=50时，池化后长度25，64×25=1600）
    return x.size(1)
```
Forward Pass with CNN
``` python
# src/models/DeepPink.py:325-327
elif self.model_type == "cnn":
    X = X.unsqueeze(1)  # Add channel dimension: (batch, p) -> (batch, 1, p)
    X = self.cnn(X)
```

**explanation**
- global_feature_importances()
  ``` python
       def global_feature_importances(self):
            
            if self.model_type == "mlp":# 计算全局特征重要性
                with torch.no_grad():
                    # Calculate weights from MLP
                    W = self._get_W()
                    if self.use_Z_weight:
                        # Multiply by Z weights
                        Z = self._fetch_Z_weight().cpu().numpy()
                        feature_imp = Z[:self.p] * W
                        knockoff_imp = Z[self.p:] * W
                        return np.concatenate([feature_imp, knockoff_imp])
                    else:
                        return W #如果不使用z权重，则直接返回W
            else:
                raise NotImplementedError("Feature importances are only implemented for MLP models.")

                # 这个重要性综合了网络权重和 Z 权重，可以用于评估每个原始特征及其 Knockoff 副本对输出的贡献。通常，原始特征的重要性减去其 Knockoff 的重要性可以用于控制 FDR。
  ```
- global_feature_interactions()
    $I_i = z_i\cdot \tilde{W}_i$
  ``` python
  def global_feature_interactions(self):
    if self.model_type == "mlp":
        with torch.no_grad():
            weights = self.get_weights()
            w_input = weights[0]
            w_later = weights[-1]
            for i in range(len(weights)-2, 0, -1):
                w_later = np.matmul(w_later, weights[i])
            if self.use_Z_weight: Z = self._fetch_Z_weight().cpu().numpy()
            else: Z = np.ones(self.p*2)
            attributions = np.zeros((1, self.p*2))
            interactions = np.zeros((1, self.p*2, self.p*2))
            def inter_func(i, j):
                w_input_i = Z[i]*w_input[:, i%w_input.shape[1]]
                w_input_j = Z[j]*w_input[:, j%w_input.shape[1]]
                attributions[0, i] = np.abs((w_input_i*w_later).sum())
                attributions[0, j] = np.abs((w_input_j*w_later).sum())
                interactions[0, i, j] =  np.abs((np.multiply(w_input_i, w_input_j)*w_later).sum())

            for i, j in itertools.product(np.arange(self.p*2), repeat=2):
                inter_func(i, j)

            return attributions, interactions
    else:
        raise NotImplementedError(...)
    ```
- global_3rd_order_interactions()

DeepPINK output
```python
def global_feature_interactions(self):
    if self.model_type == "mlp":
        with torch.no_grad():
            weights = self.get_weights()
            w_input = weights[0]
            w_later = weights[-1]
            for i in range(len(weights)-2, 0, -1):
                w_later = np.matmul(w_later, weights[i])
            if self.use_Z_weight: Z = self._fetch_Z_weight().cpu().numpy()
            else: Z = np.ones(self.p*2)
            attributions = np.zeros((1, self.p*2))
            interactions = np.zeros((1, self.p*2, self.p*2))
            def inter_func(i, j):
                w_input_i = Z[i]*w_input[:, i%w_input.shape[1]]
                w_input_j = Z[j]*w_input[:, j%w_input.shape[1]]
                attributions[0, i] = np.abs((w_input_i*w_later).sum()) # 取第 i 个输入特征对应的输入层权重，再乘上 Z 权重。
                attributions[0, j] = np.abs((w_input_j*w_later).sum())
                interactions[0, i, j] =  np.abs((np.multiply(w_input_i, w_input_j)*w_later).sum())

            for i, j in itertools.product(np.arange(self.p*2), repeat=2):
                inter_func(i, j)

            return attributions, interactions
```
$interaction(i,j) = \left|\sum(Z[i] \cdot w\_input[:,i])\odot (Z[j] \cdot w\_input[:,j]) \right|$
**FTTransformer (Feature Tokenizer Transformer)**
backbone is designed for tabular data and uses attention mechanisms to capture complex feature interactions. It employs numerical embeddings and multi-head self-attention.

##### Kolmogorov-Arnold Network (KAN)
KAN represents functions using the Kolmogorov-Arnold representation theorem, replacing traditional linear transformations with learnable univariate activation functions. The architecture differs fundamentally from standard MLPs:
$$ f(x_1, \dots, x_n) = \sum_{q=1}^{2n+1} \Phi_q \left( \sum_{p=1}^n \phi_{q,p}(x_p) \right) $$
```
Traditional MLP:    h = σ(W·x + b)
KAN:               h = Σ φ_i(x_i)
```
``` python 
model = KAN(width=[2 * num_features, num_features // 2, 1], 
           device=device, #original p+original p = 2p
           seed=seed, #隐藏层是num_features // 2 -> 降维
           auto_save=False)
```
Architecture
- Input layer: 2p features (original + knockoff)
- Hidden layer: p/2 units
- Output layer: 1 unit
  
loss
$$ MSE = \frac1N \sum^{N}_{i=1}(y_i-\hat{y_i})^2 $$
$$ CoxPHLoss = - \sum_{i:\delta=1} \left(\hat{\eta}_i-log\sum_{j \in R(t_i)}e^{\hat{\eta}j} \right)$$
****Explanation** via PathExplainerTorch**
KAN models use the same gradient-based explanation method as DeepPINK neural networks.
KAN supports both Integrated Gradients (IG) and Expected Gradients (EG) but not topological explanation (which is MLP-specific).
PathExplainerTorch support IG, EG, pairwise interactions.
The approach involves calculating the gradient integral along the path from the input to the baseline to obtain feature attribution.
``` python
test_dataset = torch.tensor(X, dtype=torch.float32, device=device)
test_dataset.requires_grad = True
 
# Choose baseline
if explainer == 'ig':
    baseline = torch.zeros((1, num_features * 2), device=device)
    use_expectation = False
elif explainer == 'eg':
    baseline = torch.tensor(X_train, dtype=torch.float32, device=device)
    use_expectation = True
 
explainer = PathExplainerTorch(model)
attributions = explainer.attributions(test_dataset, baseline=baseline, 
                                     use_expectation=use_expectation)
interactions = explainer.interactions(test_dataset, baseline=baseline,
                                      use_expectation=use_expectation)
```
##### tree (XGBoost/LightGBM)
**XGBoost**
Gradient boosted decision trees（GBDT）
**Regression VS Classification**

||Regression|Classification|
|---|---|---|
|predition target|Consecutive values|Discrete values|
|loss function|MSE MAE|Cross-Entropy Hinge Loss|
|Output format|$\R$|probability/logits / class label|

Regression Configuration
``` python
model = xgb.XGBRegressor(
    n_estimators=100,
    eval_metric='rmse',
    device=device,
    random_state=seed
)
```
Classification Configuration
``` python
model = xgb.XGBClassifier(
    n_estimators=100,
    scale_pos_weight=pos_weight,
    device=device,
    random_state=seed
)
```

**LightGBM**
Traditional trees grow layer by layer (level-wise). LightGBM splits the leaf with the largest gain first (leaf-wise), which is faster and more accurate.
Regression Configuration
``` python
model = lgb.LGBMRegressor(
    n_estimators=100,
    device='cpu',
    random_state=seed
)
```
Classification Configuration
``` python
model = lgb.LGBMClassifier(
    n_estimators=100,
    scale_pos_weight=pos_weight,
    device='cpu',
    random_state=seed
)
```

****Explanation** via SHAP TreeExplainer**
Explanation method of LightGBM and XGoost is SHAP TreeExplainer.SHAP's full name is SHapley Additive exPlanations.TreeExplainer increases speed using dynamic programming. TreeExplainer computes SHAP value exactly.
SHAP has 3 properties, Additivity, Consistency and Local accuracy Consistency
``` python
explainer = shap.TreeExplainer(model)
attributions = explainer.shap_values(X=X)
interactions = explainer.shap_interaction_values(X=X)
```
**shap_values**
SHAP breaks down the model's prediction for a given sample into the "contribution" of each feature.
$ n = base\ value+\phi_1+\phi_2+\phi_3+\cdot\cdot\cdot+\phi_p$
- base value: the model’s average output on the background data
- $\phi_j$:  the contribution of the \(j\)th feature to the prediction for this sample
$$
\phi_j = \sum_{S \subseteq F \setminus \{j\}}
\frac{|S|!(p-|S|-1)!}{p!}
\big[f(S \cup \{j\}) - f(S)\big]
$$
$F$: The set of all features
$f(S)$:The model’s expected output given only the subset \(S\)

**shap_interaction_values**
$$
\phi_{i,j} = \sum_{S \subseteq N \setminus \{i,j\}} \frac{|S|! \, (M - |S| - 2)!}{2(M-1)!} \left[ f(S \cup \{i,j\}) - f(S \cup \{i\}) - f(S \cup \{j\}) + f(S) \right]
$$
**The SHAP value of a feature = the sum of its interaction values with all other features**
##### FM Factorization Machines
prediction
\[
\hat{y}(x) = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n}\sum_{j=i+1}^{n} \langle v_i, v_j \rangle x_i x_j
\]


- \(\hat{y}(x)\): The model’s predicted value for sample \(x\)
- \(w_0\): Global bias term
- \(w_i\): Linear weight for the \(i\)-th feature
- \(x_i\): Input value for the \(i\)-th feature
- \(v_i \in \mathbb{R}^k\): The latent vector corresponding to the \(i\)th feature
- \(\langle v_i, v_j \rangle\): The vector inner product representing the interaction strength between features \(i\) and \(j\)
- $\sum_{i=1}^{n} w_i x_i$ represents the independent linear contribution of each feature.
- $\sum_{i=1}^{n}\sum_{j=i+1}^{n} \langle v_i, v_j \rangle x_i x_j$ represents the interaction between any two features. we represent the interaction through the inner product of low-dimensional latent vectors \(v_i\) and \(v_j\), instead of  learn a separate parameter for each feature pair \((i, j)\). This significantly reduces the number of parameters and improves generalization performance.
LibSVM

**explanation**
``` python
ws = []  # 每个特征对应的Linear weights 
vs = []  # Latent factor vectors
with open(f"{output_dir}/{dataset}_seed{seed}.txt") as f:
    for line in f:
        if line.startswith('i'):
            ws.append(float(line.split()[1]))
        elif line.startswith('v'):
            vs.append(np.array([float(x) for x in line.split()[1:]]))
```
attributions
``` python
attributions = np.zeros((1, num_features * 2))
for i in range(num_features * 2):
    attributions[0, i] = ws[i]
```
interactions
``` python
interactions = np.zeros((1, num_features * 2, num_features * 2))
for i in range(num_features * 2):
    interactions[0, i, i] = np.dot(vs[i], vs[i])
for i, j in itertools.combinations(range(num_features * 2), 2):
    interactions[0, i, j] = np.dot(vs[i], vs[j])#点积绝对值越大，交互越强。正值为正交互
    interactions[0, j, i] = interactions[0, i, j]
```
## Interaction selection / FDR control
Build knockoff statistics from calibrated interaction scores and select interactions using a data-dependent threshold that controls FDR.

#### Knockoff statistics and thresholding
With the constructed knockoff,  feature importances are quantified by computing the knockoff statistics$W_j = g_j(Z_j,\tilde{Z}_j),1 \leq j \leq p$

$g_j()$ is an antisymmetric function, $g_j(Z_j,\tilde{Z}_j) = -g_j(\tilde{Z}_j,Z_j)$.If knockoff more important than original,$W_j < 0$,otherwise $W_j > 0$.
**coin-flip** property means If any pair $X_j$ and $\tilde{X}_j$ is swapped, then it will only change the sign of the corresponding $W_j$; the signs of other $W_k$ (k ≠ j) remain unchanged.

A desirable property for knockoff statistics $W_j$ is that important features are expected to have large absolute values, whereas unimportant features should have small symmetric values around 0.
Sort $\left| W_j \right|$ from largest to smallest, and features with values exceeding a certain threshold T will be selected.

$T=\min\{t\in {\mathcal{W}},\frac{1+| \left\{\;j:{W}_{j}\le -t\right\}| }{| \left\{j:{W}_{j}\ge t\right\}| }\le q\}, {\mathcal{W}}=\left\{| {W}_{j}| :1\le j\le p\right\}\backslash \left\{0\right\}$

$\mathcal{W}$ is all non-zero ($\left| W_j\right|$) collection
q is set manually
- $W_j=\left\|score_j\right\|-\left\|score_{j+p}\right\|$ (individual) 
- $ T = min{FDP(t) \leqslant q }$


#### Selection
model_utils
``` python
max_score = -np.inf#初始化
score = np.abs((TD_count-DD_count)) / max(1.0, TT_count) #防止分母为0 用最大score更新
```
T = True | D = knockoff


#### Evaluation
- False Discovery Proportion (FDP)
FDP measures the proportion of selected interactions that are false positives.
$$ \text{FDR} = \mathbb{E}[\text{FDP}] \quad \text{and} \quad \text{FDP} = \frac{|\hat{S} \cap S^c|}{|\hat{S}|}$$
``` python
def get_interaction_fdp(sel_interactions, ground_truth):
    binaries, _ = get_gt_bins(sel_interactions, ground_truth)
    fdp = (len(binaries) - sum(binaries)) / max(1, len(binaries))
    return fdp
```

#### Statistical Power
Power measures the proportion of true interactions that are successfully discovered.
Power = TP / total_ground_truth_interactions
``` python
  # From model_utils.py:526-536
def get_interaction_power(sel_interactions, ground_truth, order=2):
    binaries, _ = get_gt_bins(sel_interactions, ground_truth)
    num_gt = 0
    all_gt = []
    for gt in ground_truth:
        curr_gt = list(combinations(gt, order))
        all_gt += curr_gt
    all_gt = set(all_gt)
    num_gt = len(all_gt)
    power = sum(binaries) / max(1, num_gt)#Power = (true positives) / max(1, total ground truth pairs)
    return power
```
#### Area Under ROC Curve (AUC)
``` python
# From model_utils.py:183-194
def pairwise_interaction_auc(interactions, ground_truth):
    strengths = []
    gt_binary_list = []
    for inter, strength in interactions:
        inter_set = set(inter)
        strengths.append(strength)
        if any(inter_set <= gt for gt in ground_truth):
            gt_binary_list.append(1)
        else:
            gt_binary_list.append(0)
    auc = roc_auc_score(gt_binary_list, strengths)
    return auc
```
Evaluation against ground truth
feat_import_power
feat_import_fdp
pairwise_interaction_auc
get_interaction_fdp
get_interaction_power
FDR = 0.2, Process 20 rounds with calibration(Fig3e.g.), Report 95% confidence interval
#### visualize_results
- stripplots: Typically used to display the distribution of W_j values for each feature and highlight features that exceed the threshold.
- ECDFs (Empirical Cumulative Distribution Functions): These can display the cumulative distribution of $W_j$ values or show the curve of the false positive rate (FPR) as the threshold changes, helping to understand how the false positive rate is controlled.
- Q-Value Plots