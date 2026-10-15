"""
diamond_core.py — Diamond 的「轻量 CPU」分析内核（网页版后端使用）

来源与许可：
    函数实现抽取自 Diamond 项目（Chen W, Jiang Y, Noble WS, Lu YY.
    *Error-controlled non-additive interaction discovery in machine learning
    models.* Nat Mach Intell, 2025），Apache 2.0 License。
    抽取时**移除了对 PyTorch / TensorFlow / xlearn / pykan 的依赖**，只保留
    纯 numpy / scikit-learn / scipy / statsmodels 的实现，以便在 CPU 环境
    （如 WorkBuddy 发布沙箱）运行。核心算法（model-X knockoffs + FDR 控制）
    与原版一致。

流程：
    1. 用 knockoffsdiag（Lasso 条件序列生成）为 X 生成 knockoff 副本；
    2. 用 LightGBM / XGBoost 在 [X, X_knockoff] 上训练，预测 Y；
    3. 用 SHAP（TreeExplainer）计算特征归因与两两交互；
    4. 用 knockoff 统计量做 FDR 控制，筛选显著特征与显著交互。
"""

import itertools
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from scipy.linalg import toeplitz
from scipy.stats import hmean
from sklearn.linear_model import Lasso
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state, shuffle
from sklearn.utils.validation import check_memory
from statsmodels.distributions.empirical_distribution import (
    ECDF, monotone_fn_inverter)
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1) Knockoff 生成（来自 src/knockoffs/gen_Knockoffsdiag.py，已去掉 torch 依赖）
# ---------------------------------------------------------------------------

def _get_single_clf_ko(X, j, method="lasso"):
    """用除第 j 列外的所有变量预测 X[:, j]，返回预测值。"""
    n, p = X.shape
    idc = np.array([i for i in np.arange(0, p) if i != j])
    if method == "lasso":
        lambda_max = np.max(np.abs(np.dot(X[:, idc].T, X[:, j]))) / (2 * (p - 1))
        alpha = (lambda_max / 100)
        clf = Lasso(alpha)
    elif method == "logreg_cv":
        from sklearn.linear_model import LogisticRegressionCV
        clf = LogisticRegressionCV(cv=5, max_iter=int(10e4), n_jobs=-1)
    elif method == "xgb":
        import xgboost as xgb
        clf = xgb.XGBRegressor(n_jobs=-1)
    else:
        raise ValueError(f"unknown method_ko_gen: {method}")
    clf.fit(X[:, idc], X[:, j])
    pred = clf.predict(X[:, idc])
    return pred


def _adjust_marginal(v, ref, discrete=False):
    """使 v 的边际分布匹配 ref。"""
    if discrete:
        sorter = np.argsort(v)
        sorter_ = np.argsort(sorter)
        return np.sort(ref)[sorter_]
    else:
        G = ECDF(ref)
        F = ECDF(v)
        unif_ = F(v)
        G_inv = monotone_fn_inverter(G, ref)
        return G_inv(unif_)


def _get_samples_ko(X, pred, j, discrete=False, adjust_marg=True, seed=None):
    """为第 j 个变量生成 knockoff。"""
    np.random.seed(seed)
    n, p = X.shape
    residuals = X[:, j] - pred
    indices_ = np.arange(residuals.shape[0])
    np.random.shuffle(indices_)
    sample = pred + residuals[indices_]
    if adjust_marg:
        sample = _adjust_marginal(sample, X[:, j], discrete=discrete)
    return sample[np.newaxis].T


def conditional_sequential_gen_ko(X, discrete, n_jobs=1, method_ko_gen='lasso',
                                  adjust_marg=False, seed=None):
    """为 X 的所有变量生成 knockoff。"""
    rng = check_random_state(seed)
    n, p = X.shape
    preds = np.array(Parallel(n_jobs=n_jobs)(delayed(
        _get_single_clf_ko)(X, j, method_ko_gen) for j in tqdm(range(p))))
    samples = np.hstack(Parallel(n_jobs=n_jobs)(delayed(
        _get_samples_ko)(X, preds[j], j, discrete=discrete, adjust_marg=adjust_marg)
        for j, discrete in tqdm(enumerate(discrete))))
    return samples


# ---------------------------------------------------------------------------
# 2) FDR 控制筛选（来自 src/models/model_utils.py，纯 numpy 实现）
# ---------------------------------------------------------------------------

def get_interaction_type_idx(inter_indices_arr, p):
    TD_idx, TT_idx, DD_idx = [], [], []
    for idx, indices in enumerate(inter_indices_arr):
        if all(i < p for i in indices):
            TT_idx.append(idx)
        elif all(i >= p for i in indices):
            DD_idx.append(idx)
        else:
            TD_idx.append(idx)
    return TT_idx, TD_idx, DD_idx


def get_selected_interactions(interactions, p, q):
    interactions = interactions.copy()
    interactions.sort(key=lambda x: x[1], reverse=True)
    inter_idices_arr = np.array([i for i, _ in interactions])
    inter_score_arr = np.array([s for _, s in interactions])
    TT_idx, TD_idx, DD_idx = get_interaction_type_idx(inter_idices_arr, p)
    cutoff = np.inf
    max_score = -np.inf
    for idx, t in interactions:
        TT_count = (inter_score_arr[TT_idx] >= t).sum()
        TD_count = (inter_score_arr[TD_idx] >= t).sum()
        DD_count = (inter_score_arr[DD_idx] >= t).sum()
        score = np.abs((TD_count - DD_count)) / max(1.0, TT_count)
        score = np.clip(score, 0, 1)
        if score > max_score:
            max_score = score
        else:
            score = max_score
        if (t < cutoff) and (score <= q):
            cutoff = t
        if score > q:
            break
    selected_interaction, selected_knockoffs = [], []
    for idx, score in interactions:
        i, j = idx
        if score >= cutoff:
            if (i < p) and (j < p):
                selected_interaction.append((idx, score))
            else:
                selected_knockoffs.append((idx, score))
    if len(selected_interaction) == 1 and len(selected_knockoffs) == 0:
        selected_interaction = []
    max_score = np.max(inter_score_arr)
    if cutoff == np.inf or len(selected_interaction) == 0:
        cutoff = max_score + 1e-6
    return selected_interaction, cutoff


def get_q_values(interactions, p):
    interactions = interactions.copy()
    interactions.sort(key=lambda x: x[1], reverse=True)
    inter_idices_arr = np.array([i for i, _ in interactions])
    inter_score_arr = np.array([s for _, s in interactions])
    TT_idx, TD_idx, DD_idx = get_interaction_type_idx(inter_idices_arr, p)
    q_values = {}
    max_score = -np.inf
    for idx, t in interactions:
        TT_count = (inter_score_arr[TT_idx] >= t).sum()
        TD_count = (inter_score_arr[TD_idx] >= t).sum()
        DD_count = (inter_score_arr[DD_idx] >= t).sum()
        score = np.abs((TD_count - DD_count)) / max(1.0, TT_count)
        score = np.clip(score, 0, 1)
        if score > max_score:
            max_score = score
        else:
            score = max_score
        q_values[idx] = score
    return q_values


def get_interaction_ranking_from_matrix(matrix):
    """把 (2p x 2p) 交互矩阵展平成所有配对（含 original-original / knockoff-knockoff
    / original-knockoff），按 |score| 降序返回。"""
    assert len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]
    p = matrix.shape[0] // 2
    interaction_ranking = []
    def inter_func(i, j):
        score = np.abs(matrix[int(i), int(j)])
        interaction_ranking.append(((i, j), score))
    for comb in itertools.combinations(np.arange(p), 2):
        inter_func(*comb)
    for comb in itertools.combinations(np.arange(p, 2 * p), 2):
        inter_func(*comb)
    for i in np.arange(p):
        for j in np.arange(p, 2 * p):
            if (i != j - p):
                inter_func(i, j)
    for i in np.arange(p, 2 * p):
        for j in np.arange(p):
            if (i != j + p):
                inter_func(i, j)
    interaction_ranking.sort(key=lambda x: x[1], reverse=True)
    return interaction_ranking


def data_dependent_threshold(W, fdr=0.10, offset=1):
    """model-X knockoffs 的特征筛选阈值（基于 W 统计量）。"""
    if len(W.shape) == 1:
        W = W.reshape(-1, 1)
    p = W.shape[0]
    batch = W.shape[1]
    ind = np.argsort(-1 * np.abs(W), axis=0)
    sorted_W = np.take_along_axis(W, ind, axis=0)
    negatives = np.cumsum(sorted_W <= 0, axis=0)
    positives = np.cumsum(sorted_W > 0, axis=0)
    positives[positives == 0] = 1
    ratios = (negatives + offset) / positives
    sorted_W = np.concatenate([sorted_W, np.zeros((1, batch))], axis=0)
    helper = (ratios <= fdr) + np.arange(0, p, 1).reshape(-1, 1) / p
    sorted_W[1:][helper < 1] = np.inf
    T_inds = np.argmax(helper, axis=0) + 1
    more_inds = np.indices(T_inds.shape)
    acceptable = np.abs(sorted_W)[T_inds, more_inds][0]
    if np.sum(acceptable == 0) != 0:
        W_new = W.copy()
        W_new[W_new == 0] = np.abs(W).max()
        zero_replacement = np.abs(W_new).min(axis=0)
        acceptable[acceptable == 0] = zero_replacement[acceptable == 0]
    if batch == 1:
        acceptable = acceptable[0]
    return acceptable


# ---------------------------------------------------------------------------
# 3) 编排：上传数据 -> 筛选特征 / 交互
# ---------------------------------------------------------------------------

def _autodetect_task(y):
    y = np.asarray(y)
    uniq = np.unique(y[~np.isnan(y)] if y.dtype.kind in 'fc' else y)
    if y.dtype.kind in ('i', 'u', 'b') or (y.dtype.kind == 'O'):
        if len(uniq) <= 10:
            return 'classification'
    return 'regression'


def run_diamond(X, y, feature_names, model_type='lightgbm', target_fdr=0.2,
                n_jobs=1, seed=0, task=None, n_estimators=100,
                attr_max_n=3000, iv_max_n=600, progress=None):
    """
    参数
    ----
    X : np.ndarray (n, p)            特征矩阵（已数值化）
    y : np.ndarray (n,)               响应变量
    feature_names : list[str]         特征名（长度 p）
    model_type : 'lightgbm' | 'xgboost'
    target_fdr : float                目标错误发现率（默认 0.2）
    n_jobs : int                      knockoff 生成的并行数
    seed : int
    task : 'classification' | 'regression' | None(自动)
    attr_max_n : int   计算 SHAP 归因时的最大抽样样本数（训练仍用全量）
    iv_max_n : int     计算 SHAP 交互值时的最大抽样样本数（交互值计算量
                       O(n·(2p)^2·树数)，全量大样本会极慢，必须抽样）
    progress : callable(stage:str, frac:float) | None   进度回调

    返回
    ----
    dict，包含 attributions / 特征筛选结果 / 交互筛选结果 / 交互矩阵。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape

    if task is None:
        task = _autodetect_task(y)
    print(f"[diamond] task={task}, n={n}, p={p}, model={model_type}, fdr={target_fdr}")

    # 1) knockoff 生成（全部按连续变量处理）
    if progress: progress('knockoff', 0.05)
    discrete = np.array([False] * p)
    Xk = conditional_sequential_gen_ko(
        X, discrete, n_jobs=n_jobs, method_ko_gen='lasso',
        adjust_marg=False, seed=seed)
    Xc = np.concatenate((X, Xk), axis=1)  # (n, 2p)

    # 2) 训练模型（全量数据）
    if progress: progress('train', 0.30)
    if model_type == 'lightgbm':
        import lightgbm as lgb
        if task == 'classification':
            model = lgb.LGBMClassifier(n_estimators=n_estimators, random_state=seed)
        else:
            model = lgb.LGBMRegressor(n_estimators=n_estimators, random_state=seed)
    elif model_type == 'xgboost':
        import xgboost as xgb
        if task == 'classification':
            model = xgb.XGBClassifier(n_estimators=n_estimators, random_state=seed)
        else:
            model = xgb.XGBRegressor(n_estimators=n_estimators, random_state=seed)
    else:
        raise ValueError(f"unknown model_type: {model_type}")
    model.fit(Xc, y)

    # 3) SHAP 归因与交互（大样本自动抽样：交互值计算量 O(n·(2p)²·树数)）
    if progress: progress('shap', 0.50)
    import shap
    rng = np.random.default_rng(seed)

    def _sample_rows(max_n):
        if n <= max_n:
            return None
        return np.sort(rng.choice(n, max_n, replace=False))

    attr_idx = _sample_rows(attr_max_n)
    iv_idx = _sample_rows(iv_max_n)
    X_attr = Xc if attr_idx is None else Xc[attr_idx]
    X_iv = Xc if iv_idx is None else Xc[iv_idx]

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_attr)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    attributions = np.mean(np.abs(sv), axis=0)      # (2p,)

    if progress: progress('shap_inter', 0.70)
    iv = explainer.shap_interaction_values(X_iv)
    if isinstance(iv, list):
        iv = iv[1] if len(iv) > 1 else iv[0]
    inter_matrix = np.mean(np.abs(iv), axis=0)       # (2p, 2p)
    attr_n_used = int(X_attr.shape[0])
    iv_n_used = int(X_iv.shape[0])

    # 4a) 特征筛选（knockoff W 统计量）
    W = attributions[:p] - attributions[p:]          # 原始 - knockoff
    T = data_dependent_threshold(W, fdr=target_fdr, offset=1)
    import math
    if (not isinstance(T, (int, float))) or math.isinf(T) or math.isnan(T) or T < 0:
        T = None
        selected_feats = []
    else:
        selected_feats = [int(j) for j in range(p) if W[j] > T]
    # 候选集：原始特征归因 > 其 knockoff 归因（W>0），直观且不会因 FDR 退化为空
    candidate_feats = [int(j) for j in range(p) if W[j] > 0]

    feature_table = [{
        'name': feature_names[j],
        'attr': float(attributions[j]),
        'attr_knockoff': float(attributions[j + p]),
        'W': float(W[j]),
        'selected': bool(j in selected_feats),
        'candidate': bool(j in candidate_feats),
    } for j in range(p)]
    feature_table.sort(key=lambda r: r['attr'], reverse=True)

    # 4b) 交互筛选
    if progress: progress('select', 0.88)
    ranking = get_interaction_ranking_from_matrix(inter_matrix)  # 全配对
    selected_inter, cutoff = get_selected_interactions(ranking, p, target_fdr)
    qvals = get_q_values(ranking, p)

    interaction_table = []
    for (i, j), score in selected_inter:
        interaction_table.append({
            'i': int(i), 'j': int(j),
            'name_i': feature_names[i], 'name_j': feature_names[j],
            'score': float(score),
            'q': float(qvals.get((i, j), np.nan)),
        })
    # 附带 top 交互（含 q 值）用于展示
    top_interactions = []
    for (i, j), score in ranking:
        if i < p and j < p:  # 仅 original-original
            top_interactions.append({
                'i': int(i), 'j': int(j),
                'name_i': feature_names[i], 'name_j': feature_names[j],
                'score': float(score),
                'q': float(qvals.get((i, j), np.nan)),
            })
        if len(top_interactions) >= 30:
            break

    return {
        'task': task,
        'attr_n_used': attr_n_used,
        'iv_n_used': iv_n_used,
        'n': int(n), 'p': int(p),
        'attributions': attributions.tolist(),
        'inter_matrix': inter_matrix.tolist(),
        'feature_table': feature_table,
        'selected_features': [feature_names[j] for j in selected_feats],
        'candidate_features': [feature_names[j] for j in candidate_feats],
        'selected_interactions': interaction_table,
        'top_interactions': top_interactions,
        'target_fdr': float(target_fdr),
        'model_type': model_type,
        'feature_threshold': (float(T) if (T is not None and isinstance(T, (int, float))) else None),
        'interaction_cutoff': float(cutoff),
    }


# ---------------------------------------------------------------------------
# 本地自测（合成数据，含已知交互）
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    n, p = 400, 12
    X = rng.standard_normal((n, p))
    # 真实主效应：特征 0,1,2；真实交互：特征 3 × 4
    y = (2 * X[:, 0] + 1.5 * X[:, 1] - X[:, 2]
         + 1.2 * X[:, 3] * X[:, 4]
         + rng.standard_normal(n) * 0.5)
    names = [f"X{i}" for i in range(p)]
    res = run_diamond(X, y, names, model_type='lightgbm', target_fdr=0.2, n_jobs=4, seed=0)
    print("选定特征:", res['selected_features'])
    print("选定交互:")
    for it in res['selected_interactions']:
        print(f"  {it['name_i']} × {it['name_j']}  score={it['score']:.3f}  q={it['q']:.3f}")
