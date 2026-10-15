# Diamond 特征交互发现 · 网页版 (diamond-webapp)

把 Diamond 论文的核心方法（**model-X knockoffs + SHAP + FDR 控制筛选**）做成一个
**「上传 CSV 即可分析」**的网页应用。纯 CPU、无需 GPU、无需原论文的复杂环境。

在线地址（已发布）：**https://diamond-fdr.app.workbuddy.host/**

## 它能做什么

1. 上传一份 CSV（一行一个样本，一列一个特征，最后一列是目标变量 `target`）；
2. 在网页上指定：目标变量 / 特征列 / 标识列；
3. 选择模型（**LightGBM** 或 **XGBoost**）与目标 FDR（0.05–0.5）；
4. 一键分析，得到：
   - 特征重要性柱状图（按 mean|SHAP| 排序）；
   - 特征两两交互热力图；
   - **FDR 控制下的非加性交互列表**（knockoff+ / data-dependent threshold）；
   - 完整特征排序表（含「候选集」标记：原始效应 > 其 knockoff 副本）；
   - 可下载的结果 CSV。

## 算法流程

```
CSV ──▶ knockoffsdiag 生成统计等价的 knockoff 副本 X̃
     ──▶ 训练树模型 (LightGBM / XGBoost)
     ──▶ SHAP TreeExplainer 计算 attributions 与 interaction_values
     ──▶ 特征筛选: W = attr(原始) − attr(knockoff)，knockoff+ 阈值控制 FDR
     ──▶ 交互筛选: 展平 (2p×2p) 交互矩阵 → 配对排序 → FDR 阈值
```

核心算法从原 Diamond 仓库抽取并**去掉了 torch 依赖**，保留 knockoff 生成 +
SHAP + FDR 筛选三条主线，走轻量 CPU 路径。

- `diamond_core.py` — 分析内核（纯 numpy / sklearn / lightgbm / xgboost / shap）。
- `app.py` — FastAPI 单端口服务（`/api/upload` `/api/process`（后台任务+轮询）
  `/api/job/{id}` `/api/download/{token}` `/api/sample/{name}` `/api/health`）。
- `static/index.html` — 前端页面。
- `samples/` — 两个可直接上传的真实数据集（enhancer / mortality），由原 Diamond
  的专有格式（`.Rdata` / 定宽 `.txt` 磁带文件）按 `prep_data.py` 逻辑转换而来。
- `static/assets/paper.pdf` — 论文 PDF（网页内嵌阅读）。
- `make_sample_csvs.py` — 把原 Diamond 数据转为可上传 CSV 的脚本。

## 本地运行

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
# 浏览器打开 http://127.0.0.1:8000
```

> 注：Windows 上若 `import lightgbm` 报找不到 `VCOMP140.DLL`（OpenMP 运行时），
> 需先把 Visual C++ 运行时的 `vcomp140.dll` 放到
> `site-packages/lightgbm/bin/`；Linux/macOS 的官方 wheel 已自带依赖，无需处理。

## 与原论文的关系

- 论文：Chen, Jiang, Noble, Lu. *Error-controlled non-additive interaction
  discovery in machine learning models*. Nature Machine Intelligence (2025).
  doi:10.1038/s42256-025-01086-8
- 本应用是论文方法的**轻量复现/可用化**，并非原仓库（原仓库依赖 GPU 版
  PyTorch/TensorFlow/xlearn/pykan）。算法内核保持一致：model-X knockoffs +
  SHAP 归因 + FDR 控制的特征/交互发现。

## 许可

分析内核 `diamond_core.py` 源自原 Diamond 项目，遵循其 **Apache-2.0 License**。
其余网页代码按本仓库约定开放。
