"""
app.py — Diamond 网页版后端（FastAPI，单端口 HTTP 服务）

端点
----
GET  /                        网页前端
GET  /api/health             健康检查
GET  /api/template           下载示例 CSV 模板
POST /api/upload             上传 CSV，返回列信息
POST /api/process            运行 Diamond 分析，返回筛选结果与图表
GET  /api/download/{token}   下载分析结果 CSV
GET  /api/stats              访问量统计（每次打开首页累加）

部署：监听 $PORT，绑定 0.0.0.0。
"""

import base64
import io
import json
import os
import threading
import uuid

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diamond_core import run_diamond

app = FastAPI()
BASE = os.path.dirname(os.path.abspath(__file__))
SESSIONS = {}          # session_id -> {'df': DataFrame}
DOWNLOADS = {}         # token -> result dict
JOBS = {}              # job_id -> {'status','stage','progress','error','result'}
VISITS_FILE = os.path.join(BASE, "visits.json")

# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _read_csv(content: bytes) -> pd.DataFrame:
    last_err = None
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(content), encoding=enc)
        except Exception as e:  # noqa
            last_err = e
    raise HTTPException(400, f"无法解析 CSV：{last_err}")


def _sanitize(o):
    """递归把 NaN/inf 转成 None，使结果可被 JSON 序列化。"""
    import math
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, (np.floating,)):
        o = float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _sanitize(o.tolist())
    if isinstance(o, float):
        if math.isnan(o) or math.isinf(o):
            return None
        return o
    return o


def _to_numeric_codes(series: pd.Series) -> np.ndarray:
    """非数值列用类别编码转成整数码（v1 简化：不 one-hot）。"""
    if np.issubdtype(series.dtype, np.number):
        return series.values.astype(float)
    return series.astype("category").cat.codes.values.astype(float)


def _bump_visits():
    try:
        n = 0
        if os.path.exists(VISITS_FILE):
            with open(VISITS_FILE, "r", encoding="utf-8") as f:
                n = int(json.load(f).get("visits", 0))
        n += 1
        with open(VISITS_FILE, "w", encoding="utf-8") as f:
            json.dump({"visits": n}, f)
        return n
    except Exception:  # noqa
        return 0


# ---------------------------------------------------------------------------
# 页面 / 健康检查
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    _bump_visits()
    return FileResponse(os.path.join(BASE, "static", "index.html"))


@app.get("/api/stats")
def stats():
    try:
        if os.path.exists(VISITS_FILE):
            with open(VISITS_FILE, "r", encoding="utf-8") as f:
                return {"visits": int(json.load(f).get("visits", 0))}
    except Exception:  # noqa
        pass
    return {"visits": 0}


@app.get("/api/template")
def template():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.standard_normal((n, 6))
    y = (2 * X[:, 0] + 1.5 * X[:, 1]
         + 1.2 * X[:, 2] * X[:, 3]
         + rng.standard_normal(n) * 0.5)
    df = pd.DataFrame({
        "id": np.arange(1, n + 1),
        "X1": X[:, 0], "X2": X[:, 1], "X3": X[:, 2],
        "X4": X[:, 3], "X5": X[:, 4], "X6": X[:, 5],
        "target": y,
    })
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return Response(
        buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=diamond_template.csv"},
    )


SAMPLES_DIR = os.path.join(BASE, "samples")
SAMPLE_FILES = {
    "enhancer": "enhancer_upload.csv",
    "mortality": "mortality_upload.csv",
}


@app.get("/api/sample/{name}")
def sample(name: str):
    if name not in SAMPLE_FILES:
        raise HTTPException(404, "未知示例数据集：" + name)
    path = os.path.join(SAMPLES_DIR, SAMPLE_FILES[name])
    if not os.path.exists(path):
        raise HTTPException(404, "示例文件缺失，请联系管理员")
    return FileResponse(path, media_type="text/csv", filename=SAMPLE_FILES[name])


# ---------------------------------------------------------------------------
# 上传 / 分析
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    df = _read_csv(content)
    if df.shape[1] < 2:
        raise HTTPException(400, "CSV 至少需要 2 列（特征 + 目标）")
    sid = uuid.uuid4().hex
    SESSIONS[sid] = {"df": df}
    cols = list(df.columns)
    dtypes = {c: str(df[c].dtype) for c in cols}
    samples = {c: [str(v) for v in df[c].dropna().head(3).tolist()] for c in cols}
    # 默认目标列：最后一列
    return {
        "session_id": sid,
        "row_count": int(df.shape[0]),
        "columns": cols,
        "dtypes": dtypes,
        "samples": samples,
        "suggest_target": cols[-1],
    }


@app.post("/api/process")
async def process(payload: dict):
    """启动后台分析任务，立即返回 job_id（避免网关超时 504）。

    前端用 GET /api/job/{job_id} 轮询进度，完成后取结果。
    """
    sid = payload.get("session_id")
    sess = SESSIONS.get(sid)
    if not sess:
        raise HTTPException(404, "会话不存在或已过期，请重新上传文件")
    df = sess["df"]
    target = payload.get("target")
    if not target or target not in df.columns:
        raise HTTPException(400, "请指定有效的目标列（target）")
    id_cols = set(payload.get("id_cols") or [])
    feature_cols = [c for c in df.columns if c != target and c not in id_cols]
    if not feature_cols:
        raise HTTPException(400, "至少需要 1 个特征列")

    # 数值化
    y = df[target]
    if not np.issubdtype(y.dtype, np.number):
        from sklearn.preprocessing import LabelEncoder
        y = LabelEncoder().fit_transform(y.astype(str))
    y = np.asarray(y, dtype=float).ravel()

    X_cols = []
    for c in feature_cols:
        X_cols.append(_to_numeric_codes(df[c]))
    X = np.column_stack(X_cols)
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise HTTPException(400, "数据含非数值/空值，请先清洗（v1 暂不支持缺失值）")

    task = payload.get("task") or None
    if task == "auto":
        task = None

    params = dict(
        model_type=payload.get("model", "lightgbm"),
        target_fdr=float(payload.get("target_fdr", 0.2)),
        n_jobs=int(payload.get("n_jobs", 2)),
        seed=int(payload.get("seed", 0)),
        task=task,
    )

    # 清理过期任务（只保留最近 20 个）
    if len(JOBS) > 20:
        for k in sorted(JOBS.keys())[:-20]:
            JOBS.pop(k, None)

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "running", "stage": "knockoff", "progress": 0.0,
                    "error": None, "result": None}
    threading.Thread(target=_run_job, args=(job_id, X, y, feature_cols, params),
                     daemon=True).start()
    return {"job_id": job_id, "status": "running"}


def _run_job(job_id, X, y, feature_cols, params):
    def cb(stage, frac):
        job = JOBS.get(job_id)
        if job is not None:
            job["stage"] = stage
            job["progress"] = float(frac)

    try:
        res = run_diamond(X, y, feature_cols, progress=cb, **params)
        cb("plots", 0.95)
        feat_b64, inter_b64 = _make_plots(res, feature_cols)
        res["plots"] = {"feature": feat_b64, "interaction": inter_b64}

        res["stats"] = {
            "n": res["n"], "p": res["p"], "task": res["task"],
            "selected_feature_count": len(res["selected_features"]),
            "candidate_feature_count": len(res.get("candidate_features", [])),
            "selected_interaction_count": len(res["selected_interactions"]),
        }

        token = uuid.uuid4().hex
        DOWNLOADS[token] = res
        res = _sanitize(res)
        res["token"] = token
        job = JOBS.get(job_id)
        if job is not None:
            job.update(status="done", stage="done", progress=1.0, result=res)
    except Exception as e:  # noqa
        job = JOBS.get(job_id)
        if job is not None:
            job.update(status="error", error=f"分析失败：{e}")


@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在或已过期（服务可能已重启），请重新上传并分析")
    if job["status"] == "done":
        return {"status": "done", "result": job["result"]}
    if job["status"] == "error":
        return {"status": "error", "error": job["error"]}
    return {"status": "running", "stage": job["stage"], "progress": job["progress"]}


def _make_plots(res, feature_names):
    p = res["p"]
    attr = np.array(res["attributions"][:p], dtype=float)
    order = np.argsort(-attr)
    top = order[: min(20, p)]

    # 特征重要性柱状图
    fig, ax = plt.subplots(figsize=(6, max(3, len(top) * 0.38)))
    ax.barh([feature_names[i] for i in top][::-1], attr[top][::-1], color="#2563eb")
    ax.set_title("特征重要性 (mean |SHAP|)", fontsize=12)
    ax.set_xlabel("mean |SHAP|")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    feat_b64 = base64.b64encode(buf.getvalue()).decode()

    # 交互热力图（original-original, p×p）
    M = np.array(res["inter_matrix"], dtype=float)[:p, :p]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(M, cmap="viridis")
    ax.set_xticks(range(p))
    ax.set_yticks(range(p))
    ax.set_xticklabels(feature_names, rotation=90, fontsize=7)
    ax.set_yticklabels(feature_names, fontsize=7)
    ax.set_title("特征两两交互 (|SHAP interaction|)", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    inter_b64 = base64.b64encode(buf.getvalue()).decode()
    return feat_b64, inter_b64


@app.get("/api/download/{token}")
def download(token: str):
    res = DOWNLOADS.get(token)
    if not res:
        raise HTTPException(404, "下载链接不存在或已过期")
    feat_df = pd.DataFrame(res["feature_table"])[
        ["name", "attr", "attr_knockoff", "W", "selected", "candidate"]]
    inter_df = pd.DataFrame(res["selected_interactions"])[
        ["name_i", "name_j", "score", "q"]]
    out = io.StringIO()
    out.write("=== 选定特征 ===\n")
    feat_df.to_csv(out, index=False)
    out.write("\n=== 选定交互（FDR 控制）===\n")
    inter_df.to_csv(out, index=False)
    return Response(
        out.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=diamond_results.csv"},
    )


# ---------------------------------------------------------------------------
# 静态资源（论文 PDF、示例等）：/static/*  ->  static/ 目录
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=os.path.join(BASE, "static")), name="static")


# ---------------------------------------------------------------------------
# 启动（供 deploy 调用：uvicorn app:app --host 0.0.0.0 --port $PORT）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
