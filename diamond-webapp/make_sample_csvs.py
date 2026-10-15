"""
把原 Diamond 的两种专有数据格式转成网页版 (diamond-fdr) 可上传的 CSV。

原代码 src/real.py --dataset 支持:
  - enhancer   : enhancer.Rdata (R 二进制, 经 pyreadr 读 X/Y/varnames.all)
  - mortality  : 4 个定宽 .txt 磁带文件 (按字符列位置解析)
  - diabetes / cal_housing : sklearn 内置, 无文件

本脚本只处理你 data/ 里实际存在的两类: enhancer 与 mortality。
输出: datasets_csv/enhancer_upload.csv, datasets_csv/mortality_upload.csv
格式: 一行一个样本, 一列一个特征, 最后一列 target 为目标变量。
"""
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MaxAbsScaler

ROOT = r"C:\Users\Administrator\WPSDrive\1727206706\WPS企业云盘\北京师范大学\我的企业文档\Science Forever\diamond-main\data\data"
OUT = os.path.join(os.path.dirname(__file__), "datasets_csv")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# 1) enhancer : 读 RData -> 按 Predictor_collapsed 聚合(group=mean) -> MaxAbs -> CSV
# ----------------------------------------------------------------------------
def make_enhancer_csv():
    import rdata
    path = os.path.join(ROOT, "enhancer", "enhancer.Rdata")
    obj = rdata.read_rda(path)
    X = np.asarray(obj["X"].values, dtype=float)   # (n_samples, 80)
    Y = np.asarray(obj["Y"], dtype=float).ravel()
    varnames_df = obj["varnames.all"]

    collapsed = varnames_df["Predictor_collapsed"].unique()
    pos = np.arange(len(varnames_df))  # 0-based 位置，对应 X 的列
    grouped = {}
    for feat in collapsed:
        idx = pos[varnames_df["Predictor_collapsed"].values == feat]
        grouped[feat] = X[:, idx].mean(axis=1)
    gdf = pd.DataFrame(grouped)
    Xg = gdf.values
    feat_names = gdf.columns.values

    # 与原代码一致: MaxAbs 缩放
    Xg = MaxAbsScaler().fit_transform(Xg)

    out = pd.DataFrame(Xg, columns=[str(c) for c in feat_names])
    out["target"] = Y
    op = os.path.join(OUT, "enhancer_upload.csv")
    out.to_csv(op, index=False)
    print(f"[enhancer] X={Xg.shape} -> {op}  (target=Y, 分类任务)")
    return op


# ----------------------------------------------------------------------------
# 2) mortality : 复刻 prep_data.load_mortality_data (纯 pandas/numpy/sklearn)
# ----------------------------------------------------------------------------
def is_discrete_array(arr):
    return np.issubdtype(arr.dtype, np.integer) or np.all(np.mod(arr, 1) == 0)


def _load_mortality_data(data_dir):
    biochemtapepath = os.path.join(data_dir, "DU4800.txt")
    medexamtapepath = os.path.join(data_dir, "DU4233.txt")
    anthropometrypath = os.path.join(data_dir, "DU4111.txt")
    vitlpath = os.path.join(data_dir, "N92vitl.txt")

    d = {}
    with open(biochemtapepath, "r") as handle:
        for line in handle:
            seqn = int(line[0:5]); d[seqn] = {}
            d[seqn]["exam_month"] = int(line[137:139]); d[seqn]["exam_year"] = int(line[141:143])
            sex = int(line[103]); d[seqn]["sex_isFemale"] = (sex == 2)
            d[seqn]["age"] = int(line[143:145])
            d[seqn]["physical_activity"] = int(line[225])
            d[seqn]["physical_activity_isBlank"] = (d[seqn]["physical_activity"] == 8)
            try: d[seqn]["serum_albumin"] = float(line[231:235])
            except ValueError: d[seqn]["serum_albumin"] = np.nan
            d[seqn]["serum_albumin_isBlank"] = np.isnan(d[seqn]["serum_albumin"])
            d[seqn]["serum_albumin_isMissingAge1to3"] = (d[seqn]["serum_albumin"] == 9999)
            if d[seqn]["serum_albumin"] == 9999: d[seqn]["serum_albumin"] = np.nan
            d[seqn]["serum_albumin"] /= 10
            try: isImputed = (int(line[235]) == 1)
            except ValueError: isImputed = False
            if isImputed:
                d[seqn]["serum_albumin"] = np.nan; d[seqn]["serum_albumin_isBlank"] = True
            try: d[seqn]["alkaline_phosphatase"] = float(line[458:462])
            except ValueError: d[seqn]["alkaline_phosphatase"] = np.nan
            d[seqn]["alkaline_phosphatase_isUnacceptable"] = (d[seqn]["alkaline_phosphatase"] == 7777)
            d[seqn]["alkaline_phosphatase_isBlankbutapplicable"] = (d[seqn]["alkaline_phosphatase"] == 8888)
            d[seqn]["alkaline_phosphatase_isTestnotdone"] = (d[seqn]["alkaline_phosphatase"] == 9999)
            d[seqn]["alkaline_phosphatase_isBlank"] = np.isnan(d[seqn]["alkaline_phosphatase"])
            if d[seqn]["alkaline_phosphatase"] in [7777, 8888, 9999]: d[seqn]["alkaline_phosphatase"] = np.nan
            d[seqn]["alkaline_phosphatase"] /= 10
            try: d[seqn]["SGOT"] = float(line[454:458])
            except ValueError: d[seqn]["SGOT"] = np.nan
            d[seqn]["SGOT_isUnacceptable"] = (d[seqn]["SGOT"] == 7777)
            d[seqn]["SGOT_isBlankbutapplicable"] = (d[seqn]["SGOT"] == 8888)
            d[seqn]["SGOT_isTestnotdone"] = (d[seqn]["SGOT"] == 9999)
            d[seqn]["SGOT_isBlank"] = np.isnan(d[seqn]["SGOT"])
            if d[seqn]["SGOT"] in [7777, 8888, 9999]: d[seqn]["SGOT"] = np.nan
            d[seqn]["SGOT"] /= 100
            try: d[seqn]["BUN"] = float(line[471:474])
            except ValueError: d[seqn]["BUN"] = np.nan
            d[seqn]["BUN_isUnacceptable"] = (d[seqn]["BUN"] == 777)
            d[seqn]["BUN_isTestnotdone"] = (d[seqn]["BUN"] == 999)
            d[seqn]["BUN_isBlank"] = np.isnan(d[seqn]["BUN"])
            if d[seqn]["BUN"] in [777, 999]: d[seqn]["BUN"] = np.nan
            d[seqn]["BUN"] /= 10
            try: d[seqn]["calcium"] = float(line[465:468])
            except ValueError: d[seqn]["calcium"] = np.nan
            d[seqn]["calcium_isUnacceptable"] = (d[seqn]["calcium"] == 777)
            d[seqn]["calcium_isBlankbutapplicable"] = (d[seqn]["calcium"] == 888)
            d[seqn]["calcium_isTestnotdone"] = (d[seqn]["calcium"] == 999)
            d[seqn]["calcium_isBlank"] = np.isnan(d[seqn]["calcium"])
            if d[seqn]["calcium"] in [777, 888, 999]: d[seqn]["calcium"] = np.nan
            d[seqn]["calcium"] /= 10
            try: d[seqn]["creatinine"] = float(line[474:477])
            except ValueError: d[seqn]["creatinine"] = np.nan
            d[seqn]["creatinine_isUnacceptable"] = (d[seqn]["creatinine"] == 777)
            d[seqn]["creatinine_isTestnotdone"] = (d[seqn]["creatinine"] == 999)
            d[seqn]["creatinine_isBlank"] = np.isnan(d[seqn]["creatinine"])
            if d[seqn]["creatinine"] in [777, 999]: d[seqn]["creatinine"] = np.nan
            d[seqn]["creatinine"] /= 10
            try: d[seqn]["potassium"] = float(line[273:276])
            except ValueError: d[seqn]["potassium"] = np.nan
            d[seqn]["potassium_isUnacceptable"] = (d[seqn]["potassium"] == 888)
            d[seqn]["potassium_isBlank"] = np.isnan(d[seqn]["potassium"])
            if d[seqn]["potassium"] in [888]: d[seqn]["potassium"] = np.nan
            d[seqn]["potassium"] /= 10
            try: d[seqn]["sodium"] = float(line[270:273])
            except ValueError: d[seqn]["sodium"] = np.nan
            d[seqn]["sodium_isUnacceptable"] = (d[seqn]["sodium"] == 888)
            d[seqn]["sodium_isBlank"] = np.isnan(d[seqn]["sodium"])
            if d[seqn]["sodium"] in [888]: d[seqn]["sodium"] = np.nan
            try: d[seqn]["total_bilirubin"] = float(line[450:454])
            except ValueError: d[seqn]["total_bilirubin"] = np.nan
            d[seqn]["total_bilirubin_isUnacceptable"] = (d[seqn]["total_bilirubin"] == 7777)
            d[seqn]["total_bilirubin_isBlankbutapplicable"] = (d[seqn]["total_bilirubin"] == 8888)
            d[seqn]["total_bilirubin_isTestnotdone"] = (d[seqn]["total_bilirubin"] == 9999)
            d[seqn]["total_bilirubin_isBlank"] = np.isnan(d[seqn]["total_bilirubin"])
            if d[seqn]["total_bilirubin"] in [7777, 8888, 9999]: d[seqn]["total_bilirubin"] = np.nan
            d[seqn]["total_bilirubin"] /= 100
            try: d[seqn]["serum_protein"] = float(line[226:230])
            except ValueError: d[seqn]["serum_protein"] = np.nan
            d[seqn]["serum_protein_isMissingAge1to3"] = (d[seqn]["serum_protein"] == 9999)
            d[seqn]["serum_protein_isBlank"] = np.isnan(d[seqn]["serum_protein"])
            if d[seqn]["serum_protein"] in [7777, 8888, 9999]: d[seqn]["serum_protein"] = np.nan
            d[seqn]["serum_protein"] /= 10
            try: isImputed = (int(line[230]) == 1)
            except ValueError: isImputed = False
            if isImputed:
                d[seqn]["serum_protein"] = np.nan; d[seqn]["serum_protein_isBlank"] = True
            try: d[seqn]["red_blood_cells"] = float(line[525:528])
            except ValueError: d[seqn]["red_blood_cells"] = np.nan
            d[seqn]["red_blood_cells_isUnacceptable"] = (d[seqn]["red_blood_cells"] == 777)
            d[seqn]["red_blood_cells_isBlankbutapplicable"] = (d[seqn]["red_blood_cells"] == 888)
            if d[seqn]["red_blood_cells"] in [777, 888]: d[seqn]["red_blood_cells"] = np.nan
            d[seqn]["red_blood_cells"] /= 100
            try: d[seqn]["white_blood_cells"] = float(line[528:531])
            except ValueError: d[seqn]["white_blood_cells"] = np.nan
            d[seqn]["white_blood_cells_isUnacceptable"] = (d[seqn]["white_blood_cells"] == 777)
            d[seqn]["white_blood_cells_isBlankbutapplicable"] = (d[seqn]["white_blood_cells"] == 888)
            if d[seqn]["white_blood_cells"] in [777, 888]: d[seqn]["white_blood_cells"] = np.nan
            d[seqn]["white_blood_cells"] /= 10
            try: d[seqn]["hemoglobin"] = float(line[246:250])
            except ValueError: d[seqn]["hemoglobin"] = np.nan
            d[seqn]["hemoglobin_isMissing"] = (d[seqn]["hemoglobin"] == 8888)
            d[seqn]["hemoglobin_isUnacceptable"] = (d[seqn]["hemoglobin"] == 7777)
            d[seqn]["hemoglobin_isBlank"] = np.isnan(d[seqn]["hemoglobin"])
            if d[seqn]["hemoglobin"] in [7777, 8888]: d[seqn]["hemoglobin"] = np.nan
            d[seqn]["hemoglobin"] /= 10
            try: isImputed = (int(line[250]) == 1)
            except ValueError: isImputed = False
            if isImputed:
                d[seqn]["hemoglobin"] = np.nan; d[seqn]["hemoglobin_isBlank"] = True
            try: d[seqn]["hematocrit"] = float(line[251:254])
            except ValueError: d[seqn]["hematocrit"] = np.nan
            d[seqn]["hematocrit_isUnacceptable"] = (d[seqn]["hematocrit"] == 777)
            d[seqn]["hematocrit_isMissing"] = (d[seqn]["hematocrit"] == 888)
            d[seqn]["hematocrit_isBlank"] = np.isnan(d[seqn]["hematocrit"])
            if d[seqn]["hematocrit"] in [777, 888]: d[seqn]["hematocrit"] = np.nan
            try: isImputed = (int(line[254]) == 1)
            except ValueError: isImputed = False
            if isImputed:
                d[seqn]["hematocrit"] = np.nan; d[seqn]["hematocrit_isBlank"] = True
            try: platelets = int(line[337])
            except: platelets = np.nan
            d[seqn]["platelets_isNormal"] = (platelets == 0)
            d[seqn]["platelets_isIncreased"] = (platelets == 2)
            d[seqn]["platelets_isDecreased"] = (platelets == 3)
            d[seqn]["platelets_isNoestimate"] = (platelets == 9)
            d[seqn]["platelets_isBlank"] = np.isnan(platelets)
            try: d[seqn]["segmented_neutrophils"] = float(line[320:322])
            except ValueError: d[seqn]["segmented_neutrophils"] = np.nan
            d[seqn]["segmented_neutrophils_isBlank"] = np.isnan(d[seqn]["segmented_neutrophils"])
            try: d[seqn]["lymphocytes"] = float(line[326:328])
            except ValueError: d[seqn]["lymphocytes"] = np.nan
            d[seqn]["lymphocytes_isBlank"] = np.isnan(d[seqn]["lymphocytes"])
            try: d[seqn]["monocytes"] = float(line[328:330])
            except ValueError: d[seqn]["monocytes"] = np.nan
            d[seqn]["monocytes_isBlank"] = np.isnan(d[seqn]["monocytes"])
            try: d[seqn]["eosinophils"] = float(line[322:324])
            except ValueError: d[seqn]["eosinophils"] = np.nan
            d[seqn]["eosinophils_isBlank"] = np.isnan(d[seqn]["eosinophils"])
            try: d[seqn]["basophils"] = float(line[324:326])
            except ValueError: d[seqn]["basophils"] = np.nan
            d[seqn]["basophils_isBlank"] = np.isnan(d[seqn]["basophils"])
            try: d[seqn]["band_neutrophils"] = float(line[318:320])
            except ValueError: d[seqn]["band_neutrophils"] = np.nan
            d[seqn]["band_neutrophils_isBlank"] = np.isnan(d[seqn]["band_neutrophils"])
            try: d[seqn]["cholesterol"] = float(line[236:240])
            except ValueError: d[seqn]["cholesterol"] = np.nan
            d[seqn]["cholesterol_isMissing"] = (d[seqn]["cholesterol"] == 8888)
            d[seqn]["cholesterol_isMissingAge1to3"] = (d[seqn]["cholesterol"] == 9999)
            d[seqn]["cholesterol_isBlank"] = np.isnan(d[seqn]["cholesterol"])
            if d[seqn]["cholesterol"] in [8888, 9999]: d[seqn]["cholesterol"] = np.nan
            try: isImputed = (int(line[240]) == 1)
            except ValueError: isImputed = False
            if isImputed:
                d[seqn]["cholesterol"] = np.nan; d[seqn]["cholesterol_isBlank"] = True
            try: urine_albumin = int(line[500])
            except ValueError: urine_albumin = np.nan
            d[seqn]["urine_albumin_isNegative"] = (urine_albumin == 0)
            d[seqn]["urine_albumin_is>=30"] = (urine_albumin == 1)
            d[seqn]["urine_albumin_is>=100"] = (urine_albumin == 2)
            d[seqn]["urine_albumin_is>=300"] = (urine_albumin == 3)
            d[seqn]["urine_albumin_is>=1000"] = (urine_albumin == 4)
            d[seqn]["urine_albumin_isTrace"] = (urine_albumin == 5)
            d[seqn]["urine_albumin_isBlankbutapplicable"] = (urine_albumin == 8)
            d[seqn]["urine_albumin_isBlank"] = np.isnan(urine_albumin)
            try: urine_glucose = int(line[501])
            except ValueError: urine_glucose = np.nan
            d[seqn]["urine_glucose_isNegative"] = (urine_glucose == 0)
            d[seqn]["urine_glucose_isLight"] = (urine_glucose == 1)
            d[seqn]["urine_glucose_isMedium"] = (urine_glucose == 2)
            d[seqn]["urine_glucose_isDark"] = (urine_glucose == 3)
            d[seqn]["urine_glucose_isVerydark"] = (urine_glucose == 4)
            d[seqn]["urine_glucose_isTrace"] = (urine_glucose == 5)
            d[seqn]["urine_glucose_isBlankbutapplicable"] = (urine_glucose == 8)
            d[seqn]["urine_glucose_isBlank"] = np.isnan(urine_glucose)
            try: d[seqn]["urine_pH"] = int(line[502])
            except ValueError: d[seqn]["urine_pH"] = np.nan
            d[seqn]["urine_pH_isBlank"] = np.isnan(d[seqn]["urine_pH"])
            if d[seqn]["urine_pH"] == 4:
                d[seqn]["urine_pH_isBlankbutapplicable"] = True; d[seqn]["urine_pH"] = np.nan
            else: d[seqn]["urine_pH_isBlankbutapplicable"] = False
            try: urine_hematest = int(line[503])
            except ValueError: urine_hematest = np.nan
            d[seqn]["urine_hematest_isNegative"] = (urine_hematest == 0)
            d[seqn]["urine_hematest_isSmall"] = (urine_hematest == 1)
            d[seqn]["urine_hematest_isModerate"] = (urine_hematest == 2)
            d[seqn]["urine_hematest_isLarge"] = (urine_hematest == 3)
            d[seqn]["urine_hematest_isVerylarge"] = (urine_hematest == 4)
            d[seqn]["urine_hematest_isTrace"] = (urine_hematest == 5)
            d[seqn]["urine_hematest_isBlankbutapplicable"] = (urine_hematest == 8)
            d[seqn]["urine_hematest_isBlank"] = np.isnan(urine_hematest)
            try: d[seqn]["sedimentation_rate"] = float(line[279:282])
            except ValueError: d[seqn]["sedimentation_rate"] = np.nan
            d[seqn]["sedimentation_rate_isBlank"] = np.isnan(d[seqn]["sedimentation_rate"])
            if d[seqn]["sedimentation_rate"] == 888:
                d[seqn]["sedimentation_rate_isBlankbutapplicable"] = True; d[seqn]["sedimentation_rate"] = np.nan
            else: d[seqn]["sedimentation_rate_isBlankbutapplicable"] = False
            try: d[seqn]["uric_acid"] = float(line[462:465])
            except ValueError: d[seqn]["uric_acid"] = np.nan
            d[seqn]["uric_acid_isUnacceptable"] = (d[seqn]["uric_acid"] == 777)
            d[seqn]["uric_acid_isBlankbutapplicable"] = (d[seqn]["uric_acid"] == 888)
            d[seqn]["uric_acid_isTestnotdone"] = (d[seqn]["uric_acid"] == 999)
            d[seqn]["uric_acid_isBlank"] = np.isnan(d[seqn]["uric_acid"])
            if d[seqn]["uric_acid"] in [777, 888, 999]: d[seqn]["uric_acid"] = np.nan
            d[seqn]["uric_acid"] /= 10
    with open(medexamtapepath, "r") as handle:
        for line in handle:
            seqn = int(line[:5])
            try: d[seqn]["systolic_blood_pressure"] = int(line[227:230])
            except ValueError: d[seqn]["systolic_blood_pressure"] = np.nan
            d[seqn]["systolic_blood_pressure_isBlank"] = (d[seqn]["systolic_blood_pressure"] == 888)
            d[seqn]["systolic_blood_pressure_isAgeUnder6"] = (d[seqn]["systolic_blood_pressure"] == 999)
            if d[seqn]["systolic_blood_pressure"] in [888, 999]: d[seqn]["systolic_blood_pressure"] = np.nan
            try: diastolic = int(line[230:233])
            except ValueError: diastolic = np.nan
            if diastolic in [888, 999]: diastolic = np.nan
            if np.isnan(d[seqn]["systolic_blood_pressure"]) or np.isnan(diastolic):
                d[seqn]["pulse_pressure"] = np.nan
            else: d[seqn]["pulse_pressure"] = d[seqn]["systolic_blood_pressure"] - diastolic
    with open(anthropometrypath, "r") as handle:
        for line in handle:
            seqn = int(line[0:5])
            try: d[seqn]["weight"] = int(line[259:264])
            except ValueError: d[seqn]["weight"] = np.nan
            if d[seqn]["weight"] == 88888: d[seqn]["weight"] = np.nan
            weightIsImputed = (int(line[264]) == 1)
            if weightIsImputed: d[seqn]["weight"] = np.nan
            d[seqn]["weight_isBlank"] = np.isnan(d[seqn]["weight"])
            d[seqn]["weight"] /= 100
            try: d[seqn]["height"] = int(line[265:269])
            except ValueError: d[seqn]["height"] = np.nan
            if d[seqn]["height"] == 8888: d[seqn]["height"] = np.nan
            heightIsImputed = (int(line[272]) == 1)
            if heightIsImputed: d[seqn]["height"] = np.nan
            d[seqn]["height_isBlank"] = np.isnan(d[seqn]["height"])
            d[seqn]["height"] /= 10
    d2 = {}
    with open(vitlpath, "r") as handle:
        for line in handle:
            seqn = int(line[11:16])
            try: d[seqn]
            except KeyError: continue
            d2[seqn] = {}
            d2[seqn]["month_last_known_alive"] = int(line[17:19])
            d2[seqn]["month_last_known_alive_isDontknow"] = (d2[seqn]["month_last_known_alive"] == 98)
            d2[seqn]["month_last_known_alive_isNotascertained"] = (d2[seqn]["month_last_known_alive"] == 99)
            if d2[seqn]["month_last_known_alive"] in [98, 99]: d2[seqn]["month_last_known_alive"] = np.nan
            d2[seqn]["year_last_known_alive"] = int(line[21:23])
            try: d2[seqn]["month_deceased"] = int(line[60:62])
            except ValueError: d2[seqn]["month_deceased"] = np.nan
            try: d2[seqn]["year_deceased"] = int(line[64:66])
            except ValueError: d2[seqn]["year_deceased"] = np.nan
            d2[seqn]["month_deceased_isDontknow"] = (d2[seqn]["month_deceased"] == 98)
            d2[seqn]["month_deceased_isNotascertained"] = (d2[seqn]["month_deceased"] == 99)
            d2[seqn]["month_deceased_isBlank"] = np.isnan(d2[seqn]["month_deceased"])
            if d2[seqn]["month_deceased"] in [98, 99]: d2[seqn]["month_deceased"] = np.nan
            d2[seqn]["year_deceased_isBlank"] = np.isnan(d2[seqn]["year_deceased"])
    for seqn in list(d.keys()):
        try: d2[seqn]
        except KeyError:
            d[seqn]["survived_15_years"] = np.nan
            del d[seqn]["exam_year"]; del d[seqn]["exam_month"]; continue
        alive_time = d2[seqn]["year_last_known_alive"] + d2[seqn]["month_last_known_alive"] / 12.0
        dead_time = d2[seqn]["year_deceased"] + d2[seqn]["month_deceased"] / 12.0
        exam_time = d[seqn]["exam_year"] + d[seqn]["exam_month"] / 12.0
        if np.isnan(dead_time):
            if alive_time > exam_time: years = -(alive_time - exam_time)
        else:
            if dead_time >= alive_time and dead_time > exam_time: years = dead_time - exam_time
        d[seqn]["survived_15_years"] = years
        del d[seqn]["exam_year"]; del d[seqn]["exam_month"]
    dataframe = pd.DataFrame.from_dict(d, orient="index")
    y = np.array(dataframe["survived_15_years"], dtype=float)
    del dataframe["survived_15_years"]
    bad_idxs = list(np.where(np.isnan(y))[0])
    dataframe.drop(dataframe.index[bad_idxs], inplace=True)
    y = y[~np.isnan(y)]
    for colname in list(dataframe.columns):
        if np.all(dataframe[colname] == dataframe[colname].iloc[0]): dataframe.drop(colname, inplace=True, axis=1)
    for colname in list(dataframe.columns):
        assert not np.all(dataframe[colname] == dataframe[colname].iloc[0])
    for c in dataframe.columns:
        if c.endswith("_isBlank"): del dataframe[c]
    dataframe["bmi"] = 10000 * dataframe["weight"].values.copy() / (dataframe["height"].values.copy() * dataframe["height"].values.copy())
    del dataframe["weight"]; del dataframe["height"]
    del dataframe["urine_hematest_isTrace"]; del dataframe["SGOT_isBlankbutapplicable"]
    del dataframe["calcium_isBlankbutapplicable"]; del dataframe["uric_acid_isBlankbutapplicable"]
    del dataframe["urine_hematest_isVerylarge"]; del dataframe["total_bilirubin_isBlankbutapplicable"]
    del dataframe["alkaline_phosphatase_isBlankbutapplicable"]; del dataframe["hemoglobin_isUnacceptable"]
    del dataframe["alkaline_phosphatase_isUnacceptable"]; del dataframe["alkaline_phosphatase_isTestnotdone"]
    del dataframe["SGOT_isUnacceptable"]; del dataframe["SGOT_isTestnotdone"]
    del dataframe["BUN_isUnacceptable"]; del dataframe["BUN_isTestnotdone"]
    del dataframe["calcium_isUnacceptable"]; del dataframe["calcium_isTestnotdone"]
    del dataframe["creatinine_isUnacceptable"]; del dataframe["creatinine_isTestnotdone"]
    del dataframe["potassium_isUnacceptable"]; del dataframe["sodium_isUnacceptable"]
    del dataframe["total_bilirubin_isUnacceptable"]; del dataframe["total_bilirubin_isTestnotdone"]
    del dataframe["red_blood_cells_isUnacceptable"]; del dataframe["red_blood_cells_isBlankbutapplicable"]
    del dataframe["white_blood_cells_isUnacceptable"]; del dataframe["white_blood_cells_isBlankbutapplicable"]
    del dataframe["hemoglobin_isMissing"]; del dataframe["hematocrit_isUnacceptable"]; del dataframe["hematocrit_isMissing"]
    del dataframe["cholesterol_isMissing"]; del dataframe["urine_albumin_isBlankbutapplicable"]
    del dataframe["urine_glucose_isBlankbutapplicable"]; del dataframe["urine_pH_isBlankbutapplicable"]
    del dataframe["urine_hematest_isBlankbutapplicable"]; del dataframe["sedimentation_rate_isBlankbutapplicable"]
    del dataframe["uric_acid_isUnacceptable"]; del dataframe["uric_acid_isTestnotdone"]
    rows = np.where(np.invert(np.isnan(dataframe["systolic_blood_pressure"]) | np.isnan(dataframe["bmi"])))[0]
    dataframe = dataframe.iloc[rows, :]; y = y[rows]
    name_map = {
        "sex_isFemale": "Sex", "age": "Age", "systolic_blood_pressure": "Systolic blood pressure",
        "bmi": "BMI", "white_blood_cells": "White blood cells", "sedimentation_rate": "Sedimentation rate",
        "serum_albumin": "Blood albumin", "alkaline_phosphatase": "Alkaline phosphatase",
        "cholesterol": "Total cholesterol", "physical_activity": "Physical activity", "hematocrit": "Hematocrit",
        "uric_acid": "Uric acid", "red_blood_cells": "Red blood cells", "urine_albumin_isNegative": "Albumin present in urine",
        "serum_protein": "Blood protein"}
    feat_names = list(map(lambda x: name_map.get(x, x), dataframe.columns))
    X = SimpleImputer().fit_transform(dataframe)
    y = y.reshape(-1, 1)
    categorical = np.zeros(X.shape[1], dtype=bool)
    for i in range(X.shape[1]):
        if is_discrete_array(X[:, i]) and np.unique(X[:, i]).size == 2: categorical[i] = True
    continuous_indices = [i for i in range(X.shape[1]) if not categorical[i]]
    if len(continuous_indices) > 0:
        X[:, continuous_indices] = MaxAbsScaler().fit_transform(X[:, continuous_indices])
    return X, y, feat_names


def make_mortality_csv():
    X, y, feat_names = _load_mortality_data(os.path.join(ROOT, "mortality"))
    df = pd.DataFrame(X, columns=[str(c) for c in feat_names])
    df["target"] = y.ravel()
    op = os.path.join(OUT, "mortality_upload.csv")
    df.to_csv(op, index=False)
    print(f"[mortality] X={X.shape} -> {op}  (target=survived_15_years, 回归任务)")
    return op


if __name__ == "__main__":
    make_enhancer_csv()
    make_mortality_csv()
    print("done ->", OUT)
