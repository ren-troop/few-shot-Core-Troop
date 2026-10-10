"""获取 / 校验 exp2、exp5 需要的 USGS 水数据。

用法
----
    python ssl_maml/prepare_water_data.py --verify
        校验 data/water_dataset.mat 是不是 expected 的那份（站点、特征、形状、范围）。

    python ssl_maml/prepare_water_data.py --download
        从 USGS 官方 Water Services API 拉这 37 个站的原始日值，
        写到 data/usgs_raw/，并在 manifest.md 里记录每一条请求的完整 URL。

    python ssl_maml/prepare_water_data.py --download --start 2015-10-01 --end 2017-09-06
        指定时间段（默认就是上面这个）。

为什么是「下载原始数据 + 校验」，而不是「一键重建 water_dataset.mat」
--------------------------------------------------------------------
`water_dataset.mat` 是 Zhao / Gkountouna / Pfoser 在 ACM TSAS 2019 那篇论文里
**整理过**的版本，不是 USGS 原始日值的直接导出：

* 37 个站里有 16 个的连续监测记录是 2012–2015 年才开始的；
* 在 2015-10 之后，**不存在一段 705 天里 37 个站每天都有数据**（最长完整区间只有 47 天）。
  所以论文的「705 天」一定做过缺测填补，或者挑的是非连续的天，而这个规则没有公开；
* 数据还做过按特征 min-max 归一化，用的是哪一段时间/哪些站的极值也没有公开。

重建规则一旦猜错，产出的是一份**同名不同数**的数据，别人拿去跑实验会得到对不上的结果——
比不给脚本更糟。所以本脚本只做两件能验证的事：拉原始数据（可独立核查），校验成品文件。

来源与许可
----------
USGS National Water Information System (NWIS)，公开数据，无需 API key：
https://waterservices.usgs.gov/nwis/dv/
整理出处：Liang Zhao, Olga Gkountouna, Dieter Pfoser.
*Spatial Auto-regressive Dependency Interpretable Learning Based on Spatial
Topological Constraints.* ACM TSAS 5(3), Article 19, 2019. DOI 10.1145/3339823
"""
import os
import sys

# 必须在 import urllib 之前做：
# 本文件位于 ssl_maml/ 目录下，而该目录里有个 ssl.py（半监督损失项）。
# 直接运行本脚本时 Python 会把脚本所在目录放进 sys.path[0]，
# 导致 `import ssl` 命中我们的 ssl.py 而不是标准库，urllib 无法启用 HTTPS，
# 报 "unknown url type: https"。所以先把 sys.path[0] 换成仓库根目录。
sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse          # noqa: E402
import json              # noqa: E402
import time              # noqa: E402
import urllib.error      # noqa: E402
import urllib.request    # noqa: E402

WS = sys.path[0]         # 仓库根目录

# 仓库根目录：ssl_maml/prepare_water_data.py 往上两层
WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WS)

DATA_MAT = os.path.join(WS, "data", "water_dataset.mat")
RAW_DIR = os.path.join(WS, "data", "usgs_raw")

# ---------------------------------------------------------------- 站点与特征
# water_dataset.mat 里的 location_ids 丢了前导 0，补回来就是 USGS station no.
_RAW_IDS = [2198840, 2198920, 2198950, 2203603, 2203655, 2203700, 2203831,
            2203863, 2203873, 2203900, 2203950, 2203960, 2204037, 2207135,
            2207160, 2208450, 2208493, 2336120, 2336152, 2336240, 2336300,
            2336313, 2336340, 2336360, 2336410, 2336526, 2336728, 2337170,
            2344630, 2344673, 21989715, 21989773, 21989792, 21989793,
            22035975, 23362095, 219897945]
SITES = ["0" + str(i) for i in _RAW_IDS]

# water_dataset.mat 的 features 顺序 = USGS 参数码 × 统计量
#   00095 电导率  00400 pH  00300 溶解氧  00010 水温
#   00001 Max     00002 Min 00003 Mean
FEATURES = [
    ("00095", "00001", "Specific conductance, water, unfiltered, "
                       "microsiemens per centimeter at 25 degrees Celsius (Maximum)"),
    ("00400", "00001", "pH, water, unfiltered, field, standard units (Maximum)"),
    ("00400", "00002", "pH, water, unfiltered, field, standard units (Minimum)"),
    ("00095", "00002", "Specific conductance, water, unfiltered, "
                       "microsiemens per centimeter at 25 degrees Celsius (Minimum)"),
    ("00095", "00003", "Specific conductance, water, unfiltered, "
                       "microsiemens per centimeter at 25 degrees Celsius (Mean)"),
    ("00300", "00001", "Dissolved oxygen, water, unfiltered, "
                       "milligrams per liter (Maximum)"),
    ("00300", "00003", "Dissolved oxygen, water, unfiltered, "
                       "milligrams per liter (Mean)"),
    ("00300", "00002", "Dissolved oxygen, water, unfiltered, "
                       "milligrams per liter (Minimum)"),
    ("00010", "00003", "Temperature, water, degrees Celsius (Mean)"),
    ("00010", "00002", "Temperature, water, degrees Celsius (Minimum)"),
    ("00010", "00001", "Temperature, water, degrees Celsius (Maximum)"),
]
N_SITES, N_FEAT, N_TRAIN, N_TEST = 37, 11, 423, 282
API = "https://waterservices.usgs.gov/nwis/dv/"


def _get(url, tries=3, timeout=300):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if k == tries - 1:
                raise
            print(f"    重试 {k + 1}/{tries - 1} ({type(e).__name__}) ...")
            time.sleep(3 * (k + 1))


# ------------------------------------------------------------------- 下载
def download(start, end):
    os.makedirs(RAW_DIR, exist_ok=True)
    manifest = ["# USGS 原始日值下载清单", "",
                f"时间段：{start} -> {end}", f"站点数：{len(SITES)}", ""]
    for param, stat, label in FEATURES:
        url = (f"{API}?format=json&sites={','.join(SITES)}"
               f"&startDT={start}&endDT={end}"
               f"&parameterCd={param}&statCd={stat}&siteStatus=all")
        print(f"[{param}/{stat}] {label[:52]}")
        d = _get(url)
        ts = d["value"]["timeSeries"]
        rows, ndays_all = [], set()
        for t in ts:
            code = t["sourceInfo"]["siteCode"][0]["value"]
            name = t["sourceInfo"]["siteName"]
            for v in t["values"][0]["value"]:
                try:
                    f = float(v["value"])
                except ValueError:
                    continue
                if f <= -900:
                    continue
                dt = v["dateTime"][:10]
                rows.append((code, name, dt, f))
                ndays_all.add(dt)
        fn = os.path.join(RAW_DIR, f"{param}_{stat}.csv")
        with open(fn, "w", encoding="utf-8", newline="") as fh:
            fh.write("site_no,site_name,date,value\n")
            for r in sorted(rows, key=lambda r: (r[0], r[2])):
                fh.write(f'{r[0]},"{r[1]}",{r[2]},{r[3]}\n')
        print(f"    站点 {len(ts)}，记录 {len(rows)} 条，覆盖 {len(ndays_all)} 天"
              f" -> {os.path.relpath(fn, WS)}")
        manifest += [f"## {label}", "", f"```", url, f"```",
                     f"- 返回站点：{len(ts)} - 记录数：{len(rows)}", ""]
    with open(os.path.join(RAW_DIR, "manifest.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(manifest))
    print(f"\n清单 -> {os.path.relpath(os.path.join(RAW_DIR, 'manifest.md'), WS)}")


# ------------------------------------------------------------------- 校验
def verify():
    print(f"校验 {os.path.relpath(DATA_MAT, WS)}")
    if not os.path.exists(DATA_MAT):
        print("  文件不存在。获取方式见 ssl_maml/README.md 第 4 节。")
        return 1
    from ssl_maml.matio import loadmat
    d = loadmat(DATA_MAT)
    ok = True
    need = {"X_tr", "X_te", "Y_tr", "Y_te", "location_ids", "features",
            "location_group"}
    miss = need - set(d)
    if miss:
        print("  !! 缺变量:", sorted(miss)); ok = False
    if "X_tr" in d and (d["X_tr"].shape != (1, N_TRAIN)
                        or d["X_tr"].ravel()[0].shape != (N_SITES, N_FEAT)):
        print(f"  !! X_tr 形状异常: {d['X_tr'].shape} / "
              f"{d['X_tr'].ravel()[0].shape}，应为 (1,{N_TRAIN}) / ({N_SITES},{N_FEAT})")
        ok = False
    if "X_te" in d and (d["X_te"].shape != (1, N_TEST)
                        or d["X_te"].ravel()[0].shape != (N_SITES, N_FEAT)):
        print(f"  !! X_te 形状异常: {d['X_te'].shape} / "
              f"{d['X_te'].ravel()[0].shape}"); ok = False
    for k, shp in (("Y_tr", (N_SITES, N_TRAIN)), ("Y_te", (N_SITES, N_TEST))):
        if k in d and d[k].shape != shp:
            print(f"  !! {k} 形状 {d[k].shape}，应为 {shp}"); ok = False
    if "location_ids" in d:
        got = [int(x) for x in d["location_ids"].ravel()]
        want = _RAW_IDS
        if got != want:
            print("  !! 站点 ID 不匹配")
            print("     文件里:", got[:6], "...")
            print("     期望  :", want[:6], "...")
            ok = False
    if "features" in d:
        got = [str(x).strip() for x in d["features"].ravel()]
        want = [f[2] for f in FEATURES]
        norm = lambda s: " ".join(s.split()).lower()
        if [norm(g) for g in got] != [norm(w) for w in want]:
            print("  !! features 名称/顺序不匹配")
            for i, (g, w) in enumerate(zip(got, want)):
                if norm(g) != norm(w):
                    print(f"     [{i}] 文件: {g[:60]}")
                    print(f"         期望: {w[:60]}")
            ok = False
    print("  站点 %d，特征 %d，训练 %d 天，测试 %d 天" % (N_SITES, N_FEAT,
                                                N_TRAIN, N_TEST))
    print("\n结论:", "[OK] 文件正确" if ok else "[FAIL] 与预期不一致，请重新获取")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--download", action="store_true",
                    help="从 USGS 拉原始日值到 data/usgs_raw/")
    ap.add_argument("--verify", action="store_true",
                    help="校验 data/water_dataset.mat")
    ap.add_argument("--start", default="2015-10-01")
    ap.add_argument("--end", default="2017-09-06")
    a = ap.parse_args()
    if a.download:
        download(a.start, a.end)
        if not a.verify:
            return 0
    if a.verify or not a.download:
        return verify()
    return 0


if __name__ == "__main__":
    sys.exit(main())
