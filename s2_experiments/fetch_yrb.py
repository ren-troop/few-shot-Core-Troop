# -*- coding: utf-8 -*-
"""抓取黄河水文数据：GloFAS 径流（目标）+ ERA5 气象（协变量）。"""
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

OUT = r"C:\Users\wjx\Desktop\半监督研究\yrb_daily.csv"
FLOOD = "https://flood-api.open-meteo.com/v1/flood"
ARCH = "https://archive-api.open-meteo.com/v1/archive"
START, END = "2015-01-01", "2019-12-31"

PTS = [
    ("玛曲", 33.98, 102.07), ("兰州", 36.06, 103.83), ("中卫", 37.51, 105.19),
    ("石嘴山", 38.98, 106.38), ("巴彦高勒", 40.33, 107.00),
    ("三湖河", 40.60, 109.00), ("头道拐", 40.27, 111.07),
    ("吴堡", 37.45, 110.74), ("龙门", 35.68, 110.60), ("潼关", 34.61, 110.29),
    ("三门峡", 34.83, 111.35), ("小浪底", 34.93, 112.40),
    ("花园口", 34.90, 113.66), ("高村", 35.38, 115.05), ("艾山", 36.30, 116.30),
    ("利津", 37.52, 118.30),
    ("西宁_湟水", 36.62, 101.78), ("临洮_洮河", 35.38, 103.86),
    ("绥德_无定河", 37.51, 110.26), ("河津_汾河", 35.60, 110.70),
    ("华县_渭河", 34.50, 109.77), ("彬县_泾河", 35.04, 108.08),
    ("大荔_北洛河", 34.80, 109.94), ("武陟_沁河", 35.10, 113.40),
]


def get(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(2 + 2 * i)


def main():
    rows = []
    for k, (name, lat, lon) in enumerate(PTS, 1):
        # 径流
        u1 = ("%s?latitude=%.4f&longitude=%.4f&daily=river_discharge"
              "&start_date=%s&end_date=%s" % (FLOOD, lat, lon, START, END))
        d1 = get(u1)
        dd1 = d1["daily"]
        # 气象
        u2 = ("%s?latitude=%.4f&longitude=%.4f&start_date=%s&end_date=%s"
              "&daily=precipitation_sum,temperature_2m_mean,"
              "temperature_2m_max,et0_fao_evapotranspiration,"
              "soil_moisture_0_to_7cm_mean,wind_speed_10m_mean"
              % (ARCH, lat, lon, START, END))
        d2 = get(u2)
        dd2 = d2["daily"]
        t1, t2 = dd1.get("time", []), dd2.get("time", [])
        assert t1 == t2, "时间轴不一致"
        q = dd1.get("river_discharge", [None] * len(t1))
        for i, day in enumerate(t1):
            rows.append(dict(
                site=name, lat=lat, lon=lon, date=day,
                discharge=q[i],
                prec=dd2["precipitation_sum"][i],
                temp=dd2["temperature_2m_mean"][i],
                tmax=dd2["temperature_2m_max"][i],
                et0=dd2["et0_fao_evapotranspiration"][i],
                soil=dd2["soil_moisture_0_to_7cm_mean"][i],
                wind=dd2["wind_speed_10m_mean"][i]))
        qq = [v for v in q if v is not None]
        print("  [%2d/%d] %-12s 天数 %4d  径流非空 %4d  均值 %10.3f  最大 %10.3f"
              % (k, len(PTS), name, len(t1), len(qq),
                 float(np.mean(qq)) if qq else float("nan"),
                 float(np.max(qq)) if qq else float("nan")))
        time.sleep(0.3)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print("\n已存:", OUT, " 行数", len(df), " (%.2f MB)"
          % (os.path.getsize(OUT) / 1e6))
    print("\n=== 各站点径流统计（m3/s）===")
    g = df.groupby("site")["discharge"].agg(["count", "mean", "min", "max"])
    print(g.round(3).to_string())
    print("\n=== 协变量缺失率 ===")
    for c in ["prec", "temp", "tmax", "et0", "soil", "wind"]:
        print("  %-6s %5.1f%%" % (c, 100 * df[c].isna().mean()))


if __name__ == "__main__":
    main()
