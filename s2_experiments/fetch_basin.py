# -*- coding: utf-8 -*-
"""按流域抓 GloFAS 径流 + ERA5 气象。用法: python fetch_basin.py <basin_key>"""
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

WS = r"C:\Users\wjx\Desktop\半监督研究"
FLOOD = "https://flood-api.open-meteo.com/v1/flood"
ARCH = "https://archive-api.open-meteo.com/v1/archive"
START, END = "2015-01-01", "2019-12-31"

BASINS = {
    "yangtze": [
        ("宜宾", 28.77, 104.62), ("泸州", 28.87, 105.44), ("重庆", 29.56, 106.55),
        ("涪陵", 29.71, 107.39), ("万州", 30.81, 108.41), ("宜昌", 30.70, 111.29),
        ("沙市", 30.31, 112.24), ("武汉", 30.58, 114.30), ("九江", 29.71, 116.00),
        ("安庆", 30.51, 117.05), ("铜陵", 30.94, 117.81), ("南京", 32.06, 118.80),
        ("镇江", 32.19, 119.42), ("江阴", 31.92, 120.28), ("南通", 31.98, 120.89),
        ("襄阳_汉水", 32.04, 112.14), ("长沙_湘江", 28.19, 112.98),
        ("南昌_赣江", 28.68, 115.89), ("北碚_嘉陵江", 29.83, 106.44),
        ("富阳_富春江", 30.05, 119.96),
    ],
    "mississippi": [
        ("明尼阿波利斯", 44.98, -93.27), ("圣保罗", 44.94, -93.09),
        ("达文波特", 41.52, -90.58), ("圣路易斯", 38.63, -90.20),
        ("开罗", 37.00, -89.18), ("孟菲斯", 35.15, -90.05),
        ("维克斯堡", 32.35, -90.88), ("纳奇兹", 31.56, -91.40),
        ("巴吞鲁日", 30.45, -91.19), ("新奥尔良", 29.95, -90.07),
        ("路易斯维尔_俄亥俄", 38.25, -85.76), ("杰斐逊城_密苏里", 38.58, -92.17),
        ("小石城_阿肯色", 34.75, -92.29), ("查塔努加_田纳西", 35.05, -85.31),
        ("什里夫波特_红河", 32.51, -93.75), ("奥马哈_密苏里", 41.26, -95.93),
        ("苏城_密苏里", 42.50, -96.40), ("锡达拉皮兹", 41.98, -91.67),
        ("皮奥里亚_伊利诺伊", 40.69, -89.59), ("辛辛那提_俄亥俄", 39.10, -84.51),
    ],
}


def get(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 + 2 * i)


def main(key):
    pts = BASINS[key]
    out = os.path.join(WS, "%s_daily.csv" % key)
    rows = []
    for k, (name, lat, lon) in enumerate(pts, 1):
        u1 = ("%s?latitude=%.4f&longitude=%.4f&daily=river_discharge"
              "&start_date=%s&end_date=%s" % (FLOOD, lat, lon, START, END))
        u2 = ("%s?latitude=%.4f&longitude=%.4f&start_date=%s&end_date=%s"
              "&daily=precipitation_sum,temperature_2m_mean,"
              "temperature_2m_max,et0_fao_evapotranspiration,"
              "soil_moisture_0_to_7cm_mean,wind_speed_10m_mean"
              % (ARCH, lat, lon, START, END))
        d1, d2 = get(u1), get(u2)
        dd1, dd2 = d1["daily"], d2["daily"]
        if dd1["time"] != dd2["time"]:
            print("  时间轴不一致，跳过", name)
            continue
        q = dd1.get("river_discharge", [None] * len(dd1["time"]))
        for i, day in enumerate(dd1["time"]):
            rows.append(dict(
                site=name, lat=lat, lon=lon, date=day, discharge=q[i],
                prec=dd2["precipitation_sum"][i],
                temp=dd2["temperature_2m_mean"][i],
                tmax=dd2["temperature_2m_max"][i],
                et0=dd2["et0_fao_evapotranspiration"][i],
                soil=dd2["soil_moisture_0_to_7cm_mean"][i],
                wind=dd2["wind_speed_10m_mean"][i]))
        qq = [v for v in q if v is not None]
        print("  [%2d/%d] %-16s n=%4d  径流均值 %10.3f  最大 %10.3f"
              % (k, len(pts), name, len(qq),
                 float(np.mean(qq)) if qq else float("nan"),
                 float(np.max(qq)) if qq else float("nan")))
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n已存 %s  (%d 行, %.2f MB)"
          % (out, len(df), os.path.getsize(out) / 1e6))


if __name__ == "__main__":
    main(sys.argv[1])
