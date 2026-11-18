# -*- coding: utf-8 -*-
"""
rev5_leak_check.py  (修改建议第 9 项: image-level split leakage 核查)
====================================================================
检查两件事:
  1. 文件级重复: 同一张图(内容哈希相同)是否出现在多个划分
  2. 相邻帧泄漏: 同一飞行序列(DJI_<时间戳>_<序列号>)的帧被拆到 train/test
     时, 同序列跨划分帧的内容相似度是否显著高于跨序列对照

用法: python rev5_leak_check.py
"""
import os
import re
import hashlib
import random
import collections

import numpy as np
from PIL import Image

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
SPLITS = ["train/hot", "train/normal", "val/hot", "val/normal",
          "test/hot", "test/normal", "baseline/hot", "baseline/normal"]
PAT = re.compile(r"^(DJI_\d+_\d+)_T_(\d+)\.png$")


def part1_hash_duplicates():
    print("=" * 60)
    print("PART 1  文件级重复 (md5)")
    print("=" * 60)
    h2s = collections.defaultdict(set)
    tot = 0
    for sp in SPLITS:
        d = os.path.join(BASE, sp)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if not f.lower().endswith(".png"):
                continue
            tot += 1
            h = hashlib.md5(open(os.path.join(d, f), "rb").read()).hexdigest()
            h2s[h].add(sp)
    cross = {h: s for h, s in h2s.items() if len(s) > 1}
    print("图片总数 %d | 唯一内容 %d | 跨划分重复内容 %d"
          % (tot, len(h2s), len(cross)))
    for h, s in list(cross.items())[:10]:
        print("   ", sorted(s))
    return cross


def load(p, size=(128, 128)):
    return np.asarray(Image.open(p).convert("L").resize(size),
                      dtype=np.float32)


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a * b).sum() / d) if d > 0 else 0.0


def part2_frame_leakage(n_seq=40, n_ctrl=40, seed=1):
    print()
    print("=" * 60)
    print("PART 2  相邻帧泄漏 (同序列 test 帧 vs train 帧)")
    print("=" * 60)

    seqs = collections.defaultdict(lambda: collections.defaultdict(list))
    for sp in SPLITS:
        d = os.path.join(BASE, sp)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            m = PAT.match(f)
            if m:
                seqs[m.group(1)][sp].append(f)

    print("总序列数 %d" % len(seqs))
    multi = {k: v for k, v in seqs.items() if len(v) > 1}
    print("跨多个划分的序列 %d (%.0f%%)"
          % (len(multi), 100.0 * len(multi) / len(seqs)))

    cross = {k: v for k, v in seqs.items()
             if any(s.startswith("train") for s in v)
             and any(s.startswith("test") for s in v)}
    print("同时含 train 与 test 的序列 %d" % len(cross))

    train_seqs = [k for k in seqs
                  if any(s.startswith("train") for s in seqs[k])]
    if not train_seqs or not cross:
        print("数据不足, 跳过")
        return

    random.seed(seed)
    keys = random.sample(sorted(cross), min(n_seq, len(cross)))
    cache = {}

    def L(p):
        if p not in cache:
            cache[p] = load(p)
        return cache[p]

    same, diff = [], []
    for k in keys:
        trf = [(s, f) for s in cross[k] if s.startswith("train")
               for f in cross[k][s]]
        tef = [(s, f) for s in cross[k] if s.startswith("test")
               for f in cross[k][s]]
        if not trf or not tef:
            continue
        st, ft = tef[0]
        a = L(os.path.join(BASE, st, ft))
        # 同序列 train 帧 (排除同名)
        m1 = 0.0
        for s, f in trf:
            if f == ft:
                continue
            m1 = max(m1, ncc(a, L(os.path.join(BASE, s, f))))
        same.append(m1)
        # 对照: 其他序列的 train 帧
        m2 = 0.0
        for _ in range(n_ctrl):
            k2 = random.choice(train_seqs)
            if k2 == k:
                continue
            s2 = random.choice([s for s in seqs[k2] if s.startswith("train")])
            f2 = random.choice(seqs[k2][s2])
            p2 = os.path.join(BASE, s2, f2)
            if os.path.exists(p2):
                m2 = max(m2, ncc(a, L(p2)))
        diff.append(m2)

    same = np.array(same)
    diff = np.array(diff)
    print("样本 same=%d diff=%d" % (len(same), len(diff)))
    print()
    print("同序列 (test vs train) 最大 NCC : mean %.3f  median %.3f  max %.3f"
          % (same.mean(), np.median(same), same.max()))
    print("跨序列 对照            最大 NCC : mean %.3f  median %.3f  max %.3f"
          % (diff.mean(), np.median(diff), diff.max()))
    print()
    for t in (0.95, 0.90, 0.80, 0.70):
        print("  阈值 %.2f : 同序列 %2d/%d   跨序列 %2d/%d"
              % (t, (same > t).sum(), len(same), (diff > t).sum(), len(diff)))


if __name__ == "__main__":
    part1_hash_duplicates()
    part2_frame_leakage()
