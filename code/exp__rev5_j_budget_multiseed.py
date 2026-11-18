# -*- coding: utf-8 -*-
"""
rev5_j_budget_multiseed.py
==========================
给标注预算曲线补训练种子, 得到 mean +/- std。

为什么可以很便宜
----------------
预算曲线里唯一与 seed 有关的量是"训练"。校准 (budget_calib_N{N}.json) 与
伪标签数据集 (yolo_<tag>/) 都是确定性的, 与 seed 无关, 因此补 seed 的成本
= 纯训练时间, 且不会引入任何新的数据构造差异 (即: 曲线各点之间的差异只来自
训练随机性, 这正是误差棒要刻画的东西)。

臂与 tag 约定
-------------
  ours_r1 : rev5_bud_ours{N}_r1              所有 seed 共用同一份数据集
  ts      : teacher rev5_bud_ts{N}_teacher   所有 seed 共用同一份数据集
            student rev5_bud_ts{N}_s{SEED}   student 数据集依赖 teacher 权重,
                                            故每个 seed 一份
  ours    : rev5_bud_ours{N}                 (seed 0, 由 rev5_f 产出)
            rev5_bud_ours{N}_ref_s{SEED}     (seed>0, refined 依赖 R1 权重)
  sup     : rev5_sup{N} / rev5_s197          (由 rev5_h 产出, 本脚本只汇总)

汇总时会重写 budget_curve.json, 每点在 mAP50/mAP50_95/P/R 之外附加:
  n_seeds, seeds, std_<key>, runs[]

用法
----
  python rev5_j_budget_multiseed.py --seeds 1,2               # ours_r1 + ts
  python rev5_j_budget_multiseed.py --seeds 1,2 --with-refine # 同时补 refined
  python rev5_j_budget_multiseed.py --seeds 1 --n 20,50
  python rev5_j_budget_multiseed.py --merge-only              # 只汇总, 不训练
"""
import json
import os
import sys

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

from rev5_common import REV5, out_json, result_exists, train_and_eval  # noqa: E402
import rev5_f_budget as FB  # noqa: E402
import rev5_h_supervised_curve as H  # noqa: E402

BUDGETS = list(FB.BUDGETS)
TAU_ADOPT = FB.TAU_ADOPT_FIXED
LOCK_PATH = os.path.join(_HERE, "rev5_j.lock")

# 汇总时扫描的 seed 范围 (训练用 --seeds 指定, 汇总只读)
SCAN_SEEDS = (0, 1, 2, 3, 4)

# seed 0 的复用表 (与 rev5_f / rev5_h 保持一致)
REUSE_TS0 = {(197): ("rev5_ts_student", 0)}
REUSE_SUP0 = {197: ("rev5_s197", 0)}

ARMS_ALL = ("ours_r1", "ours", "ts", "sup")


# ==========================================================
# 单实例锁
# ==========================================================

def acquire_lock():
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print("[lock] %s exists -> another instance is running, exit."
              % LOCK_PATH)
        return False
    with os.fdopen(fd, "w") as f:
        f.write("pid=%d\n" % os.getpid())
    return True


def release_lock():
    try:
        os.remove(LOCK_PATH)
    except OSError:
        pass


# ==========================================================
# tag 约定
# ==========================================================

def tag_ours_r1(N):
    return "rev5_bud_ours%d_r1" % N


def tag_ours_full(N, seed):
    if seed == 0:
        return "rev5_bud_ours%d" % N
    return "rev5_bud_ours%d_ref_s%d" % (N, seed)


def tag_ts_teacher(N):
    return "rev5_bud_ts%d_teacher" % N


def tag_ts_student(N, seed):
    if seed == 0:
        return "rev5_bud_ts%d" % N
    return "rev5_bud_ts%d_s%d" % (N, seed)


def tag_sup(N):
    return "rev5_sup%d" % N


def weights_path(tag, seed):
    return os.path.join(FB.REV5_RUNS, "pseudo_%s_seed%d" % (tag, seed),
                        "weights", "best.pt")


# ==========================================================
# 训练
# ==========================================================

def train_ours_r1(N, seed):
    tag = tag_ours_r1(N)
    if result_exists(tag, seed):
        print("[ours_r1 N=%d seed=%d] exists, skip" % (N, seed), flush=True)
        return
    d = FB.calibrate(N)                       # 缓存命中则秒回
    params = FB.calib_to_params(d)
    yaml_path = FB.build_pseudo_dataset(tag, params, d["tau"])
    train_and_eval(tag, yaml_path, seed=seed, project=FB.REV5_RUNS)


def train_ours_full(N, seed):
    tag = tag_ours_full(N, seed)
    if result_exists(tag, seed):
        print("[ours N=%d seed=%d] exists, skip" % (N, seed), flush=True)
        return
    r1_tag = tag_ours_r1(N)
    if not result_exists(r1_tag, seed):
        train_ours_r1(N, seed)
    r1_w = weights_path(r1_tag, seed)
    if not os.path.exists(r1_w):
        raise RuntimeError("R1 weights missing: %s" % r1_w)
    yaml_path = FB.build_refined_dataset(tag, TAU_ADOPT, r1_w)
    train_and_eval(tag, yaml_path, seed=seed, project=FB.REV5_RUNS)


def train_ts(N, seed):
    fin_tag = tag_ts_student(N, seed)
    if result_exists(fin_tag, seed):
        print("[ts N=%d seed=%d] exists, skip" % (N, seed), flush=True)
        return
    t_tag = tag_ts_teacher(N)
    if not result_exists(t_tag, seed):
        yaml_path = FB.build_teacher_dataset(t_tag, N)
        train_and_eval(t_tag, yaml_path, seed=seed, project=FB.REV5_RUNS)
    t_w = weights_path(t_tag, seed)
    if not os.path.exists(t_w):
        raise RuntimeError("teacher weights missing: %s" % t_w)
    yaml_path = FB.build_student_dataset(fin_tag, N, t_w)
    train_and_eval(fin_tag, yaml_path, seed=seed, project=FB.REV5_RUNS)


def train_sup(N, seed):
    """N 张人工 GT 直接监督 detector (委托 rev5_h, 复用同一数据集构造)。"""
    H.run_sup(N, seed=seed)


TRAINERS = {"ours_r1": train_ours_r1, "ours": train_ours_full,
            "ts": train_ts, "sup": train_sup}


# ==========================================================
# 汇总
# ==========================================================

def read_result(tag, seed):
    p = out_json(tag, seed)
    if not (os.path.exists(p) and os.path.getsize(p) > 10):
        return None
    with open(p, encoding="utf8") as f:
        r = json.load(f)
    return {"seed": seed, "tag": tag,
            "mAP50": r["test_map50"], "mAP50_95": r["test_map50_95"],
            "P": r["test_precision"], "R": r["test_recall"]}


def agg_point(runs):
    runs = [r for r in runs if r]
    if not runs:
        return None
    out = {"tag": runs[0]["tag"], "n_seeds": len(runs),
           "seeds": [r["seed"] for r in runs], "runs": runs}
    for k in ("mAP50", "mAP50_95", "P", "R"):
        vals = [r[k] for r in runs]
        m = sum(vals) / len(vals)
        out[k] = m
        if len(vals) > 1:
            out["std_" + k] = (sum((v - m) ** 2 for v in vals)
                               / (len(vals) - 1)) ** 0.5
        else:
            out["std_" + k] = 0.0
    return out


def gather(N, arm):
    """按 arm 收集所有可用 seed 的结果。"""
    runs = []
    for s in SCAN_SEEDS:
        if arm == "ours_r1":
            runs.append(read_result(tag_ours_r1(N), s))
        elif arm == "ours":
            if N == 197 and s == 0:
                pass                       # ours 臂不复用, 见 rev5_f 注释
            runs.append(read_result(tag_ours_full(N, s), s))
        elif arm == "ts":
            r = None
            if s == 0 and N in REUSE_TS0:
                tg, sd = REUSE_TS0[N]
                r = read_result(tg, sd)
            if r is None:
                r = read_result(tag_ts_student(N, s), s)
            runs.append(r)
        elif arm == "sup":
            r = None
            if s == 0 and N in REUSE_SUP0:
                tg, sd = REUSE_SUP0[N]
                r = read_result(tg, sd)
            if r is None:
                r = read_result(tag_sup(N), s)
            runs.append(r)
    return runs


def collect(budgets=None):
    budgets = list(budgets or BUDGETS)
    curve = {"N": budgets}
    for arm in ("ours", "ours_r1", "ts", "sup"):
        curve[arm] = [agg_point(gather(N, arm)) for N in budgets]

    out = os.path.join(FB.REV5, "budget_curve.json")
    with open(out, "w", encoding="utf8") as f:
        json.dump(curve, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 88)
    print("CALIBRATION-BUDGET CURVE  (test mAP@50 | mean +/- std over seeds)")
    print("=" * 88)
    print("%-6s %-22s %-22s %-22s %-22s"
          % ("N", "Ours(R1 only)", "Ours(full)", "Teacher-Student",
             "Direct-sup"))
    for i, N in enumerate(budgets):
        cells = []
        for arm in ("ours_r1", "ours", "ts", "sup"):
            v = curve[arm][i]
            if v is None:
                cells.append("pending")
            elif v["n_seeds"] > 1:
                cells.append("%.4f +/- %.4f (n=%d)"
                             % (v["mAP50"], v["std_mAP50"], v["n_seeds"]))
            else:
                cells.append("%.4f (n=1)" % v["mAP50"])
        print("%-6d %-22s %-22s %-22s %-22s"
              % (N, cells[0], cells[1], cells[2], cells[3]))

    print("\n%-6s %-24s %-24s" % ("N", "Ours(R1) recall", "TS recall"))
    for i, N in enumerate(budgets):
        cells = []
        for arm in ("ours_r1", "ts"):
            v = curve[arm][i]
            if v is None:
                cells.append("pending")
            elif v["n_seeds"] > 1:
                cells.append("%.4f +/- %.4f" % (v["R"], v["std_R"]))
            else:
                cells.append("%.4f" % v["R"])
        print("%-6d %-24s %-24s" % (N, cells[0], cells[1]))

    print("\nsaved ->", out)
    return curve


# ==========================================================
# main
# ==========================================================

def parse_args(argv):
    opt = {"seeds": [], "budgets": BUDGETS, "arms": ["ours_r1", "ts"],
           "with_refine": False, "merge_only": False}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--seeds" and i + 1 < len(argv):
            opt["seeds"] = [int(x) for x in argv[i + 1].split(",")]
            i += 2
        elif a == "--n" and i + 1 < len(argv):
            opt["budgets"] = [int(x) for x in argv[i + 1].split(",")]
            i += 2
        elif a == "--arms" and i + 1 < len(argv):
            opt["arms"] = [x.strip() for x in argv[i + 1].split(",")]
            i += 2
        elif a == "--with-refine":
            opt["arms"] = ["ours_r1", "ours", "ts"]
            opt["with_refine"] = True
            i += 1
        elif a == "--merge-only":
            opt["merge_only"] = True
            i += 1
        else:
            i += 1
    return opt


def main():
    opt = parse_args(sys.argv[1:])

    if opt["merge_only"]:
        collect(opt["budgets"])
        return

    seeds = opt["seeds"] or [1, 2]
    for a in opt["arms"]:
        if a not in TRAINERS:
            raise SystemExit("unknown arm: %s (choose from %s)"
                             % (a, list(TRAINERS)))

    if not acquire_lock():
        return
    try:
        print("=" * 88)
        print("rev5_j_budget_multiseed | seeds=%s | budgets=%s | arms=%s"
              % (seeds, opt["budgets"], opt["arms"]))
        print("=" * 88, flush=True)

        for N in opt["budgets"]:
            for sd in seeds:
                for arm in opt["arms"]:
                    print("\n########## %s | N=%d | seed=%d ##########"
                          % (arm, N, sd), flush=True)
                    TRAINERS[arm](N, sd)

        collect(opt["budgets"])
    finally:
        release_lock()
    print("\n===== rev5_j_budget_multiseed finished =====", flush=True)


if __name__ == "__main__":
    main()
