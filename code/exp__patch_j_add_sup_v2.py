# -*- coding: utf-8 -*-
"""patch_j_add_sup_v2.py —— 完成 rev5_j 的 supervised 臂接入 (v1 因模式写错中止)。"""
import os
import shutil
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "rev5_j_budget_multiseed.py")

REPL = [
    (
        "import rev5_f_budget as FB  # noqa: E402\n",
        "import rev5_f_budget as FB  # noqa: E402\n"
        "import rev5_h_supervised_curve as H  # noqa: E402\n",
    ),
    (
        'ARMS_ALL = ("ours_r1", "ours", "ts")\n',
        'ARMS_ALL = ("ours_r1", "ours", "ts", "sup")\n',
    ),
    (
        'TRAINERS = {"ours_r1": train_ours_r1, "ours": train_ours_full, "ts": train_ts}\n',
        "def train_sup(N, seed):\n"
        "    \"\"\"N 张人工 GT 直接监督 detector (委托 rev5_h, 复用同一数据集构造)。\"\"\"\n"
        "    H.run_sup(N, seed=seed)\n"
        "\n"
        "\n"
        'TRAINERS = {"ours_r1": train_ours_r1, "ours": train_ours_full,\n'
        '            "ts": train_ts, "sup": train_sup}\n',
    ),
    (
        '        elif arm == "sup":\n'
        '            r = None\n'
        '            if s != 0:\n'
        '                runs.append(None)\n'
        '                continue\n'
        '            if N in REUSE_SUP0:\n'
        '                tg, sd = REUSE_SUP0[N]\n'
        '                r = read_result(tg, sd)\n'
        '            if r is None:\n'
        '                r = read_result(tag_sup(N), 0)\n'
        '            runs.append(r)\n',
        '        elif arm == "sup":\n'
        '            r = None\n'
        '            if s == 0 and N in REUSE_SUP0:\n'
        '                tg, sd = REUSE_SUP0[N]\n'
        '                r = read_result(tg, sd)\n'
        '            if r is None:\n'
        '                r = read_result(tag_sup(N), s)\n'
        '            runs.append(r)\n',
    ),
    (
        '    print("%-6s %-24s %-24s %-24s" % ("N", "Ours(R1 only)", "Ours(full)",\n'
        '                                      "Teacher-Student"))\n',
        '    print("%-6s %-22s %-22s %-22s %-22s"\n'
        '          % ("N", "Ours(R1 only)", "Ours(full)", "Teacher-Student",\n'
        '             "Direct-sup"))\n',
    ),
    (
        '        for arm in ("ours_r1", "ours", "ts"):\n'
        '            v = curve[arm][i]\n'
        '            if v is None:\n'
        '                cells.append("pending")\n'
        '            elif v["n_seeds"] > 1:\n'
        '                cells.append("%.4f +/- %.4f (n=%d)"\n'
        '                             % (v["mAP50"], v["std_mAP50"], v["n_seeds"]))\n'
        '            else:\n'
        '                cells.append("%.4f (n=1)" % v["mAP50"])\n'
        '        print("%-6d %-24s %-24s %-24s" % (N, cells[0], cells[1], cells[2]))\n',
        '        for arm in ("ours_r1", "ours", "ts", "sup"):\n'
        '            v = curve[arm][i]\n'
        '            if v is None:\n'
        '                cells.append("pending")\n'
        '            elif v["n_seeds"] > 1:\n'
        '                cells.append("%.4f +/- %.4f (n=%d)"\n'
        '                             % (v["mAP50"], v["std_mAP50"], v["n_seeds"]))\n'
        '            else:\n'
        '                cells.append("%.4f (n=1)" % v["mAP50"])\n'
        '        print("%-6d %-22s %-22s %-22s %-22s"\n'
        '              % (N, cells[0], cells[1], cells[2], cells[3]))\n',
    ),
]


def main():
    with open(TARGET, encoding="utf8") as f:
        src = f.read()

    for i, (old, new) in enumerate(REPL, 1):
        if old not in src:
            raise SystemExit("[patch] ABORT: pattern #%d not found" % i)
        if src.count(old) != 1:
            raise SystemExit("[patch] ABORT: pattern #%d not unique (%d)"
                             % (i, src.count(old)))
        src = src.replace(old, new)
        print("[patch] applied #%d" % i)

    bak = TARGET + ".bak_before_sup_arm_" + time.strftime("%Y%m%d_%H%M%S")
    shutil.copy2(TARGET, bak)
    with open(TARGET, "w", encoding="utf8") as f:
        f.write(src)
    print("[patch] OK ->", TARGET)
    print("[patch] backup ->", bak)


if __name__ == "__main__":
    main()
