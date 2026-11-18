# -*- coding: utf-8 -*-
"""
patch_final_ops.py
==================
为"最终操作点 g=0.42 / Delta_min=4.0(先验) / tau 冗余显式化"做准备:

  (1) gen_weak_dataset.py 增加 CLI 覆盖 (--tag/--tau/--local-min-delta/
      --score-threshold/--overwrite), 默认行为完全不变;
      并把 effective gate = max(score_threshold, tau) 显式打印出来。
  (2) run_ablation.py 的 build_refined_dataset 增加 base_lbl_dir / r1_weights
      两个可选参数 (默认保持 yolo_weak + weak_r1)。

两个文件都先 dry-run 校验全部 pattern 唯一, 再写盘 + 时间戳备份。
"""
import os
import shutil
import sys
import time

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"

GEN = os.path.join(BASE, "gen_weak_dataset.py")
ABL = os.path.join(BASE, "run_ablation.py")

GEN_REPL = []

GEN_REPL.append((
    "def main():\n"
    "\n"
    "    params = DiagParams(**FROZEN_PARAMS)\n"
    "\n"
    '    root = os.path.join(BASE, "yolo_weak")\n'
    "\n"
    "    if os.path.exists(root):\n"
    "        shutil.rmtree(root)\n",
    "def parse_args(argv):\n"
    '    """CLI 覆盖项。默认值 = 论文冻结操作点, 不传参时行为与原来完全一致。"""\n'
    "    opt = {\"tag\": \"yolo_weak\", \"tau\": TAU_QUALITY,\n"
    '           "local_min_delta": FROZEN_PARAMS["local_min_delta"],\n'
    '           "score_threshold": FROZEN_PARAMS["score_threshold"],\n'
    '           "overwrite": False}\n'
    "    i = 0\n"
    "    while i < len(argv):\n"
    "        a = argv[i]\n"
    '        if a == "--tag" and i + 1 < len(argv):\n'
    '            opt["tag"] = argv[i + 1]\n'
    "            i += 2\n"
    '        elif a == "--tau" and i + 1 < len(argv):\n'
    '            opt["tau"] = float(argv[i + 1])\n'
    "            i += 2\n"
    '        elif a == "--local-min-delta" and i + 1 < len(argv):\n'
    '            opt["local_min_delta"] = float(argv[i + 1])\n'
    "            i += 2\n"
    '        elif a == "--score-threshold" and i + 1 < len(argv):\n'
    '            opt["score_threshold"] = float(argv[i + 1])\n'
    "            i += 2\n"
    '        elif a == "--overwrite":\n'
    '            opt["overwrite"] = True\n'
    "            i += 1\n"
    "        else:\n"
    "            i += 1\n"
    "    return opt\n"
    "\n"
    "\n"
    "def main(argv=None):\n"
    "\n"
    '    opt = parse_args(argv if argv is not None else sys.argv[1:])\n'
    "\n"
    "    # 有效门限 g = max(score_threshold, tau): 候选接受由 score_threshold 控制,\n"
    "    # 写标签再由 tau 过滤, 二者取大。必须显式打印, 避免冗余参数再次掩盖真实门限。\n"
    "    pd = dict(FROZEN_PARAMS)\n"
    '    pd["score_threshold"] = opt["score_threshold"]\n'
    '    pd["local_min_delta"] = opt["local_min_delta"]\n'
    '    tau = opt["tau"]\n'
    '    print("[opt] tag=%s tau=%.2f st=%.2f -> effective_gate=%.2f "\n'
    '          "local_min_delta=%.2f"\n'
    '          % (opt["tag"], tau, pd["score_threshold"],\n'
    '             max(pd["score_threshold"], tau), pd["local_min_delta"]),\n'
    "          flush=True)\n"
    "\n"
    "    params = DiagParams(**pd)\n"
    "\n"
    '    root = os.path.join(BASE, opt["tag"])\n'
    "\n"
    "    if os.path.exists(root):\n"
    '        if not opt["overwrite"]:\n'
    '            raise SystemExit("[abort] %s 已存在; 需要重建请显式加 --overwrite"\n'
    "                             % root)\n"
    "        shutil.rmtree(root)\n",
))

GEN_REPL.append((
    "            n = write_labels(\n"
    '                entries, os.path.join(lbl_dir, stem + ".txt"),\n'
    "                TAU_QUALITY\n"
    "            )\n",
    "            n = write_labels(\n"
    '                entries, os.path.join(lbl_dir, stem + ".txt"),\n'
    "                tau\n"
    "            )\n",
))

GEN_REPL.append((
    '                "frozen_params": FROZEN_PARAMS,\n'
    '                "tau_quality": TAU_QUALITY,\n',
    '                "frozen_params": pd,\n'
    '                "tau_quality": tau,\n'
    '                "effective_gate": max(pd["score_threshold"], tau),\n',
))

ABL_REPL = []

ABL_REPL.append((
    "def build_refined_dataset(tag, conf_adopt):\n"
    '    """基于 yolo_weak + R1模型, 难样本挖掘+自训练(adopt阈值可调)"""\n',
    "def build_refined_dataset(tag, conf_adopt, base_lbl_dir=None, r1_weights=None):\n"
    '    """基于 base 数据集 + R1 模型, 难样本挖掘 + 自训练 (adopt 阈值可调)。\n'
    "\n"
    "    base_lbl_dir / r1_weights 可覆盖; 默认保持论文操作点 (yolo_weak + weak_r1)。\n"
    '    """\n',
))

ABL_REPL.append((
    "    print(\"[gen] loading R1 model for refine ...\")\n"
    "    model = YOLO(R1_WEIGHTS)\n",
    "    print(\"[gen] loading R1 model for refine ...\")\n"
    "    model = YOLO(r1_weights or R1_WEIGHTS)\n",
))

ABL_REPL.append((
    '        src_img_dir = os.path.join(BASE, split, "hot")\n'
    '        src_lbl_dir = os.path.join(BASE, "yolo_weak", "labels", split)\n',
    '        src_img_dir = os.path.join(BASE, split, "hot")\n'
    '        lbl_root = base_lbl_dir or os.path.join(BASE, "yolo_weak", "labels")\n'
    "        src_lbl_dir = os.path.join(lbl_root, split)\n",
))


def run(path, repl, label, write):
    with open(path, encoding="utf8") as f:
        src = f.read()
    for i, (old, new) in enumerate(repl, 1):
        if old not in src:
            raise SystemExit("[%s] ABORT: pattern #%d not found" % (label, i))
        if src.count(old) != 1:
            raise SystemExit("[%s] ABORT: pattern #%d not unique (%d)"
                             % (label, i, src.count(old)))
        src = src.replace(old, new)
        print("[%s] pattern #%d OK" % (label, i))
    if not write:
        return src
    import ast
    ast.parse(src)
    ts = time.strftime("%Y%m%d_%H%M%S")
    shutil.copy2(path, path + ".bak_finalops_" + ts)
    with open(path, "w", encoding="utf8") as f:
        f.write(src)
    print("[%s] written (backup .bak_finalops_%s)" % (label, ts))
    return src


if __name__ == "__main__":
    write = "--write" in sys.argv
    run(GEN, GEN_REPL, "gen_weak", write)
    run(ABL, ABL_REPL, "run_ablation", write)
    print("[done] write=%s" % write)
