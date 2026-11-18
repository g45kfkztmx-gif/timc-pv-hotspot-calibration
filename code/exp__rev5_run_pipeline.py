# -*- coding: utf-8 -*-
"""
rev5_run_pipeline.py  (master 顺序流水线, 断点续跑)
======================================================
串行执行 v5 新增实验 (每步落盘 json, 已存在自动跳过, 可反复启动续跑):

  1. S-197 全监督 baseline            (rev5_a_s197, seed0)
  2. A5-adopt@0.6 (校准选择阈值重训)    (见下 build, 复用 A5 refine 流程)
  3. teacher-student SSOD baseline     (rev5_d_teacher_student, 依赖 S-197 权重)
  4. 2x2 因子拆分                      (rev5_c_fact2x2: os / ad / rand)

用法: python rev5_run_pipeline.py [--only 步骤名]
"""
import os
import sys
import argparse

BASE = r"E:\PV DATASET\PV_EIJA\hot_cell_dataset\split"
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rev5_common import result_exists


def step_s197():
    print("\n##### STEP 1: S-197 supervised baseline (seed 0) #####")
    import rev5_a_s197
    rev5_a_s197.main()


def step_a5_tau060():
    print("\n##### STEP 2: A5 with calibration-selected tau_adopt=0.6 #####")
    import rev5_c_fact2x2 as C
    tag = "rev5_a5_tau060"
    if result_exists(tag, 0):
        print("[%s] exists, skip" % tag)
        return
    # 复用 refine 构造逻辑: adopt on, oversample on, conf=0.60 (校准集选择)
    yaml_path = C.build_fact_dataset(tag, do_adopt=True,
                                     do_oversample=True,
                                     conf_adopt=0.60)
    from rev5_common import train_and_eval
    train_and_eval(tag, yaml_path, seed=0,
                   project=os.path.join(BASE, "rev5_runs"))


def step_ts():
    print("\n##### STEP 3: teacher-student baseline #####")
    import rev5_d_teacher_student
    rev5_d_teacher_student.main()


def step_fact2x2():
    print("\n##### STEP 4: 2x2 factorization #####")
    import rev5_c_fact2x2
    rev5_c_fact2x2.main()


STEPS = {"s197": step_s197, "a5tau060": step_a5_tau060,
         "ts": step_ts, "fact2x2": step_fact2x2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="run single step name only")
    ap.add_argument("--from", dest="from_step", default=None,
                    help="start from this step (inclusive)")
    args = ap.parse_args()

    order = ["s197", "a5tau060", "ts", "fact2x2"]

    if args.only:
        STEPS[args.only]()
        return

    start = args.from_step or order[0]
    for name in order:
        if name < start and args.from_step:
            continue
        STEPS[name]()

    print("\n===== pipeline finished =====")


if __name__ == "__main__":
    main()
