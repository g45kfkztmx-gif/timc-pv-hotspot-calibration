# -*- coding: utf-8 -*-
"""
rev5_queue2.py  —— v5/TIM 审稿清单驱动的单实例训练队列
=========================================================
设计要点 (踩坑教训):
  * 单实例锁: 只有本脚本能拉起训练, 杜绝"链式等待器 + 手工启动"重复进程。
  * 每个 job 用 subprocess 顺序执行, 输出追加到同一个日志, 便于 tail。
  * 任一 job 失败只记录, 不中断后续 (上一版 waiter 因判据错误跳到下一个阶段)。

用法:
  python rev5_queue2.py <phase> [...]
    phase = core    : 预算曲线 seed0 三臂 + 直接监督 seed0   (~3.5h)
            multiseed: 预算曲线补 seed (3 seeds; N=197 补到 5) (~7h)
            abl       : A1/A2 5-seed + 2x2 因子表              (~3h)
            all       : core + multiseed + abl
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = r"E:\PV DATASET\.venv\Scripts\python.exe"
LOG = os.path.join(HERE, "rev5_queue2.log")
LOCK = os.path.join(HERE, "rev5_queue2.lock")

PHASES = {
    "core": [
        ("budget curve | seed 0 | ours + ts",
         ["rev5_f_budget.py", "--seeds", "0"]),
        ("direct supervision | seed 0",
         ["rev5_h_supervised_curve.py", "--seeds", "0"]),
    ],
    "multiseed": [
        ("budget curve | seeds 1,2 (all N)",
         ["rev5_f_budget.py", "--seeds", "1,2"]),
        ("budget curve | seeds 3,4 (N=197 only)",
         ["rev5_f_budget.py", "--n", "197", "--seeds", "3,4"]),
        ("direct supervision | seeds 1,2",
         ["rev5_h_supervised_curve.py", "--seeds", "1,2"]),
        ("direct supervision | seeds 3,4 (N=197 only)",
         ["rev5_h_supervised_curve.py", "--n", "197", "--seeds", "3,4"]),
    ],
    "abl": [
        ("A1 5-seed", ["rev5_e_5seed.py", "rev5_a1"]),
        ("A2 5-seed", ["rev5_e_5seed.py", "rev5_a2"]),
        ("A3 5-seed", ["rev5_e_5seed.py", "rev5_a3"]),
        ("A5@0.6 5-seed", ["rev5_e_5seed.py", "rev5_a5_tau060"]),
        ("2x2 factorization (seed 0)", ["rev5_c_fact2x2.py"]),
    ],
}


def acquire_lock():
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print("[lock] %s exists -> another queue is running, exit." % LOCK)
        return False
    with os.fdopen(fd, "w") as f:
        f.write("pid=%d\n" % os.getpid())
    return True


def release_lock():
    try:
        os.remove(LOCK)
    except OSError:
        pass


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf8") as f:
        f.write(line + "\n")


def main():
    args = sys.argv[1:]
    if not args:
        args = ["core"]
    wanted = []
    for a in args:
        if a == "all":
            wanted += ["core", "multiseed", "abl"]
        elif a in PHASES:
            wanted.append(a)
        else:
            print("unknown phase:", a)
            return

    if not acquire_lock():
        return
    try:
        log("=" * 70)
        log("QUEUE2 START | phases=%s" % wanted)
        log("=" * 70)
        t_all = time.time()
        for ph in wanted:
            log("PHASE: %s" % ph)
            for title, cmd in PHASES[ph]:
                log("JOB START: %s" % title)
                t0 = time.time()
                with open(LOG, "a", encoding="utf8") as fl:
                    rc = subprocess.call([PY] + cmd, cwd=HERE,
                                         stdout=fl, stderr=subprocess.STDOUT)
                dt = (time.time() - t0) / 60.0
                log("JOB END: %s | rc=%d | %.1f min" % (title, rc, dt))
        log("QUEUE2 DONE | total %.1f min" % ((time.time() - t_all) / 60.0))
    finally:
        release_lock()


if __name__ == "__main__":
    main()
