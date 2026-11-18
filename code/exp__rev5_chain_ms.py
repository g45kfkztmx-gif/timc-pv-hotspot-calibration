# -*- coding: utf-8 -*-
"""
rev5_chain_ms.py —— 等 core 队列结束后, 自动补 N=20/50 的多 seed 误差棒
====================================================================
背景 (2026-09-10 决策):
  seed-0 三臂曲线显示 direct-supervision 从 N=20 起领先, teacher-student 在
  N>=50 领先约 13 点, 而 ours 自 N=20 起持平于 0.80。N>=100 的差距远超
  sigma≈0.029, 不需要误差棒; 只有 N=20 / N=50 需要, 且必须包含 sup 臂,
  否则"低预算下谁更好"下不了结论。

本脚本只做一件事: 等 rev5_queue2 的 core 阶段收尾 (日志出现 QUEUE2 DONE 且
锁定文件消失), 然后顺序执行

  rev5_j_budget_multiseed.py --seeds 1,2 --n 20,50 --arms ours_r1,ts,sup

单实例锁: rev5_chain_ms.lock。写链式等待器之后不要再手工启动同一命令。
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = r"E:\PV DATASET\.venv\Scripts\python.exe"
LOG = os.path.join(HERE, "rev5_chain_ms.log")
LOCK = os.path.join(HERE, "rev5_chain_ms.lock")
UP_LOG = os.path.join(HERE, "rev5_queue2.log")
UP_LOCK = os.path.join(HERE, "rev5_queue2.lock")
UP_MARK = "QUEUE2 DONE"
MAX_WAIT_H = 4.0

JOB = ["rev5_j_budget_multiseed.py", "--seeds", "1,2",
       "--n", "20,50", "--arms", "ours_r1,ts,sup"]


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf8") as f:
        f.write(line + "\n")


def upstream_done():
    if os.path.exists(UP_LOCK):
        return False
    try:
        with open(UP_LOG, encoding="utf8", errors="ignore") as f:
            return UP_MARK in f.read()
    except OSError:
        return False


def acquire_lock():
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print("[lock] %s exists -> another chain is running, exit." % LOCK)
        return False
    with os.fdopen(fd, "w") as f:
        f.write("pid=%d\n" % os.getpid())
    return True


def release_lock():
    try:
        os.remove(LOCK)
    except OSError:
        pass


def main():
    if not acquire_lock():
        return
    try:
        log("=" * 70)
        log("CHAIN START | waiting for upstream core queue")
        log("next job: %s" % " ".join(JOB))
        log("=" * 70)

        t0 = time.time()
        while not upstream_done():
            if (time.time() - t0) / 3600.0 > MAX_WAIT_H:
                log("TIMEOUT waiting for upstream; abort.")
                return
            time.sleep(30)
        log("upstream finished (%.1f min) -> starting multiseed"
            % ((time.time() - t0) / 60.0))

        t1 = time.time()
        with open(LOG, "a", encoding="utf8") as fl:
            rc = subprocess.call([PY] + JOB, cwd=HERE,
                                 stdout=fl, stderr=subprocess.STDOUT)
        log("JOB END: rc=%d | %.1f min" % (rc, (time.time() - t1) / 60.0))
        log("CHAIN DONE")
    finally:
        release_lock()


if __name__ == "__main__":
    main()
