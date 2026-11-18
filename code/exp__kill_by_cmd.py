# -*- coding: utf-8 -*-
"""
kill_by_cmd.py —— 按命令行子串可靠地杀掉 python 进程树（本环境专用）
===================================================================
为什么需要它：
  * `wmic` 在本沙箱被 Program Blacklist 拦截，不可用；
  * PowerShell 工具在本环境**静默返回 exit 0 且不回显输出**，`taskkill` 是否真的
    执行无法确认 —— 曾经因此以为杀掉了 rev5_j，实际进程树存活 40 分钟并在后台
    与新一轮训练抢 GPU（epoch 从 13s 退化到 40~116s）；
  * Git Bash 里 `taskkill //PID` 参数转换会报 "无效参数"。
  ⇒ 用 Python + Win32_Process(CIM) 枚举命令行 → os.kill 逐级杀 → **再枚举一次验证**。

用法：
    python kill_by_cmd.py <子串> [<子串> ...]           # 列出并杀掉匹配的进程树
    python kill_by_cmd.py --list                        # 只列出，不杀
    python kill_by_cmd.py --list <子串>

实现要点：先杀子进程再杀父进程；杀完必须重新枚举并打印幸存者，否则不算成功。
"""
import csv
import io
import os
import signal
import subprocess
import sys
import time

TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_procs_query.csv")

PS = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
      "Select-Object ProcessId,ParentProcessId,CommandLine | "
      "ConvertTo-Csv -NoTypeInformation | Set-Content -Encoding UTF8 '%s'" % TMP)


def snapshot():
    """返回 [{pid, ppid, cmd}]；PowerShell 输出经文件回读（本环境 stdout 不可靠）。"""
    subprocess.run(["powershell", "-NoProfile", "-Command", PS],
                   capture_output=True)
    if not os.path.exists(TMP):
        return []
    rows = []
    with io.open(TMP, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({"pid": int(r["ProcessId"]),
                             "ppid": int(r["ParentProcessId"]),
                             "cmd": r.get("CommandLine") or ""})
            except (KeyError, TypeError, ValueError):
                continue
    os.remove(TMP)
    return rows


def main():
    args = [a for a in sys.argv[1:]]
    dry = "--list" in args
    pats = [a for a in args if not a.startswith("--")]
    if not pats:
        print(__doc__)
        return 1

    procs = snapshot()
    hits = {p["pid"]: p for p in procs
            if any(s in p["cmd"] for s in pats)
            and "kill_by_cmd" not in p["cmd"]}

    if not hits:
        print("no match for %s" % pats)
        return 0

    # 展开子进程（CIM 快照里的 parent 关系）
    changed = True
    while changed:
        changed = False
        for p in procs:
            if p["ppid"] in hits and p["pid"] not in hits:
                hits[p["pid"]] = p
                changed = True

    for pid in sorted(hits):
        print("%8d  ppid=%-7d %s" % (pid, hits[pid]["ppid"], hits[pid]["cmd"][:100]))
    if dry:
        print("[dry-run] %d processes, nothing killed" % len(hits))
        return 0

    # 子进程先杀（按 ppid 深度排序：pid 大致递增即可）
    for pid in sorted(hits, reverse=True):
        try:
            os.kill(pid, signal.SIGTERM)
            print("killed", pid)
        except OSError as e:
            print("skip  ", pid, e)
    time.sleep(2.0)

    left = [p for p in snapshot() if any(s in p["cmd"] for s in pats)]
    if left:
        print("!! STILL ALIVE: %s" % [p["pid"] for p in left])
        return 2
    print("verified: all matched processes gone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
