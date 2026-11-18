# -*- coding: utf-8 -*-
"""rev6c2_win_fine.py -- 局部窗口 (win) 的细扫, 判断 31 是否是平台中心。"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import rev6c_param_sensitivity as R  # noqa: E402
import rev5_f_budget as FB  # noqa: E402

WINS = [19, 23, 25, 27, 29, 31, 33, 35, 37, 41, 51, 61]
F = R.FINAL


def main():
    proto = FB.base_params()
    cache = R.build_cache(None)
    rows = []
    for w in WINS:
        per = R.detect_for(cache, F["k"], F["k_local"], w, F["lmd"], F["lbg"])
        TP, FP, FN, p, r, f1, f2 = FB.eval_cached(per, proto, F["g"], F["bmf"], 0.0)
        rows.append({"win": w, "F2": round(f2, 4), "P": round(p, 4),
                     "R": round(r, 4), "FP_per_image": round(FP / len(cache), 3)})
        print("win=%-3d F2=%.4f P=%.3f R=%.3f FP/img=%.3f"
              % (w, f2, p, r, FP / len(cache)), flush=True)
    f2s = [x["F2"] for x in rows]
    print("range = %.4f" % (max(f2s) - min(f2s)))
    with open(os.path.join(_HERE, "rev6c2_win_fine.json"), "w",
              encoding="utf8") as f:
        json.dump({"wins": WINS, "rows": rows,
                   "f2_range": round(max(f2s) - min(f2s), 4)}, f, indent=2)


if __name__ == "__main__":
    main()
