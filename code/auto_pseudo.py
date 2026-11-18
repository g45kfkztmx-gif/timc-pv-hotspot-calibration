# -*- coding: utf-8 -*-

"""
PV EIJA 热斑伪标签自动生成系统

适配目录:

split/
├── baseline/
│   ├── hot/
│   ├── hot_labels/
│   └── normal/
│
├── test/
│   ├── hot/
│   ├── hot_labels/
│   └── normal/
│
├── train/
│   ├── hot/
│   └── normal/
│
└── val/
    ├── hot/
    └── normal/

功能:
1. baseline自动寻找最佳检测参数
2. test独立评价
3. train/val生成伪标签
4. normal自动生成空标签
"""


import os
import cv2
import json
import time
import itertools

import numpy as np

from dataclasses import dataclass, asdict



# ==========================================================
# 1. 路径配置
# ==========================================================


BASE_DIR = (
    r"E:\PV DATASET"
    r"\PV_EIJA"
    r"\hot_cell_dataset"
    r"\split"
)


# ---------- baseline ----------

BASELINE_HOT_DIR = os.path.join(
    BASE_DIR,
    "baseline",
    "hot"
)

BASELINE_LABEL_DIR = os.path.join(
    BASE_DIR,
    "baseline",
    "hot_labels"
)



# ---------- test ----------

TEST_HOT_DIR = os.path.join(
    BASE_DIR,
    "test",
    "hot"
)

TEST_LABEL_DIR = os.path.join(
    BASE_DIR,
    "test",
    "hot_labels"
)



# ---------- train ----------

TRAIN_HOT_DIR = os.path.join(
    BASE_DIR,
    "train",
    "hot"
)

TRAIN_NORMAL_DIR = os.path.join(
    BASE_DIR,
    "train",
    "normal"
)



# ---------- val ----------

VAL_HOT_DIR = os.path.join(
    BASE_DIR,
    "val",
    "hot"
)

VAL_NORMAL_DIR = os.path.join(
    BASE_DIR,
    "val",
    "normal"
)



# ---------- 输出 ----------


OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "pseudo_results"
)


PSEUDO_LABEL_DIR = os.path.join(
    OUTPUT_DIR,
    "labels"
)


VIS_DIR = os.path.join(
    OUTPUT_DIR,
    "visualization"
)


PARAM_FILE = os.path.join(
    OUTPUT_DIR,
    "best_params.json"
)


SEARCH_FILE = os.path.join(
    OUTPUT_DIR,
    "search_results.json"
)



for p in [
    OUTPUT_DIR,
    PSEUDO_LABEL_DIR,
    VIS_DIR
]:
    os.makedirs(
        p,
        exist_ok=True
    )



# ==========================================================
# 2. 参数
# ==========================================================


@dataclass
class DetectParams:


    # 灰度异常阈值

    k_sigma: float = 3.0


    # 最小面积比例

    min_area_ratio: float = 0.0015


    # 最大长宽比

    max_aspect: float = 5.0


    # 综合质量阈值

    score_threshold: float = 0.55



    # 框扩展

    expand_ratio: float = 0.05



    # 高斯

    blur_size: int = 5



    # 形态学

    morph_size: int = 3



    # 稳定性

    delta_k: float = 0.25



    # 五维权重


    w_contrast: float = 0.30

    w_peak: float = 0.25

    w_stability: float = 0.20

    w_edge: float = 0.15

    w_fill: float = 0.10




# ==========================================================
# 3. 基础工具
# ==========================================================



IMG_EXT = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
)



def list_images(folder):

    if not os.path.exists(folder):
        return []


    return sorted(
        [
            f
            for f in os.listdir(folder)
            if f.lower().endswith(
                IMG_EXT
            )
        ]
    )





def sigmoid(x):

    return 1 / (
        1 + np.exp(
            -np.clip(
                x,
                -30,
                30
            )
        )
    )




# ==========================================================
# 4. YOLO格式转换
# ==========================================================



def yolo_to_xywh(
        line,
        w,
        h
):

    data = line.strip().split()


    if len(data)<5:
        return None


    xc = float(data[1])
    yc = float(data[2])
    bw = float(data[3])
    bh = float(data[4])



    box_w = int(
        bw*w
    )

    box_h = int(
        bh*h
    )


    x = int(
        xc*w-box_w/2
    )

    y = int(
        yc*h-box_h/2
    )


    return (
        x,
        y,
        box_w,
        box_h
    )





def xywh_to_yolo(
        box,
        w,
        h
):

    x,y,bw,bh = box


    xc = (
        x+bw/2
    )/w


    yc = (
        y+bh/2
    )/h


    return (
        xc,
        yc,
        bw/w,
        bh/h
    )





def load_labels(
        label_file,
        img_w,
        img_h
):

    boxes=[]


    if not os.path.exists(
        label_file
    ):
        return boxes



    with open(
        label_file,
        "r",
        encoding="utf8"
    ) as f:

        for line in f:

            box = yolo_to_xywh(
                line,
                img_w,
                img_h
            )

            if box:

                boxes.append(
                    box
                )


    return boxes





# ==========================================================
# 5. 图像预处理
# ==========================================================



def generate_mask(
        gray,
        k,
        params
):


    blur=cv2.GaussianBlur(
        gray,
        (
            params.blur_size,
            params.blur_size
        ),
        0
    )


    mean=np.mean(
        blur
    )

    std=np.std(
        blur
    )


    threshold=(
        mean+k*std
    )



    mask=(
        blur>threshold
    ).astype(
        np.uint8
    )*255



    kernel=cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            params.morph_size,
            params.morph_size
        )
    )


    mask=cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )


    mask=cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )


    return (
        mask,
        blur,
        threshold
    )





def extract_components(
        mask
):


    num,labels,stats,cent=cv2.connectedComponentsWithStats(
        mask,
        8
    )


    comps=[]


    for i in range(1,num):


        x,y,w,h,area=stats[i]


        comps.append(
            {
                "id":i,
                "box":(
                    int(x),
                    int(y),
                    int(w),
                    int(h)
                ),
                "area":int(area)
            }
        )


    return comps,labels




# ==========================================================
# 6. 框扩展
# ==========================================================


def expand_box(
        box,
        W,
        H,
        ratio
):


    x,y,w,h=box


    dx=int(
        w*ratio
    )

    dy=int(
        h*ratio
    )


    x1=max(
        0,
        x-dx
    )

    y1=max(
        0,
        y-dy
    )


    x2=min(
        W,
        x+w+dx
    )

    y2=min(
        H,
        y+h+dy
    )


    return (
        x1,
        y1,
        x2-x1,
        y2-y1
    )




# ==========================================================
# 7. 热斑候选检测
# ==========================================================


def detect_candidates(
        gray,
        params
):


    H,W=gray.shape



    base_mask,blur,threshold=generate_mask(
        gray,
        params.k_sigma,
        params
    )



    low_mask,_,_=generate_mask(
        gray,
        params.k_sigma-params.delta_k,
        params
    )


    high_mask,_,_=generate_mask(
        gray,
        params.k_sigma+params.delta_k,
        params
    )



    comps,labels=extract_components(
        base_mask
    )



    detections=[]


    min_area=(
        H*W*
        params.min_area_ratio
    )



    for c in comps:


        x,y,w,h=c["box"]


        aspect=max(
            w,h
        )/max(
            1,
            min(w,h)
        )


        if c["area"] < min_area:
            continue



        if aspect > params.max_aspect:
            continue



        roi=gray[
            y:y+h,
            x:x+w
        ]



        if roi.size==0:
            continue



        contrast=(
            np.mean(roi)
            -
            np.mean(gray)
        )


        peak=np.max(
            roi
        )


        score=sigmoid(
            contrast/20
        )


        if score < params.score_threshold:
            continue



        box=expand_box(
            c["box"],
            W,
            H,
            params.expand_ratio
        )


        detections.append(
            {
                "box":box,
                "score":float(score)
            }
        )



    return detections
# ==========================================================
# 8. 五维质量评价
# ==========================================================


def calc_contrast(
        gray,
        box
):

    x,y,w,h=box


    roi=gray[
        y:y+h,
        x:x+w
    ]


    if roi.size==0:
        return 0


    outside=gray.copy()


    mask=np.ones(
        gray.shape,
        dtype=np.uint8
    )


    mask[
        y:y+h,
        x:x+w
    ]=0



    bg=gray[
        mask>0
    ]


    if len(bg)==0:
        return 0


    diff=(
        np.mean(roi)
        -
        np.mean(bg)
    )


    return float(
        sigmoid(
            diff/20
        )
    )





def calc_peak(
        gray,
        box
):

    x,y,w,h=box


    roi=gray[
        y:y+h,
        x:x+w
    ]


    if roi.size==0:
        return 0


    peak=np.max(
        roi
    )


    mean=np.mean(
        gray
    )


    std=np.std(
        gray
    )


    z=(
        peak-mean
    )/(std+1e-6)


    return float(
        sigmoid(
            z-2
        )
    )





def calc_fill(
        component_area,
        box
):

    x,y,w,h=box


    area=w*h


    if area==0:
        return 0


    ratio=(
        component_area/
        area
    )


    return float(
        min(
            ratio/0.8,
            1
        )
    )





def calc_edge(
        gray,
        box
):


    x,y,w,h=box


    roi=gray[
        y:y+h,
        x:x+w
    ]


    if roi.size==0:
        return 0



    edge=cv2.Canny(
        roi,
        50,
        150
    )


    value=np.mean(
        edge>0
    )


    return float(
        min(
            value*3,
            1
        )
    )





def calc_stability(
        gray,
        box,
        params
):


    x,y,w,h=box


    score=[]


    for dk in [
        -params.delta_k,
        0,
        params.delta_k
    ]:


        mask,_,_=generate_mask(
            gray,
            params.k_sigma+dk,
            params
        )


        roi=mask[
            y:y+h,
            x:x+w
        ]


        if roi.size:

            score.append(
                np.mean(
                    roi>0
                )
            )


    if len(score)<2:
        return 0


    return float(
        1-
        np.std(score)
    )





def quality_score(
        gray,
        box,
        area,
        params
):


    c=calc_contrast(
        gray,
        box
    )


    p=calc_peak(
        gray,
        box
    )


    s=calc_stability(
        gray,
        box,
        params
    )


    e=calc_edge(
        gray,
        box
    )


    f=calc_fill(
        area,
        box
    )


    total=(

        params.w_contrast*c

        +

        params.w_peak*p

        +

        params.w_stability*s

        +

        params.w_edge*e

        +

        params.w_fill*f

    )


    return float(total),{


        "contrast":c,

        "peak":p,

        "stability":s,

        "edge":e,

        "fill":f

    }






# ==========================================================
# 9. 带质量评分检测
# ==========================================================


def detect_hotspots(
        gray,
        params
):


    H,W=gray.shape


    mask,blur,thr=generate_mask(
        gray,
        params.k_sigma,
        params
    )



    comps,labels=extract_components(
        mask
    )


    results=[]



    min_area=(
        H*W*
        params.min_area_ratio
    )



    for c in comps:


        box=c["box"]


        x,y,w,h=box



        aspect=max(
            w,h
        )/max(
            1,
            min(w,h)
        )



        if c["area"] < min_area:
            continue


        if aspect > params.max_aspect:
            continue



        score,metrics=quality_score(
            gray,
            box,
            c["area"],
            params
        )



        if score < params.score_threshold:
            continue



        final_box=expand_box(
            box,
            W,
            H,
            params.expand_ratio
        )



        results.append(
            {

            "box":final_box,

            "raw_box":box,

            "score":score,

            "metrics":metrics

            }
        )



    return results





# ==========================================================
# 10. IoU
# ==========================================================


def IoU(
        box1,
        box2
):


    x1,y1,w1,h1=box1

    x2,y2,w2,h2=box2



    xa=max(
        x1,x2
    )

    ya=max(
        y1,y2
    )


    xb=min(
        x1+w1,
        x2+w2
    )

    yb=min(
        y1+h1,
        y2+h2
    )



    inter=max(
        0,
        xb-xa
    )*max(
        0,
        yb-ya
    )



    union=(
        w1*h1+
        w2*h2-
        inter
    )



    if union<=0:
        return 0



    return inter/union





# ==========================================================
# 11. TP FP FN统计
# ==========================================================



def match_result(
        preds,
        gts,
        threshold=0.5
):


    matched=[
        False
        for _ in gts
    ]


    tp=0
    fp=0



    for p in preds:


        best=0

        idx=-1


        for i,g in enumerate(gts):


            if matched[i]:
                continue


            iou=IoU(
                p["box"],
                g
            )


            if iou>best:

                best=iou

                idx=i



        if best>=threshold:

            matched[idx]=True

            tp+=1

        else:

            fp+=1



    fn=sum(
        [
            1
            for x in matched
            if not x
        ]
    )



    return tp,fp,fn





def calculate_metrics(
        tp,
        fp,
        fn
):


    precision=(
        tp/
        max(
            tp+fp,
            1
        )
    )


    recall=(
        tp/
        max(
            tp+fn,
            1
        )
    )


    f1=(

        2*
        precision*
        recall/

        max(
            precision+recall,
            1e-8
        )

    )


    return {

        "precision":precision,

        "recall":recall,

        "f1":f1,

        "tp":tp,

        "fp":fp,

        "fn":fn

    }






# ==========================================================
# 12. 数据集评价
# ==========================================================



def evaluate_dataset(
        img_dir,
        label_dir,
        params
):


    files=list_images(
        img_dir
    )


    TP=0
    FP=0
    FN=0



    for name in files:


        img=cv2.imread(
            os.path.join(
                img_dir,
                name
            ),
            0
        )


        if img is None:
            continue



        h,w=img.shape



        label=os.path.join(
            label_dir,
            os.path.splitext(name)[0]
            +
            ".txt"
        )


        gts=load_labels(
            label,
            w,
            h
        )


        preds=detect_hotspots(
            img,
            params
        )



        tp,fp,fn=match_result(
            preds,
            gts
        )


        TP+=tp

        FP+=fp

        FN+=fn



    result=calculate_metrics(
        TP,
        FP,
        FN
    )



    # 简化AP50近似

    result["ap50"]=(
        result["precision"]
        *
        result["recall"]
    )


    return result






# ==========================================================
# 13. baseline自动搜索
# ==========================================================



def search_parameters():



    best=None

    best_score=-1



    results=[]



    search_space=list(
        itertools.product(

            [
                2.5,
                2.8,
                3.0,
                3.2,
                3.5
            ],

            [
                0.001,
                0.0015,
                0.002
            ],


            [
                3,
                5,
                7
            ],


            [
                0.45,
                0.55,
                0.65
            ]

        )
    )



    print(
        "搜索组合:",
        len(search_space)
    )



    for i,item in enumerate(
        search_space
    ):


        params=DetectParams(

            k_sigma=item[0],

            min_area_ratio=item[1],

            max_aspect=item[2],

            score_threshold=item[3]

        )



        metrics=evaluate_dataset(
            BASELINE_HOT_DIR,
            BASELINE_LABEL_DIR,
            params
        )



        score=(

            0.6*
            metrics["ap50"]

            +

            0.4*
            metrics["f1"]

        )



        record={

            "params":
            asdict(params),

            "metrics":
            metrics,

            "score":
            score

        }


        results.append(
            record
        )



        if score>best_score:


            best_score=score

            best=params



            print(
                "\nBEST UPDATE"
            )

            print(
                json.dumps(
                    record,
                    indent=2
                )
            )



    with open(
        SEARCH_FILE,
        "w",
        encoding="utf8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )



    with open(
        PARAM_FILE,
        "w",
        encoding="utf8"
    ) as f:

        json.dump(
            asdict(best),
            f,
            indent=2
        )



    return best
# ==========================================================
# 14. 读取参数
# ==========================================================


def load_best_params():

    if not os.path.exists(
        PARAM_FILE
    ):

        print(
            "没有找到最佳参数，开始搜索"
        )

        return search_parameters()



    with open(
        PARAM_FILE,
        "r",
        encoding="utf8"
    ) as f:

        data=json.load(f)



    return DetectParams(
        **data
    )





# ==========================================================
# 15. 保存YOLO标签
# ==========================================================



def save_yolo_label(
        save_path,
        detections,
        W,
        H
):


    with open(
        save_path,
        "w",
        encoding="utf8"
    ) as f:



        for d in detections:


            x,y,w,h=d["box"]


            xc,yc,bw,bh=xywh_to_yolo(
                (
                    x,
                    y,
                    w,
                    h
                ),
                W,
                H
            )



            f.write(
                "0 %.6f %.6f %.6f %.6f\n"
                %
                (
                    xc,
                    yc,
                    bw,
                    bh
                )
            )





def save_empty_label(
        path
):

    with open(
        path,
        "w",
        encoding="utf8"
    ) as f:

        pass





# ==========================================================
# 16. 生成train/val伪标签
# ==========================================================



def generate_split_labels(
        hot_dir,
        normal_dir,
        split_name,
        params
):



    save_dir=os.path.join(
        PSEUDO_LABEL_DIR,
        split_name
    )


    os.makedirs(
        save_dir,
        exist_ok=True
    )



    total_hot=0

    total_normal=0



    print(
        "\n开始生成:",
        split_name
    )



    # ------------------------
    # hot
    # ------------------------


    hot_files=list_images(
        hot_dir
    )



    for name in hot_files:


        path=os.path.join(
            hot_dir,
            name
        )


        img=cv2.imread(
            path
        )


        if img is None:
            continue



        gray=cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )



        H,W=gray.shape



        preds=detect_hotspots(
            gray,
            params
        )



        save_path=os.path.join(
            save_dir,
            os.path.splitext(name)[0]
            +
            ".txt"
        )


        save_yolo_label(
            save_path,
            preds,
            W,
            H
        )


        total_hot+=len(preds)



    # ------------------------
    # normal
    # ------------------------


    normal_files=list_images(
        normal_dir
    )



    for name in normal_files:


        save_path=os.path.join(
            save_dir,
            os.path.splitext(name)[0]
            +
            ".txt"
        )


        save_empty_label(
            save_path
        )


        total_normal+=1



    print(
        split_name,
        "完成"
    )


    print(
        "检测热斑数量:",
        total_hot
    )


    print(
        "正常样本:",
        total_normal
    )





# ==========================================================
# 17. 可视化
# ==========================================================



def draw_boxes(
        img,
        boxes,
        color
):


    out=img.copy()


    for b in boxes:


        x,y,w,h=b


        cv2.rectangle(
            out,
            (
                x,
                y
            ),
            (
                x+w,
                y+h
            ),
            color,
            2
        )


    return out





def visualize_dataset(
        img_dir,
        label_dir,
        params,
        name
):


    save=os.path.join(
        VIS_DIR,
        name
    )


    os.makedirs(
        save,
        exist_ok=True
    )


    files=list_images(
        img_dir
    )



    for idx,file in enumerate(
        files
    ):


        path=os.path.join(
            img_dir,
            file
        )


        img=cv2.imread(
            path
        )


        if img is None:
            continue



        gray=cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )



        H,W=gray.shape



        preds=detect_hotspots(
            gray,
            params
        )



        pred_boxes=[
            p["box"]
            for p in preds
        ]



        out=draw_boxes(
            img,
            pred_boxes,
            (
                0,
                0,
                255
            )
        )



        label=os.path.join(
            label_dir,
            os.path.splitext(file)[0]
            +
            ".txt"
        )


        gt=load_labels(
            label,
            W,
            H
        )


        out=draw_boxes(
            out,
            gt,
            (
                0,
                255,
                0
            )
        )



        cv2.putText(
            out,
            "RED:PRED GREEN:GT",
            (
                20,
                40
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (
                255,
                255,
                255
            ),
            2
        )


        cv2.imwrite(
            os.path.join(
                save,
                file
            ),
            out
        )





# ==========================================================
# 18. test最终评价
# ==========================================================


def test_evaluation(
        params
):


    print(
        "\n======================"
    )

    print(
        "TEST evaluation"
    )


    result=evaluate_dataset(
        TEST_HOT_DIR,
        TEST_LABEL_DIR,
        params
    )


    print(
        json.dumps(
            result,
            indent=2
        )
    )



    with open(
        os.path.join(
            OUTPUT_DIR,
            "test_result.json"
        ),
        "w",
        encoding="utf8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )



    return result





# ==========================================================
# 19. main
# ==========================================================


def main():


    start=time.time()



    print(
        "PV 热斑伪标签系统启动"
    )


    # ----------------------
    # 1 参数
    # ----------------------


    params=load_best_params()



    print(
        "\n当前参数:"
    )


    print(
        json.dumps(
            asdict(params),
            indent=2
        )
    )



    # ----------------------
    # 2 baseline评价
    # ----------------------


    print(
        "\nBaseline evaluation"
    )


    baseline_result=evaluate_dataset(
        BASELINE_HOT_DIR,
        BASELINE_LABEL_DIR,
        params
    )


    print(
        json.dumps(
            baseline_result,
            indent=2
        )
    )



    # ----------------------
    # 3 test评价
    # ----------------------


    test_evaluation(
        params
    )



    # ----------------------
    # 4 生成伪标签
    # ----------------------


    generate_split_labels(
        TRAIN_HOT_DIR,
        TRAIN_NORMAL_DIR,
        "train",
        params
    )


    generate_split_labels(
        VAL_HOT_DIR,
        VAL_NORMAL_DIR,
        "val",
        params
    )



    # ----------------------
    # 5 可视化
    # ----------------------


    visualize_dataset(
        BASELINE_HOT_DIR,
        BASELINE_LABEL_DIR,
        params,
        "baseline"
    )



    visualize_dataset(
        TEST_HOT_DIR,
        TEST_LABEL_DIR,
        params,
        "test"
    )



    print(
        "\n全部完成"
    )


    print(
        "输出目录:"
    )


    print(
        OUTPUT_DIR
    )


    print(
        "耗时:",
        time.time()-start,
        "s"
    )




if __name__=="__main__":

    main()