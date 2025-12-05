import os, gc, yaml, glob, pickle
import time
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pandas as pd
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf
from types import SimpleNamespace

import warnings
# warnings.filterwarnings('ignore')

# original

import sys
sys.path.append("../utils")
from data import sep, show_df, glob_walk, set_seed, save_config_yaml, dict_to_namespace

from datetime import datetime
date = datetime.now().strftime("%Y%m%d")
print(f"TODAY is {date}")

# ===================================
# utils
# ===================================
import json

def build_video_meta_and_behavior_pairs(train_df: pl.DataFrame, config):
    # ラベル付き動画だけ
    train_labeled = (
        train_df
        .filter(pl.col("behaviors_labeled").is_not_null())
        .select(
            "lab_id",
            "video_id",
            "body_parts_tracked",
            "behaviors_labeled",
            "frames_per_second",
            "video_duration_sec",
        )
    )

    # 動画メタ情報 (body_parts_tracked / behaviors_labeled を list 化)
    video_meta = (
        train_labeled
            .with_columns([
                pl.col("body_parts_tracked")
                .str.json_decode()
                .alias("body_parts_tracked_list"),
                pl.col("behaviors_labeled")
                .str.json_decode()
                .alias("behaviors_labeled_list"),
            ])
    )

    # ↓ここは前に直した behavior_pairs の処理（ざっくり）
    behavior_pairs_raw = (
        video_meta
        .select(
            "lab_id",
            "video_id",
            "behaviors_labeled_list",
        )
        .explode("behaviors_labeled_list")
        .rename({"behaviors_labeled_list": "behaviors_labeled_element"})
        .with_columns(
            pl.col("behaviors_labeled_element")
              .str.replace_all("'", "")
              .str.replace_all('"', "")
              .alias("behaviors_clean")
        )
        .with_columns(
            pl.col("behaviors_clean").str.split(",").alias("parts")
        )
        .select(
            "lab_id",
            "video_id",
            pl.col("parts").list.get(0).alias("agent_str"),
            pl.col("parts").list.get(1).alias("target_str"),
            pl.col("parts").list.get(2).alias("action"),
        )
    )

    def parse_mouse_id(s):
        if s is None:
            return None
        s = str(s).strip().replace("'", "").replace('"', "")
        if s.lower() == "self":
            return None
        s = s.replace("mouse", "")
        digits = "".join(ch for ch in s if ch.isdigit())
        return int(digits) if digits else None

    behavior_pairs = (
        behavior_pairs_raw
        .with_columns([
            pl.col("agent_str")
              .map_elements(parse_mouse_id, return_dtype=pl.Int8)
              .alias("agent_mouse_id"),
            pl.col("target_str")
              .map_elements(parse_mouse_id, return_dtype=pl.Int8)
              .alias("target_mouse_raw_id"),
        ])
        .with_columns(
            pl.when(pl.col("target_mouse_raw_id").is_null())
              .then(pl.col("agent_mouse_id"))
              .otherwise(pl.col("target_mouse_raw_id"))
              .alias("target_mouse_id")
        )
        .drop("target_mouse_raw_id")
        .with_columns(
            pl.when(pl.col("action").is_in(config.SELF_BEHAVIORS))
              .then(pl.lit("self"))
              .otherwise(pl.lit("pair"))
              .alias("behavior_type")
        )
        .select(
            "lab_id",
            "video_id",
            "agent_mouse_id",
            "target_mouse_id",
            "action",
            "behavior_type",
        )
        .unique()
    )
    return video_meta, behavior_pairs

def load_all_annotations(
    annotation_paths: list[Path],
    train_dataframe: pl.DataFrame,
) -> pl.DataFrame:
    meta_small = train_dataframe.select("video_id", "lab_id", "behaviors_labeled")

    dfs = []
    for p in tqdm(annotation_paths, desc="Loading annotations"):
        df = pl.read_parquet(p)
        # video_id が parquet に無ければファイル名から補完
        if "video_id" not in df.columns:
            vid = int(p.stem)
            df = df.with_columns(pl.lit(vid).alias("video_id"))

        df = df.select(
            "video_id",
            "agent_id",
            "target_id",
            "action",
            "start_frame",
            "stop_frame",
        )

        dfs.append(df)

    annot = pl.concat(dfs, how="vertical_relaxed")

    # agent_id, target_id は既に 1..4 の int で入っている想定
    annot = annot.with_columns(
        pl.col("agent_id").cast(pl.Int8).alias("agent_mouse_id"),
        pl.col("target_id").cast(pl.Int8).alias("target_mouse_id"),
    ).drop(["agent_id", "target_id"])

    # lab_id と behaviors_labeled を付けて metric と同じ形に近づける
    annot = (
        annot.join(meta_small, on="video_id", how="left")
        .rename({"behaviors_labeled": "behaviors_labeled_json"})
    )

    # 念のため start <= stop をチェック
    assert (annot["start_frame"] <= annot["stop_frame"]).all()
    return annot  # これが「イベントラベル」

def load_and_pivot_tracking(
    video_id: int,
    train_meta: pl.DataFrame,   # video_meta から該当行を引く
    video2tracking: dict,       # video_id -> tracking path
    config,
) -> pl.DataFrame:
    """1動画の tracking を (video_frame, mouse_id) index に pivot する"""
    # video_id から tracking path を取得
    path = video2tracking[video_id]
    df = pl.read_parquet(path)  # columns: video_frame, mouse_id, bodypart, x, y, [likelihood]
    df = df.select(["video_frame", "mouse_id", "bodypart", "x", "y"])
    df = df.with_columns(
        pl.col("video_frame").cast(pl.Int32),
        pl.col("mouse_id").cast(pl.Int8),
    )
    # body_parts_tracked から、この動画で本当にトラッキングされている bodypart を取得
    body_parts = (
        train_meta
        .filter(pl.col("video_id") == video_id)
        ["body_parts_tracked_list"].item()
    )
    use_parts = [bp for bp in config.BODY_PARTS if bp in body_parts]

    df = df.filter(pl.col("bodypart").is_in(use_parts))

    # ここで x, y の欠損処理（NaN や 0,0 を None にするなど）を入れても良い


    # pivot
    df_pivot = (
        df.pivot(
            index=["video_frame", "mouse_id"],
            columns="bodypart",
            values=["x", "y"],
        )
        .sort(["video_frame", "mouse_id"])
    )

    # ない bodypart 列を Null で追加して列そろえ
    for bp in config.BODY_PARTS:
        for prefix in ("x_", "y_"):
            col = f"{prefix}{bp}"
            if col not in df_pivot.columns:
                df_pivot = df_pivot.with_columns(
                    pl.lit(None, dtype=pl.Float32).alias(col)
                )

    # ここで補間や smoothing を入れるなら:
    # position_cols = [c for c in df_pivot.columns if c.startswith("x_") or c.startswith("y_")]
    # df_pivot = (
    #     df_pivot
    #     .groupby("mouse_id", maintain_order=True)
    #     .map_groups(lambda g: g.sort("video_frame").with_columns(
    #         [pl.col(c).interpolate().alias(c) for c in position_cols]
    #     ))
    # )
    return df_pivot

def build_video_path_dict(paths: list[Path]) -> dict[int, Path]:
    video2path = {}
    for p in paths:
        # ファイル名 = "44566106.parquet" という前提
        vid = int(p.stem)
        video2path[vid] = p
    return video2path

def rename_agent(col: str) -> str:
    if col in ("video_frame", "mouse_id"):
        return col
    if col.startswith("x_"):
        return "x_agent_" + col[2:]
    if col.startswith("y_"):
        return "y_agent_" + col[2:]
    return col

def rename_target(col: str) -> str:
    if col in ("video_frame", "mouse_id"):
        return col
    if col.startswith("x_"):
        return "x_target_" + col[2:]
    if col.startswith("y_"):
        return "y_target_" + col[2:]
    return col

def make_pair_features_single(
    df_pivot: pl.DataFrame,
    agent_id: int,
    target_id: int,
) -> pl.DataFrame:
    agent = (
        df_pivot
        .filter(pl.col("mouse_id") == agent_id)
        .rename(rename_agent)
    )
    target = (
        df_pivot
        .filter(pl.col("mouse_id") == target_id)
        .rename(rename_target)
    )
    pair = (
        agent.drop("mouse_id")
        .join(target.drop("mouse_id"), on="video_frame", how="inner")
        .sort("video_frame")
    )

    # 例として body_center の相対位置・距離を入れている
    pair = pair.with_columns(
        (pl.col("x_target_body_center") - pl.col("x_agent_body_center")).alias("dx_body_center"),
        (pl.col("y_target_body_center") - pl.col("y_agent_body_center")).alias("dy_body_center"),
    ).with_columns(
        (pl.col("dx_body_center") ** 2 + pl.col("dy_body_center") ** 2)
        .sqrt()
        .alias("dist_body_center"),
    )

    return pair  # video_frame, x_agent_*, y_agent_*, x_target_*, y_target_*, dx, dy, dist, ...

def build_video_pair_features(
    video_id: int,
    df_pivot: pl.DataFrame,
    behavior_pairs: pl.DataFrame,  # behavior_pairs 全体
) -> pl.DataFrame:
    # この動画で評価対象の (agent, target) ペア一覧, pl.DataFrame
    pair_list = (
        behavior_pairs
        .filter(pl.col("video_id") == video_id)
        .select("agent_mouse_id", "target_mouse_id")
        .unique()
    )

    # 各ペアごとに特徴量を作成して結合 
    pair_features_list = []
    for row in pair_list.to_dicts(): # to_dicts -> [{'agent_mouse_id': 1, 'target_mouse_id': 2}, ...]
        a = int(row["agent_mouse_id"])
        t = int(row["target_mouse_id"])
        # make_pair_features_singleを呼び出し、ペア特徴量を取得
        pair = make_pair_features_single(df_pivot, agent_id=a, target_id=t)
        pair = pair.with_columns(
            pl.lit(video_id).alias("video_id"), # video_id 列を追加
            pl.lit(a).alias("agent_mouse_id"),
            pl.lit(t).alias("target_mouse_id"),
        )
        pair_features_list.append(pair)
    if not pair_features_list:
        return pl.DataFrame()  # ラベルだけあって tracking が無い、などは基本ないはず
    # concat
    pair_features = pl.concat(pair_features_list)



    # 最後に INDEX_COLS を先頭にそろえておく
    # pair_features = pair_features.select(
    #     [
    #         "video_id",
    #         "agent_mouse_id",
    #         "target_mouse_id",
    #         "video_frame",
    #         *[c for c in pair_features.columns
    #           if c not in ("video_id", "agent_mouse_id", "target_mouse_id", "video_frame")],
    #     ]
    # )
    # index カラム
    index_cols = ["video_id", "agent_mouse_id", "target_mouse_id", "video_frame"]
    # それ以外をアルファベット順で固定（x_agent_*, y_agent_* ...）
    other_cols = sorted(c for c in pair_features.columns if c not in index_cols)
    pair_features = pair_features.select(index_cols + other_cols)

    pair_features = pair_features.with_columns(
        pl.col("video_id").cast(pl.Int32),
        pl.col("agent_mouse_id").cast(pl.Int8),
        pl.col("target_mouse_id").cast(pl.Int8),
        pl.col("video_frame").cast(pl.Int32),
    )
    # 欠損マスク列を足してから 0 埋め
    float_cols = [
        c for c, dt in pair_features.schema.items()
        if dt == pl.Float32
    ]
    mask_cols = []
    for c in float_cols:
        m = f"{c}_is_null"
        pair_features = pair_features.with_columns(
            pl.col(c).is_null().cast(pl.Int8).alias(m)
        )
        mask_cols.append(m)
    pair_features = pair_features.with_columns(
        [pl.col(c).fill_null(0.0) for c in float_cols]
    )

    return pair_features

def build_frame_labels_for_video(
    video_id: int,
    pair_features_video: pl.DataFrame,
    events: pl.DataFrame,           # annot のうち該当 video_id だけ
    behavior_pairs: pl.DataFrame,   # behavior_pairs のうち該当 video_id だけ
) -> pl.DataFrame:
    df = pair_features_video.select(
        "video_id", "agent_mouse_id", "target_mouse_id", "video_frame"
    ).with_columns(
        pl.col("video_id").cast(pl.Int32),
        pl.col("agent_mouse_id").cast(pl.Int8),
        pl.col("target_mouse_id").cast(pl.Int8),
        pl.col("video_frame").cast(pl.Int32),
    )

    # この動画で評価される (agent, target, action) 一覧
    active = (
        behavior_pairs
        .filter(pl.col("video_id") == video_id)
        .select("agent_mouse_id", "target_mouse_id", "action")
        .unique()
    )

    # action ごとに列を作る（すべて 0 で初期化）
    actions = active["action"].unique().to_list()
    for act in actions:
        df = df.with_columns(pl.lit(0, dtype=pl.Int8).alias(f"label_{act}"))

    # イベントごとに対応するフレームに 1 を立てる
    # （実装は pandas に落として for-loop でやる方が書きやすいかも）
    for row in events.filter(pl.col("video_id") == video_id).to_dicts():
        a = int(row["agent_mouse_id"])
        t = int(row["target_mouse_id"])
        act = row["action"]
        s = int(row["start_frame"])
        e = int(row["stop_frame"])

        colname = f"label_{act}"

        df = df.with_columns(
            pl.when(
                (pl.col("agent_mouse_id") == a)
                & (pl.col("target_mouse_id") == t)
                & (pl.col("video_frame") >= s)
                & (pl.col("video_frame") < e)
            )
            .then(1)
            .otherwise(pl.col(colname))
            .alias(colname)
        )

    return df

# ===================================
# main
# ===================================
# TODO: config_pathをこのスクリプトからの相対パスにする
@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """description
    Args:
        cfg (DictConf): config
    """
    # set config
    config_dict = OmegaConf.to_container(cfg["000_preprocess"], resolve=True)
    config = dict_to_namespace(config_dict)
    # when debug
    if config.debug:
        config.exp = "000_preprocess_debug" # TODO: ファイルの連番を入れる
    # make savedir
    savedir = Path(config.OUTPUT_DIR) / config.exp
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(savedir / "yaml", exist_ok=True)
    # YAMLとして保存
    output_path = Path(savedir/"yaml"/"config.yaml")
    save_config_yaml(config, output_path)
    print(f"Config saved to {output_path.resolve()}")

    # ===============
    # main
    # ===============
    # 1. メタデータ読込
    train_df = pl.read_csv(Path(config.INPUT_DIR) / "train.csv")
    test_df  = pl.read_csv(Path(config.INPUT_DIR) / "test.csv")

    # 2. video_meta と behavior_pairs を作る
    video_meta, behavior_pairs = build_video_meta_and_behavior_pairs(train_df, config)

    # 3. annotation 読み込み
    tracking_paths = glob_walk(config.TRAIN_TRACKING_DIR, "**/*.parquet")
    video2tracking = build_video_path_dict(tracking_paths) # 全動画分のtrackingパスを辞書化
    annotation_paths = glob_walk(Path(config.TRAIN_ANNOTATION_DIR), "**/*.parquet")
    events = load_all_annotations(annotation_paths, train_df) # 全動画分のannotationを結合

    # 4. ラベル付き動画だけループ
    labeled_video_ids = behavior_pairs["video_id"].unique().to_list()

    all_pair_feats = []
    all_frame_labels = []

    for vid in tqdm(labeled_video_ids):
        # tracking を pivot
        df_pivot = load_and_pivot_tracking(vid, video_meta, video2tracking, config)  # train_df じゃなく video_meta 渡す
        # ペア特徴
        pair_feats_vid = build_video_pair_features(vid, df_pivot, behavior_pairs)
        # ラベルイベント（この動画だけ）
        events_vid = events.filter(pl.col("video_id") == vid)
        # フレームラベル
        frame_labels_vid = build_frame_labels_for_video(
            vid,
            pair_feats_vid.select(config.INDEX_COLS),
            events_vid,
            behavior_pairs,
        )
        all_pair_feats.append(pair_feats_vid)
        all_frame_labels.append(frame_labels_vid)

    # concat pair features 
    pair_features = pl.concat(all_pair_feats)
    # concat frame labels
    all_label_cols = sorted(
        {c for df in all_frame_labels for c in df.columns if c.startswith("label_")}
    ) # まず全体で出てくる label 列名を集める
    unified_frame_labels = []
    for df in all_frame_labels:
        # この df に存在しない label 列を 0 埋めで追加
        missing = [c for c in all_label_cols if c not in df.columns]
        if missing:
            df = df.with_columns(
                [pl.lit(0, dtype=pl.Int8).alias(c) for c in missing]
            )
        # 列順を固定（index 4列 + 全 label 列）
        df = df.select(
            ["video_id", "agent_mouse_id", "target_mouse_id", "video_frame"] + all_label_cols
        )
        unified_frame_labels.append(df)
    # これで全 df が同じ幅・同じ列名・同じ順番になる
    frame_labels = pl.concat(unified_frame_labels)

    pair_features.write_parquet(savedir / "train_pair_features.parquet")
    frame_labels.write_parquet(savedir / "train_frame_labels.parquet")
    pair_features.head(1000).write_csv(savedir / "train_pair_features.csv")
    frame_labels.head(1000).write_csv(savedir / "train_frame_labels.csv")
    sep("pair_features"); show_df(pair_features, "pair_features"); print()
    sep("frame_labels"); show_df(frame_labels, "frame_labels")
    print(f"Preprocessed data saved to {savedir.resolve()}")

if __name__ == "__main__":
    main()
