import os, gc, re, yaml, glob, pickle, warnings
import time
import random, math
import joblib, pickle, itertools
from pathlib import Path

import numpy as np
import scipy as sp
import polars as pl
import pandas as pd
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf
from types import SimpleNamespace

from sklearn.model_selection import StratifiedKFold

# original
import sys
sys.path.append(r"..")
from utils.data import sep, show_df, glob_walk, set_seed, format_time, save_config_yaml, dict_to_namespace

from datetime import datetime
date = datetime.now().strftime("%Y%m%d")
print(f"TODAY is {date}")


@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """2nd-stage preprocess: add folds, meta features, normalized coords, and export NumPy per fold."""
    # =========================
    # config & output dir
    # =========================
    config_dict = OmegaConf.to_container(cfg["001_preprocess_step2"], resolve=True)
    config = dict_to_namespace(config_dict)
    if config.debug:
        config.exp = "001_preprocess_step2_debug"

    savedir = Path(config.OUTPUT_DIR) / config.exp
    savedir.mkdir(parents=True, exist_ok=True)
    (savedir / "yaml").mkdir(exist_ok=True)
    output_path = savedir / "yaml" / "config.yaml"
    save_config_yaml(config, output_path)
    print(f"Config saved to {output_path.resolve()}")

    # ==================
    # read raw train.csv
    # ==================
    train_df = pl.read_csv(Path(config.INPUT_DIR) / "train.csv")
    sep("train_df"); show_df(train_df, 3, True)

    # ==================
    # load pair_features & frame_labels from 000_preprocess
    # ==================
    base_dir = Path(config.BASE_PATH)
    pair_features = pl.read_parquet(base_dir / "train_pair_features.parquet")
    frame_labels  = pl.read_parquet(base_dir / "train_frame_labels.parquet")
    sep("pair_features (head)"); show_df(pair_features, 3, True); print()
    sep("frame_labels (head)"); show_df(frame_labels, 3, True)

    # =======================================
    # 1. video-level Stratified KFold (by lab_id)
    # =======================================
    N_SPLITS = config.n_splits
    SEED = config.seed

    train_pdf = train_df.to_pandas()
    df_labeled = train_pdf[train_pdf["behaviors_labeled"].notna()].copy()
    df_labeled = df_labeled[["video_id", "lab_id"]].drop_duplicates()
    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )
    df_labeled["fold"] = -1
    df_labeled = df_labeled.reset_index(drop=True)
    for fold, (_, val_idx) in enumerate(skf.split(df_labeled["video_id"], df_labeled["lab_id"])):
        df_labeled.loc[val_idx, "fold"] = fold
    print("fold counts:\n", df_labeled["fold"].value_counts())

    fold_csv_path = savedir / "video_folds.csv"
    df_labeled.to_csv(fold_csv_path, index=False)

    # Polars で fold 情報読み込み
    fold_df = pl.read_csv(fold_csv_path)  # video_id, lab_id, fold

    # =======================================
    # 2. attach fold to pair_features / frame_labels
    # =======================================
    pair_features = pair_features.join(
        fold_df.select(["video_id", "fold"]), on="video_id", how="left"
    )
    frame_labels = frame_labels.join(
        fold_df.select(["video_id", "fold"]), on="video_id", how="left"
    )

    # =======================================
    # 3. attach numeric meta info
    # =======================================
    meta_numeric = (
        train_df
        .select(
            "video_id",
            "lab_id",
            "frames_per_second",
            "pix_per_cm_approx",
            "video_width_pix",
            "video_height_pix",
            "arena_width_cm",
            "arena_height_cm",
            "arena_shape",
            "arena_type",
            "tracking_method",
        )
    )
    pair_features = pair_features.join(meta_numeric, on="video_id", how="left")

    # =======================================
    # 4. normalize coordinates / distance
    # =======================================
    x_cols = [c for c in pair_features.columns if c.startswith("x_agent_") or c.startswith("x_target_")]
    y_cols = [c for c in pair_features.columns if c.startswith("y_agent_") or c.startswith("y_target_")]

    pair_features = pair_features.with_columns(
        [ (pl.col(c) / pl.col("video_width_pix")).alias(c + "_norm") for c in x_cols ] +
        [ (pl.col(c) / pl.col("video_height_pix")).alias(c + "_norm") for c in y_cols ]
    )
    pair_features = pair_features.with_columns(
        (pl.col("dist_body_center") / pl.col("video_width_pix")).alias("dist_body_center_norm")
    )

    # =======================================
    # 5. encode categorical meta (lab, arena, tracking)
    # =======================================
    meta_pdf = meta_numeric.to_pandas()
    for col in ["lab_id", "arena_type", "arena_shape", "tracking_method"]:
        meta_pdf[col + "_code"] = meta_pdf[col].astype("category").cat.codes

    meta_encoded = pl.from_pandas(meta_pdf).select(
        "video_id",
        "lab_id_code",
        "arena_type_code",
        "arena_shape_code",
        "tracking_method_code",
    )
    pair_features = pair_features.join(meta_encoded, on="video_id", how="left")

    # =======================================
    # 6. prepare feature / label column lists
    # =======================================
    INDEX_COLS = ["video_id", "agent_mouse_id", "target_mouse_id", "video_frame"]

    label_cols = [c for c in frame_labels.columns if c.startswith("label_")]
    mask_cols  = [c for c in frame_labels.columns if c.startswith("mask_")]  # まだ無ければ空

    exclude = set(INDEX_COLS + ["fold", "lab_id", "arena_shape", "arena_type", "tracking_method"])
    feature_cols = [
        c for c in pair_features.columns
        if c not in exclude and not c.endswith("_is_null")
    ]

    print("n_features:", len(feature_cols), "n_labels:", len(label_cols))

    # =======================================
    # 7. export NumPy per fold
    # =======================================
    out_dir = savedir / "fold_numpy"
    out_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(config.n_splits):
        print(f"\n=== fold {fold} ===")

        train_pair = pair_features.filter(pl.col("fold") != fold)
        valid_pair = pair_features.filter(pl.col("fold") == fold)

        train_lbl  = frame_labels.filter(pl.col("fold") != fold)
        valid_lbl  = frame_labels.filter(pl.col("fold") == fold)

        train_all = train_pair.join(
            train_lbl, on=INDEX_COLS + ["fold"], how="inner"
        )
        valid_all = valid_pair.join(
            valid_lbl, on=INDEX_COLS + ["fold"], how="inner"
        )

        X_train = train_all.select(feature_cols).to_numpy().astype("float32")
        y_train = train_all.select(label_cols).to_numpy().astype("float32")
        if mask_cols:
            m_train = train_all.select(mask_cols).to_numpy().astype("float32")
        else:
            m_train = None

        X_valid = valid_all.select(feature_cols).to_numpy().astype("float32")
        y_valid = valid_all.select(label_cols).to_numpy().astype("float32")
        if mask_cols:
            m_valid = valid_all.select(mask_cols).to_numpy().astype("float32")
        else:
            m_valid = None

        np.save(out_dir / f"X_train_fold{fold}.npy", X_train)
        np.save(out_dir / f"y_train_fold{fold}.npy", y_train)
        if m_train is not None:
            np.save(out_dir / f"m_train_fold{fold}.npy", m_train)

        np.save(out_dir / f"X_valid_fold{fold}.npy", X_valid)
        np.save(out_dir / f"y_valid_fold{fold}.npy", y_valid)
        if m_valid is not None:
            np.save(out_dir / f"m_valid_fold{fold}.npy", m_valid)

    # 保存しておくと後で便利（feature/label 名）
    cols_info = {
        "feature_cols": feature_cols,
        "label_cols": label_cols,
        "mask_cols": mask_cols,
    }
    import json
    with open(out_dir / "columns.json", "w") as f:
        json.dump(cols_info, f, indent=2)


if __name__ == "__main__":
    main()