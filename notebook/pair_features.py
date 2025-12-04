
def make_pair_features(
    metadata: dict,
    tracking: pl.DataFrame,
) -> pl.DataFrame:
    def body_parts_distance(agent_or_target_1, body_part_1, agent_or_target_2, body_part_2):
        # agent-targetの体の各部位間距離(cm)
        assert agent_or_target_1 == "agent" or agent_or_target_1 == "target"
        assert agent_or_target_2 == "agent" or agent_or_target_2 == "target"
        assert body_part_1 in BODY_PARTS
        assert body_part_2 in BODY_PARTS
        return (
            (pl.col(f"{agent_or_target_1}_x_{body_part_1}") - pl.col(f"{agent_or_target_2}_x_{body_part_2}")).pow(2)
            + (pl.col(f"{agent_or_target_1}_y_{body_part_1}") - pl.col(f"{agent_or_target_2}_y_{body_part_2}")).pow(2)
        ).sqrt() / metadata["pix_per_cm_approx"]

    def body_part_speed(agent_or_target, body_part, period_ms):
        # 部位の推定速度(cm/s)
        assert agent_or_target == "agent" or agent_or_target == "target"
        assert body_part in BODY_PARTS
        window_frames = max(1, int(round(period_ms * metadata["frames_per_second"] / 1000.0)))
        return (
            (
                (pl.col(f"{agent_or_target}_x_{body_part}").diff()).pow(2)
                + (pl.col(f"{agent_or_target}_y_{body_part}").diff()).pow(2)
            ).sqrt()
            / metadata["pix_per_cm_approx"]
            * metadata["frames_per_second"]
        ).rolling_mean(window_size=window_frames, center=True)

    def elongation(agent_or_target):
        # 伸長度(cm)
        assert agent_or_target == "agent" or agent_or_target == "target"
        d1 = body_parts_distance(agent_or_target, "nose", agent_or_target, "tail_base")
        d2 = body_parts_distance(agent_or_target, "ear_left", agent_or_target, "ear_right")
        return d1 / (d2 + 1e-06)

    def body_angle(agent_or_target):
        # 体角度(deg)
        assert agent_or_target == "agent" or agent_or_target == "target"
        v1x = pl.col(f"{agent_or_target}_x_nose") - pl.col(f"{agent_or_target}_x_body_center")
        v1y = pl.col(f"{agent_or_target}_y_nose") - pl.col(f"{agent_or_target}_y_body_center")
        v2x = pl.col(f"{agent_or_target}_x_tail_base") - pl.col(f"{agent_or_target}_x_body_center")
        v2y = pl.col(f"{agent_or_target}_y_tail_base") - pl.col(f"{agent_or_target}_y_body_center")
        return (v1x * v2x + v1y * v2y) / ((v1x.pow(2) + v1y.pow(2)).sqrt() * (v2x.pow(2) + v2y.pow(2)).sqrt() + 1e-06)

    def body_center_distance_rolling_agg(agg, period_ms):
        # 距離の移動集計特徴量
        assert agg in ["mean", "std", "var", "min", "max"] # 集計関数
        expr = body_parts_distance("agent", "body_center", "target", "body_center")
        window_frames = max(1, int(round(period_ms * metadata["frames_per_second"] / 1000.0)))

        if agg == "mean":
            return expr.rolling_mean(window_size=window_frames, center=True, min_samples=1)
        elif agg == "std":
            return expr.rolling_std(window_size=window_frames, center=True, min_samples=1)
        elif agg == "var":
            return expr.rolling_var(window_size=window_frames, center=True, min_samples=1)
        elif agg == "min":
            return expr.rolling_min(window_size=window_frames, center=True, min_samples=1)
        elif agg == "max":
            return expr.rolling_max(window_size=window_frames, center=True, min_samples=1)
        else:
            raise ValueError()

    n_mice = (
        (metadata["mouse1_strain"] is not None)
        + (metadata["mouse2_strain"] is not None)
        + (metadata["mouse3_strain"] is not None)
        + (metadata["mouse4_strain"] is not None)
    )
    start_frame = tracking.select(pl.col("video_frame").min()).item()
    end_frame = tracking.select(pl.col("video_frame").max()).item()

    result = []

    pivot = tracking.pivot(
        on=["bodypart"],
        index=["video_frame", "mouse_id"],
        values=["x", "y"],
    ).sort(["mouse_id", "video_frame"])
    pivot_trackings = {mouse_id: pivot.filter(pl.col("mouse_id") == mouse_id) for mouse_id in range(1, n_mice + 1)}

    for agent_mouse_id, target_mouse_id in itertools.permutations(range(1, n_mice + 1), 2):
        result_element = pl.DataFrame(
            {
                "video_id": metadata["video_id"],
                "agent_mouse_id": agent_mouse_id,
                "target_mouse_id": target_mouse_id,
                "video_frame": pl.arange(start_frame, end_frame + 1, eager=True),
            },
            schema={
                "video_id": pl.Int32,
                "agent_mouse_id": pl.Int8,
                "target_mouse_id": pl.Int8,
                "video_frame": pl.Int32,
            },
        )

        merged_pivot = (
            pivot_trackings[agent_mouse_id]
            .select(
                pl.col("video_frame"),
                pl.exclude("video_frame").name.prefix("agent_"),
            )
            .join(
                pivot_trackings[target_mouse_id].select(
                    pl.col("video_frame"),
                    pl.exclude("video_frame").name.prefix("target_"),
                ),
                on="video_frame",
                how="inner",
            )
        )
        columns = merged_pivot.columns
        merged_pivot = merged_pivot.with_columns(
            *[pl.lit(None).cast(pl.Float32).alias(f"agent_x_{bp}") for bp in BODY_PARTS if f"agent_x_{bp}" not in columns],
            *[pl.lit(None).cast(pl.Float32).alias(f"agent_y_{bp}") for bp in BODY_PARTS if f"agent_y_{bp}" not in columns],
            *[pl.lit(None).cast(pl.Float32).alias(f"target_x_{bp}") for bp in BODY_PARTS if f"target_x_{bp}" not in columns],
            *[pl.lit(None).cast(pl.Float32).alias(f"target_y_{bp}") for bp in BODY_PARTS if f"target_y_{bp}" not in columns],
        )

        features = merged_pivot.with_columns(
            pl.lit(agent_mouse_id).alias("agent_mouse_id"),
            pl.lit(target_mouse_id).alias("target_mouse_id"),
        ).select(
            pl.col("video_frame"),
            pl.col("agent_mouse_id"),
            pl.col("target_mouse_id"),
            *[
                body_parts_distance("agent", agent_body_part, "target", target_body_part).alias(
                    f"at__{agent_body_part}__{target_body_part}__distance"
                )
                for agent_body_part, target_body_part in itertools.product(BODY_PARTS, repeat=2)
            ],
            *[
                body_part_speed("agent", body_part, period_ms).alias(f"agent__{body_part}__speed_{period_ms}ms")
                for body_part, period_ms in itertools.product(["ear_left", "ear_right", "tail_base"], [500, 1000, 2000, 3000])
            ],
            *[
                body_part_speed("target", body_part, period_ms).alias(f"target__{body_part}__speed_{period_ms}ms")
                for body_part, period_ms in itertools.product(["ear_left", "ear_right", "tail_base"], [500, 1000, 2000, 3000])
            ],
            elongation("agent").alias("agent__elongation"),
            elongation("target").alias("target__elongation"),
            body_angle("agent").alias("agent__body_angle"),
            body_angle("target").alias("target__body_angle"),
        )

        result_element = result_element.join(
            features,
            on=["video_frame", "agent_mouse_id", "target_mouse_id"],
            how="left",
        )
        result.append(result_element)

    return pl.concat(result, how="vertical")
