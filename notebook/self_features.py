
def make_self_features(
    metadata: dict,
    tracking: pl.DataFrame,
) -> pl.DataFrame:
    def body_parts_distance(body_part_1, body_part_2):
        # agentの体の各部位間距離(cm)
        assert body_part_1 in BODY_PARTS
        assert body_part_2 in BODY_PARTS
        return (
            (pl.col(f"agent_x_{body_part_1}") - pl.col(f"agent_x_{body_part_2}")).pow(2)
            + (pl.col(f"agent_y_{body_part_1}") - pl.col(f"agent_y_{body_part_2}")).pow(2)
        ).sqrt() / metadata["pix_per_cm_approx"]

    def body_part_speed(body_part, period_ms):
        # 部位の推定速度(cm/s)
        assert body_part in BODY_PARTS
        window_frames = max(1, int(round(period_ms * metadata["frames_per_second"] / 1000.0)))
        return (
            ((pl.col(f"agent_x_{body_part}").diff()).pow(2) + (pl.col(f"agent_y_{body_part}").diff()).pow(2)).sqrt()
            / metadata["pix_per_cm_approx"]
            * metadata["frames_per_second"]
        ).rolling_mean(window_size=window_frames, center=True, min_samples=1)

    def elongation():
        # 伸長度
        d1 = body_parts_distance("nose", "tail_base")
        d2 = body_parts_distance("ear_left", "ear_right")
        return d1 / (d2 + 1e-06)

    def body_angle():
        # 体角度(deg)
        v1x = pl.col("agent_x_nose") - pl.col("agent_x_body_center")
        v1y = pl.col("agent_y_nose") - pl.col("agent_y_body_center")
        v2x = pl.col("agent_x_tail_base") - pl.col("agent_x_body_center")
        v2y = pl.col("agent_y_tail_base") - pl.col("agent_y_body_center")
        return (v1x * v2x + v1y * v2y) / ((v1x.pow(2) + v1y.pow(2)).sqrt() * (v2x.pow(2) + v2y.pow(2)).sqrt() + 1e-06)

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

    for agent_mouse_id in range(1, n_mice + 1):
        result_element = pl.DataFrame(
            {
                "video_id": metadata["video_id"],
                "agent_mouse_id": agent_mouse_id,
                "target_mouse_id": -1,
                "video_frame": pl.arange(start_frame, end_frame + 1, eager=True),
            },
            schema={
                "video_id": pl.Int32,
                "agent_mouse_id": pl.Int8,
                "target_mouse_id": pl.Int8,
                "video_frame": pl.Int32,
            },
        )

        pivot = pivot_trackings[agent_mouse_id].select(
            pl.col("video_frame"),
            pl.exclude("video_frame").name.prefix("agent_"),
        )
        columns = pivot.columns
        pivot = pivot.with_columns(
            *[pl.lit(None).cast(pl.Float32).alias(f"agent_x_{bp}") for bp in BODY_PARTS if f"agent_x_{bp}" not in columns],
            *[pl.lit(None).cast(pl.Float32).alias(f"agent_y_{bp}") for bp in BODY_PARTS if f"agent_y_{bp}" not in columns],
        )

        features = pivot.with_columns(
            pl.lit(agent_mouse_id).alias("agent_mouse_id"),
            pl.lit(-1).alias("target_mouse_id"),
        ).select(
            pl.col("video_frame"),
            pl.col("agent_mouse_id"),
            pl.col("target_mouse_id"),
            *[
                body_parts_distance(body_part_1, body_part_2).alias(f"aa__{body_part_1}__{body_part_2}__distance")
                for body_part_1, body_part_2 in itertools.combinations(BODY_PARTS, 2)
            ],
            *[
                body_part_speed(body_part, period_ms).alias(f"agent__{body_part}__speed_{period_ms}ms")
                for body_part, period_ms in itertools.product(["ear_left", "ear_right", "tail_base"], [500, 1000, 2000, 3000])
            ],
            elongation().alias("agent__elongation"),
            body_angle().alias("agent__body_angle"),
        )

        result_element = result_element.join(
            features,
            on=["video_frame", "agent_mouse_id", "target_mouse_id"],
            how="left",
        )
        result.append(result_element)

    return pl.concat(result, how="vertical")
