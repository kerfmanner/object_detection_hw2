import polars as pl

from src.data.classes import CLASS_NAMES


def build_class_table() -> pl.DataFrame:
    """Build a table of all classes.

    Returns:
        One row per class with `class_id` (0-based) and `class_name`.
    """
    return pl.DataFrame({"class_id": range(len(CLASS_NAMES)), "class_name": CLASS_NAMES})


def count_class_presence(images: pl.DataFrame, objects: pl.DataFrame) -> pl.DataFrame:
    """Count instances of each class and the frames and clips that contain it, per split.

    Args:
        images: Image table with `split`, `image_id`, and `clip`.
        objects: Object table with `split`, `image_id`, and `class_id`.

    Returns:
        One row per split and class, including classes without instances, with `instances`,
        `frames` and `clips` containing the class, and their shares of the split totals
        (`instance_share`, `frame_share`, `clip_share`).
    """
    split_totals = images.group_by("split").agg(
        pl.len().alias("split_frames"), pl.col("clip").n_unique().alias("split_clips")
    )
    presence = (
        objects.join(images.select("split", "image_id", "clip"), on=["split", "image_id"])
        .group_by("split", "class_id")
        .agg(
            pl.len().alias("instances"),
            pl.col("image_id").n_unique().alias("frames"),
            pl.col("clip").n_unique().alias("clips"),
        )
    )
    return (
        split_totals.join(build_class_table(), how="cross")
        .join(presence, on=["split", "class_id"], how="left")
        .with_columns(pl.col("instances", "frames", "clips").fill_null(0))
        .with_columns(
            (pl.col("instances") / pl.col("instances").sum().over("split")).alias("instance_share"),
            (pl.col("frames") / pl.col("split_frames")).alias("frame_share"),
            (pl.col("clips") / pl.col("split_clips")).alias("clip_share"),
        )
        .drop("split_frames", "split_clips")
        .sort("split", "class_id")
    )


def compute_clip_class_shares(images: pl.DataFrame, objects: pl.DataFrame) -> pl.DataFrame:
    """Compute the share of each class among the labelled objects of every clip.

    Args:
        images: Image table with `split`, `image_id`, and `clip`.
        objects: Object table with `split`, `image_id`, and `class_id`.

    Returns:
        One row per clip with `split`, `clip`, `objects`, and one share column per class
        name in class id order; shares of a clip with objects sum to 1.
    """
    clips = images.select("split", "clip").unique()
    class_counts = (
        objects.join(images.select("split", "image_id", "clip"), on=["split", "image_id"])
        .group_by("split", "clip", "class_id")
        .len("instances")
    )
    return (
        clips.join(build_class_table(), how="cross")
        .join(class_counts, on=["split", "clip", "class_id"], how="left")
        .with_columns(pl.col("instances").fill_null(0))
        .with_columns(pl.col("instances").sum().over("split", "clip").alias("objects"))
        .with_columns((pl.col("instances") / pl.col("objects")).fill_nan(0.0).alias("share"))
        .pivot(on="class_name", index=["split", "clip", "objects"], values="share")
        .select("split", "clip", "objects", *CLASS_NAMES)
        .sort("split", "clip")
    )


def compute_resize_scale(min_size: int, max_size: int) -> pl.Expr:
    """Build an expression for the torchvision detection resize factor of each frame.

    The shorter side is scaled to `min_size` unless the longer side would exceed `max_size`,
    as in `GeneralizedRCNNTransform`:
    https://github.com/pytorch/vision/blob/main/torchvision/models/detection/transform.py

    Args:
        min_size: Target length of the shorter frame side in px.
        max_size: Upper limit of the longer frame side in px.

    Returns:
        Expression over `width` and `height` columns with the scale factor.
    """
    short_side = pl.min_horizontal("width", "height")
    long_side = pl.max_horizontal("width", "height")
    return pl.min_horizontal(min_size / short_side, max_size / long_side)


def compute_resized_frame_size(min_size: int, max_size: int) -> list[pl.Expr]:
    """Build expressions for the frame size after the torchvision detection resize.

    Each side becomes `floor(side * scale)` in double precision, as in
    `torch.nn.functional.interpolate` with `recompute_scale_factor=True`.

    Args:
        min_size: Target length of the shorter frame side in px.
        max_size: Upper limit of the longer frame side in px.

    Returns:
        Expressions over `width` and `height` named `resized_width` and `resized_height` in px.
    """
    scale = compute_resize_scale(min_size, max_size)
    return [
        (pl.col(side) * scale).floor().cast(pl.Int64).alias(f"resized_{side}")
        for side in ("width", "height")
    ]


def compute_required_max_size(min_size: int) -> pl.Expr:
    """Build an expression for the smallest `max_size` that keeps the shorter side at `min_size`.

    Args:
        min_size: Target length of the shorter frame side in px.

    Returns:
        Expression over `width` and `height` columns with the required `max_size` in px.
    """
    short_side = pl.min_horizontal("width", "height")
    long_side = pl.max_horizontal("width", "height")
    return (min_size * long_side / short_side).ceil().cast(pl.Int64)


def summarize_quantiles(
    table: pl.DataFrame, column: str, by: list[str], quantiles: tuple[float, ...]
) -> pl.DataFrame:
    """Compute count, mean, and quantiles of one column per group.

    Args:
        table: Table that contains `column` and the `by` columns.
        column: Numeric column to summarize.
        by: Grouping columns.
        quantiles: Quantile levels in [0, 1].

    Returns:
        One row per group with `count`, `mean`, and one `p<percent>` column per quantile.
    """
    return (
        table.group_by(by)
        .agg(
            pl.len().alias("count"),
            pl.col(column).mean().alias("mean"),
            *(
                pl.col(column).quantile(q, interpolation="linear").alias(f"p{round(q * 100)}")
                for q in quantiles
            ),
        )
        .sort(by)
    )


def count_objects_per_frame(images: pl.DataFrame, objects: pl.DataFrame) -> pl.DataFrame:
    """Count labelled objects in every frame, including frames without objects.

    Args:
        images: Image table with `split`, `image_id`, and `clip`.
        objects: Object table with `split` and `image_id`.

    Returns:
        One row per frame with `split`, `image_id`, `clip`, and `objects`.
    """
    counts = objects.group_by("split", "image_id").len("objects")
    return (
        images.select("split", "image_id", "clip")
        .join(counts, on=["split", "image_id"], how="left")
        .with_columns(pl.col("objects").fill_null(0))
    )
