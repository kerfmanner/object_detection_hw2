import tomllib
from pathlib import Path

import polars as pl
import pytest
from decouple import config

from src.data.annotations import AnnotationTables, load_annotations, load_split_clips

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG_PATH = PROJECT_ROOT / "config" / "data.toml"


@pytest.fixture(scope="session")
def data_root() -> Path:
    return Path(config("DATA_ROOT"))


@pytest.fixture(scope="session")
def smoke_config() -> dict:
    return tomllib.loads(DATA_CONFIG_PATH.read_text())["smoke"]


@pytest.fixture(scope="session")
def split_clips() -> dict[str, list[str]]:
    return load_split_clips(DATA_CONFIG_PATH)


@pytest.fixture(scope="session")
def annotations(data_root: Path, split_clips: dict[str, list[str]]) -> AnnotationTables:
    return load_annotations(data_root, split_clips)


@pytest.fixture(scope="session")
def smoke_annotations(annotations: AnnotationTables, smoke_config: dict) -> AnnotationTables:
    smoke_images = annotations.images.filter(
        pl.col("clip").is_in([smoke_config["train_clip"], smoke_config["val_clip"]]),
        pl.col("frame_id").is_in(smoke_config["frame_ids"]),
    )
    keys = smoke_images.select("split", "image_id")
    return AnnotationTables(
        images=smoke_images,
        objects=annotations.objects.join(keys, on=["split", "image_id"], maintain_order="left"),
        ignored=annotations.ignored.join(keys, on=["split", "image_id"], maintain_order="left"),
    )
