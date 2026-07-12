from pathlib import Path

import pytest

from scst_captioner.cli import build_parser
from scst_captioner.config import load_config
from scst_captioner.tokenization import resolve_special_token_ids


class BertStyleTokenizer:
    bos_token_id = None
    eos_token_id = None
    cls_token_id = 101
    sep_token_id = 102
    pad_token_id = 0
    all_special_ids = (0, 100, 101, 102, 103)


def test_default_config_and_cli_are_loadable_without_downloads() -> None:
    config = load_config("configs/default.toml")
    assert config.model.freeze_encoder is True
    assert len(config.model.encoder_revision) == 40
    parser = build_parser()
    args = parser.parse_args(["validate-data", "--config", "configs/default.toml"])
    assert args.command == "validate-data"


def test_distilbert_cls_sep_are_resolved_as_generation_boundaries() -> None:
    special = resolve_special_token_ids(BertStyleTokenizer())
    assert special.bos == 101
    assert special.eos == 102
    assert special.pad == 0
    assert special.forbidden == (0, 100, 101, 103)


@pytest.mark.parametrize(
    "contents",
    [
        '[model]\nfreeze_encoder = "false"\n',
        "[model]\nnum_heads = true\n",
        "[model]\nd_model = 3\nnum_heads = 1\n",
        '[scst]\nreward = "typo"\n',
    ],
)
def test_config_rejects_wrong_runtime_types_and_literal_values(
    tmp_path: Path, contents: str
) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        load_config(path)
