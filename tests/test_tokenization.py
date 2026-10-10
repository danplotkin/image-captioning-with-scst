from __future__ import annotations

import pytest
from transformers import BertTokenizerFast

from scst_captioner.rewards import CiderDReward, _normalize_cider_caption
from scst_captioner.tokenization import decode_caption_batch, normalize_caption_text


@pytest.fixture
def wordpiece_tokenizer(tmp_path):
    # Construct the same BERT WordPiece decoder used by the pinned DistilBERT
    # tokenizer entirely locally; tests must never need model downloads.
    vocabulary = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "a",
        "girl",
        "holding",
        "horse",
        "s",
        "lead",
        "behind",
        "fire",
        ".",
        "'",
        "tri",
        "-",
        "colored",
        "dog",
        "the",
        "dogs",
        "tails",
        "are",
        "wagging",
        "can",
        "t",
        "hold",
        "she",
        "re",
        "m",
        "we",
        "ve",
        "ll",
        "d",
        "ca",
        "n",
        "blue",
        "sign",
        '"',
        "two",
        "year",
        "old",
        "boy",
        "c",
        "statefarm",
        "com",
        "o",
        "clock",
        "slip",
        "slide",
        "rif",
        "=",
        "ding",
        "ependent",
        "jackson",
        "thriller",
    ]
    vocab_file = tmp_path / "vocab.txt"
    vocab_file.write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
    return BertTokenizerFast(
        vocab_file=str(vocab_file),
        do_lower_case=True,
        clean_up_tokenization_spaces=True,  # Helper must override this default.
    )


@pytest.mark.parametrize(
    ("caption", "expected_tokens"),
    [
        (
            "A girl holding a horse 's lead behind a fire .",
            "a girl holding a horse 's lead behind a fire",
        ),
        ("A tri-colored dog .", "a tri-colored dog"),
        ("A tri - colored dog .", "a tri-colored dog"),
        ("A two-year-old boy .", "a two-year-old boy"),
        ("A girl can't hold the horse .", "a girl ca n't hold the horse"),
        ("A girl ca n't hold the horse .", "a girl ca n't hold the horse"),
        ("The dogs ' tails are wagging .", "the dogs tails are wagging"),
        ("The dogs' tails are wagging .", "the dogs tails are wagging"),
        ('The "blue" sign .', "the blue sign"),
        ("The ' blue ' sign .", "the blue sign"),
        ("The 's' sign .", "the s sign"),
        ("A D.C. sign .", "a d c sign"),
        ("A statefarm.com sign .", "a statefarm com sign"),
        ("A o'clock sign .", "a o clock sign"),
        ("A slip 'n' slide .", "a slip n slide"),
        ("A RIF=ding sign .", "a rif = ding sign"),
        ('A "-ependent" sign .', "a ependent sign"),
        ("Jackson 's ' Thriller ' sign .", "jackson s thriller sign"),
    ],
)
def test_wordpiece_targets_and_raw_references_have_equal_reward_tokens(
    wordpiece_tokenizer, caption, expected_tokens
):
    ids = wordpiece_tokenizer(caption)["input_ids"]
    assert wordpiece_tokenizer.unk_token_id not in ids
    decoded = decode_caption_batch(wordpiece_tokenizer, [ids])[0]
    assert _normalize_cider_caption(decoded) == expected_tokens
    assert _normalize_cider_caption(caption) == expected_tokens
    assert _normalize_cider_caption(decoded + " <eos>") == expected_tokens + " <eos>"


@pytest.mark.parametrize(
    "caption",
    [
        "A girl holding a horse 's lead behind a fire .",
        "A tri-colored dog .",
        "A girl ca n't hold the horse .",
        "The dogs ' tails are wagging .",
        "A D.C. sign .",
        "A statefarm.com sign .",
    ],
)
def test_exact_wordpiece_targets_receive_exact_reference_cider_reward(wordpiece_tokenizer, caption):
    pytest.importorskip("pycocoevalcap.cider.cider_scorer")
    references = [caption + " <eos>"] * 2
    corpus = [references, ["unrelated birds fly above green trees <eos>"] * 2]
    reward = CiderDReward(corpus, expected_references_per_image=2)
    ids = wordpiece_tokenizer(caption)["input_ids"]
    decoded = decode_caption_batch(wordpiece_tokenizer, [ids])[0] + " <eos>"
    assert reward([decoded], [references]) == pytest.approx((10.0,), abs=1e-12)
    assert reward([decoded], [references]) == reward([references[0]], [references])


def test_decoder_cleanup_is_explicit_and_does_not_merge_plural_possessives(wordpiece_tokenizer):
    ids = wordpiece_tokenizer("The dogs ' tails .")["input_ids"]
    # HF's optional string cleanup merges two lexical words in this example.
    assert "dogs'tails" in wordpiece_tokenizer.decode(ids, skip_special_tokens=True)
    assert decode_caption_batch(wordpiece_tokenizer, [ids]) == ["the dogs ' tails."]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("horse ' s", "horse's"),
        ("we ' re", "we're"),
        ("she ' d", "she'd"),
        ("we ' ve", "we've"),
        ("we ' ll", "we'll"),
        ("I ' m", "I'm"),
        ("ca n ' t", "can't"),
        ("tri - colored", "tri-colored"),
        ("tri -- colored", "tri -- colored"),
        ("the ' s ' sign", "the ' s ' sign"),
        ("dogs ' tails", "dogs ' tails"),
        ("<eos>", "<eos>"),
    ],
)
def test_spacing_normalization_is_narrow_and_idempotent(text, expected):
    assert normalize_caption_text(text) == expected
    assert normalize_caption_text(expected) == expected
