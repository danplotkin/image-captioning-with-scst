"""COCO CIDEr-D with support for a fixed training-corpus IDF denominator.

Imported lazily by rewards.py because pycocoevalcap is an optional dependency.
The upstream compute_cider() unconditionally sets ref_len to log(batch size).
Keep its scoring formula, but let an explicit corpus_ref_len take precedence.
"""

from __future__ import annotations

from pycocoevalcap.cider.cider_scorer import CiderScorer, cook_refs, cook_test


class CiderDScorer(CiderScorer):
    corpus_ref_len: float | None = None

    @property
    def ref_len(self) -> float | None:
        if self.corpus_ref_len is not None:
            return self.corpus_ref_len
        return self._batch_ref_len

    @ref_len.setter
    def ref_len(self, value: float | None) -> None:
        self._batch_ref_len = value

    def cook_append(self, test: str | None, refs: list[str] | None) -> None:
        # Upstream omits n here, always cooking four orders even for n != 4.
        if refs is not None:
            self.crefs.append(cook_refs(refs, n=self.n))
            self.ctest.append(cook_test(test, n=self.n) if test is not None else None)
