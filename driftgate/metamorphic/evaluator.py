"""Zero-cost metamorphic evaluators and pluggable binary judges.

The default semantic similarity engine deliberately has no model dependency:
it uses stable SHA-1 feature hashing over word and character n-grams. This is
not intended to replace a domain embedding model; it provides a deterministic,
free baseline for PR gates. Projects can inject an embedding provider or a
judge implementation in nightly evaluation without changing the test API.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol

import numpy as np
from numpy.typing import NDArray


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> NDArray[np.float64]:
        """Return one L2-normalized vector per input text."""


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class HashingEmbedder:
    """Stable local character n-gram plus token embedding baseline."""

    def __init__(
        self,
        dim: int = 512,
        *,
        char_ngrams: tuple[int, ...] = (2, 3, 4),
        word_weight: float = 1.0,
        char_weight: float = 1.0,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.char_ngrams = char_ngrams
        self.word_weight = word_weight
        self.char_weight = char_weight

    def _bucket(self, feature: str) -> int:
        return int(hashlib.sha1(feature.encode("utf-8")).hexdigest()[:8], 16) % self.dim

    def embed(self, texts: Sequence[str]) -> NDArray[np.float64]:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float64)
        for row, text in enumerate(texts):
            lowered = text.casefold()
            for token in _TOKEN_RE.findall(lowered):
                vectors[row, self._bucket(f"w:{token}")] += self.word_weight
            padded = f"^{lowered}$"
            for n in self.char_ngrams:
                for start in range(max(0, len(padded) - n + 1)):
                    ngram = padded[start : start + n]
                    vectors[row, self._bucket(f"c{n}:{ngram}")] += self.char_weight
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        nonzero = norms[:, 0] > 0
        vectors[nonzero] /= norms[nonzero]
        return vectors


def cosine_similarity(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    """Cosine similarity, robust to accidental un-normalized injected vectors."""
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


@dataclass(frozen=True)
class InvarianceResult:
    relation: str
    passed: bool
    similarity: float
    threshold: float
    baseline: str
    variant: str


class InvarianceChecker:
    def __init__(
        self, embedder: EmbeddingProvider | None = None, *, threshold: float = 0.90
    ) -> None:
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("cosine threshold must be between -1 and 1")
        self.embedder = embedder or HashingEmbedder()
        self.threshold = threshold

    def check(
        self,
        baseline: str,
        variant: str,
        relation: str,
        *,
        threshold: float | None = None,
    ) -> InvarianceResult:
        effective_threshold = self.threshold if threshold is None else threshold
        vectors = self.embedder.embed([baseline, variant])
        similarity = cosine_similarity(vectors[0], vectors[1])
        return InvarianceResult(
            relation=relation,
            passed=similarity >= effective_threshold,
            similarity=similarity,
            threshold=effective_threshold,
            baseline=baseline,
            variant=variant,
        )


class JudgeVerdict(NamedTuple):
    passed: bool
    rationale: str
    rubric_id: str | None = None


class Judge(Protocol):
    def score(
        self, instruction: str, output: str, rubric: str, *, rubric_id: str | None = None
    ) -> JudgeVerdict:
        """Return one decomposed binary PASS/FAIL verdict."""


GEVAL_COT_TEMPLATE = """You are grading exactly one atomic quality dimension.

Instruction:
{instruction}

Candidate output:
{output}

Binary rubric:
{rubric}

Think step by step about the rubric. Do not use a numerical scale. End with
exactly one final line: VERDICT: PASS or VERDICT: FAIL.
"""

SEMANTIC_SIMILARITY_TEMPLATE = """You are a quality-control evaluator. Your task is to determine if a new AI-generated response is a valid substitute for a previously approved 'golden' response, given the original user's prompt.

The 'New Response' is a failure (FAIL) if it loses critical information, introduces contradictions, or changes the core meaning of the 'Golden Response'. It is a success (PASS) if it preserves the essential meaning, even if the wording, tone, or formatting is different.

User Prompt:
{prompt}

Golden Response:
{golden_response}

New Response:
{actual_response}

Think step by step about whether the New Response is a valid semantic substitute for the Golden Response. End with exactly one final line: VERDICT: PASS or VERDICT: FAIL.
"""


class SemanticComparer(Protocol):
    def compare(
        self,
        prompt: str,
        golden_response: str,
        actual_response: str,
        *,
        rubric_id: str | None = None,
    ) -> JudgeVerdict:
        """Is ``actual_response`` a valid substitute for ``golden_response``?"""


class SemanticSimilarityJudge:
    """LLM judge comparing a response to an approved golden response."""

    def __init__(self, transport: Callable[[str], str]) -> None:
        self.transport = transport

    def compare(
        self,
        prompt: str,
        golden_response: str,
        actual_response: str,
        *,
        rubric_id: str | None = None,
    ) -> JudgeVerdict:
        rendered = SEMANTIC_SIMILARITY_TEMPLATE.format(
            prompt=prompt, golden_response=golden_response, actual_response=actual_response
        )
        first = parse_verdict(self.transport(rendered), rubric_id=rubric_id)
        if first.rationale.startswith("Unparsable"):
            retry = (
                rendered
                + "\nYour previous output was invalid. End with VERDICT: PASS or VERDICT: FAIL."
            )
            return parse_verdict(self.transport(retry), rubric_id=rubric_id)
        return first


class EmbeddingSimilarityJudge:
    """Free, offline stand-in used when no LLM judge is configured.

    Compares the response to the golden one with the local ``HashingEmbedder``.
    That measures lexical/n-gram overlap, not meaning: it reliably catches
    empty, truncated or off-topic responses and unchanged ones, and it will
    flag legitimate rewordings. Use an LLM judge for nightly semantic checks.
    It never passes by default: a response must clear ``threshold``.
    """

    def __init__(
        self, embedder: EmbeddingProvider | None = None, *, threshold: float = 0.8
    ) -> None:
        self.checker = InvarianceChecker(embedder, threshold=threshold)

    def compare(
        self,
        prompt: str,
        golden_response: str,
        actual_response: str,
        *,
        rubric_id: str | None = None,
    ) -> JudgeVerdict:
        del prompt
        result = self.checker.check(golden_response, actual_response, "golden_similarity")
        verdict = "PASS" if result.passed else "FAIL"
        rationale = (
            f"lexical similarity {result.similarity:.3f} "
            f"{'>=' if result.passed else '<'} threshold {result.threshold:.3f}\nVERDICT: {verdict}"
        )
        return JudgeVerdict(result.passed, rationale, rubric_id)


def parse_verdict(text: str, *, rubric_id: str | None = None) -> JudgeVerdict:
    """Fail closed if a remote judge does not emit the required binary tail."""
    matches = re.findall(
        r"^\s*VERDICT:\s*(PASS|FAIL)\s*$", text, flags=re.IGNORECASE | re.MULTILINE
    )
    if not matches:
        return JudgeVerdict(False, "Unparsable judge output: missing VERDICT: PASS|FAIL", rubric_id)
    passed = matches[-1].upper() == "PASS"
    rationale = text.strip()
    return JudgeVerdict(passed, rationale, rubric_id)


@dataclass(frozen=True)
class RubricRequirements:
    required_keywords: tuple[str, ...] = ()
    forbidden_patterns: tuple[str, ...] = ()
    require_nonempty: bool = True


class DeterministicRubricJudge:
    """Free judge used by default in PR gates.

    This is intentionally narrow and explicit: it checks only the mechanical
    requirements the caller encodes (non-empty response, keywords present,
    forbidden patterns absent). It never impersonates semantic intelligence;
    teams use an injected LLM judge callback in nightly jobs for that.
    """

    def __init__(self, requirements: RubricRequirements | None = None) -> None:
        self.requirements = requirements or RubricRequirements()

    def score(
        self, instruction: str, output: str, rubric: str, *, rubric_id: str | None = None
    ) -> JudgeVerdict:
        del instruction, rubric
        if self.requirements.require_nonempty and not output.strip():
            return JudgeVerdict(False, "Output is empty", rubric_id)
        lowered = output.casefold()
        missing = [
            keyword
            for keyword in self.requirements.required_keywords
            if keyword.casefold() not in lowered
        ]
        if missing:
            return JudgeVerdict(
                False, f"Missing required keyword(s): {', '.join(missing)}", rubric_id
            )
        matched = [
            pattern
            for pattern in self.requirements.forbidden_patterns
            if re.search(pattern, output, re.I)
        ]
        if matched:
            return JudgeVerdict(
                False, f"Matched forbidden pattern(s): {', '.join(matched)}", rubric_id
            )
        return JudgeVerdict(True, "All deterministic binary requirements satisfied", rubric_id)


class LLMRubricJudge:
    """Provider-neutral adapter around an injected model transport.

    ``transport`` owns provider SDK/authentication details and accepts a fully
    rendered G-Eval prompt. The framework makes no hidden network calls. An
    unparsable response retries once and then fails closed.
    """

    def __init__(self, transport: Callable[[str], str]) -> None:
        self.transport = transport

    def score(
        self, instruction: str, output: str, rubric: str, *, rubric_id: str | None = None
    ) -> JudgeVerdict:
        prompt = GEVAL_COT_TEMPLATE.format(instruction=instruction, output=output, rubric=rubric)
        first = parse_verdict(self.transport(prompt), rubric_id=rubric_id)
        if first.rationale.startswith("Unparsable"):
            retry_prompt = (
                prompt
                + "\nYour previous output was invalid. End with VERDICT: PASS or VERDICT: FAIL."
            )
            return parse_verdict(self.transport(retry_prompt), rubric_id=rubric_id)
        return first


def evaluate_structural_invariance(
    validate: Callable[[dict[str, Any]], bool],
    variants: Sequence[dict[str, Any]],
) -> list[bool]:
    """Oracle for adversarial-noise relations: every variant must stay valid."""
    return [bool(validate(variant)) for variant in variants]
