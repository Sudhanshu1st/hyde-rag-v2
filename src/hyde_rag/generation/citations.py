"""Post-hoc citation validation.

Rules (brief): every cited ID must exist in the context, and every sentence must carry at least
one citation. Citations are inline, e.g. "Cool food within 2 hours [fda_food_code:87:3]."
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_ID = r"[A-Za-z0-9_.\-]+:\d+:\d+"
CITATION_GROUP = re.compile(rf"\[\s*({_ID}(?:\s*[,;]\s*{_ID})*)\s*\]")
_SPLIT_IDS = re.compile(r"\s*[,;]\s*")
# Split after sentence-ending punctuation (optionally followed by a citation group).
_SENTENCE_END = re.compile(rf"(?<=[.!?])((?:\s*\[\s*{_ID}(?:\s*[,;]\s*{_ID})*\s*\])*)\s+(?=[A-Z0-9\"'(\[])")


@dataclass
class CitedSentence:
    text: str  # sentence without citation markers
    cited_ids: list[str]


@dataclass
class CitationReport:
    sentences: list[CitedSentence] = field(default_factory=list)
    unknown_ids: list[str] = field(default_factory=list)  # cited but not in the context
    uncited: list[int] = field(default_factory=list)  # indices of sentences with no citation

    @property
    def valid(self) -> bool:
        """Strict: every sentence cites, and only IDs from the context."""
        return not self.unknown_ids and not self.uncited

    @property
    def grounded(self) -> bool:
        """Lenient: the answer cites at least one passage, and only IDs from the context
        (one citation may cover a whole paragraph)."""
        return bool(self.cited_ids) and not self.unknown_ids

    @property
    def cited_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for s in self.sentences:
            for c in s.cited_ids:
                seen.setdefault(c, None)
        return list(seen)


_ABBREVIATION_END = re.compile(
    r"(?:\b[A-Z]\.|\b(?:e\.g|i\.e|etc|vs|cf|ca|approx|No|Nos|Sec|Fig|Dr|Inc|Ltd|Co|St|al)\.)$", re.IGNORECASE
)
_LABEL_LINE = re.compile(r"^[A-Z][A-Za-z]*:\s*[A-Z]")  # one-word label + value: "Verdict: SUPPORTED"
# gpt-oss sometimes cites in its native 【id】 style; same citation, different brackets.
_FULLWIDTH_BRACKETS = str.maketrans({"【": "[", "】": "]"})
_LIST_MARKER_END = re.compile(r"(?:^|:\s*)(?:\d+|[a-z])[.)]$")  # "1." alone, or after a lead-in


def _is_fragment(part: str) -> bool:
    """A piece that is not a sentence of its own: it ends in an abbreviation or initial ("U.S."),
    is a list number ("1."), is a lead-in ("... should:"), or is under three words ("No.")."""
    if CITATION_GROUP.search(part) or _LABEL_LINE.match(part):
        return False  # it carries its own citation, or is a "Verdict: ..." style label line
    return bool(
        _ABBREVIATION_END.search(part)
        or _LIST_MARKER_END.search(part)
        or part.endswith(":")
        or len(part.split()) < 3
    )


def split_sentences(text: str) -> list[str]:
    """Split into sentences, keeping citation groups that follow the full stop with their sentence.
    Uncited fragments (abbreviations, list numbers, lead-ins, "No.") join the following sentence."""
    text = text.strip()
    if not text:
        return []
    parts, start = [], 0
    for m in _SENTENCE_END.finditer(text):
        parts.append(text[start : m.end(1)].strip())
        start = m.end()
    parts.append(text[start:].strip())
    # Lines like "- item" or headings are separate sentences too.
    pieces = []
    for p in parts:
        pieces.extend(line.strip() for line in p.splitlines() if line.strip())
    out: list[str] = []
    carry = ""
    for piece in pieces:
        piece = f"{carry} {piece}".strip() if carry else piece
        carry = ""
        if _is_fragment(piece):
            carry = piece
            continue
        out.append(piece)
    if carry:
        out.append(carry)
    return out


def extract_ids(sentence: str) -> list[str]:
    ids: list[str] = []
    for group in CITATION_GROUP.findall(sentence):
        ids.extend(i for i in _SPLIT_IDS.split(group.strip()) if i)
    return ids


def strip_citations(sentence: str) -> str:
    return re.sub(r"\s+([.,;!?])", r"\1", CITATION_GROUP.sub("", sentence)).strip()


def validate(answer: str, context_ids: set[str], exempt: tuple[str, ...] = ()) -> CitationReport:
    """Check citations against the context. Sentences starting with an ``exempt`` prefix
    (e.g. a "Verdict:" label line) don't need a citation."""
    report = CitationReport()
    for sentence in split_sentences(answer.translate(_FULLWIDTH_BRACKETS)):
        ids = extract_ids(sentence)
        clean = strip_citations(sentence)
        if not clean and ids and report.sentences:
            # A citation group on its own line belongs to the previous sentence.
            report.sentences[-1].cited_ids.extend(ids)
        else:
            report.sentences.append(CitedSentence(clean, ids))
        report.unknown_ids.extend(i for i in ids if i not in context_ids)
    for idx, s in enumerate(report.sentences):
        if not s.cited_ids and not s.text.startswith(exempt):
            report.uncited.append(idx)
    report.unknown_ids = list(dict.fromkeys(report.unknown_ids))
    return report
