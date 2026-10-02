from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from typing import Any, Pattern

from .schema import canonical_target


# v3: NegEx-style mention classification (negation scope per clause, post-negation,
# coordinated lists, pseudo-negation, hedging, indication context) and non-finding report
# sections (INDICATION/HISTORY/...) are masked. v2 matched "no <term>" literally, so
# "No evidence of acute or chronic pulmonary embolism" counted as a positive PE.
# v4: a period after a number ends the sentence; "Finding: No." answers; historical
# mentions (prior/h/o/known PE) and title/indication lines are ignored; partial
# resolution is not negation; "No acute PE" does not negate an affirmed chronic clot;
# RV/LV thresholds are compared numerically; acuity must modify the embolus itself; and
# the cue window is capped so an unpunctuated report stays near-linear.
RULE_VERSION = "pe_rules_v4"


@dataclass(frozen=True)
class EvidenceSpan:
    text: str
    start: int
    end: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuleDecision:
    resolved: bool
    value: bool | str | float | None
    reason: str
    evidence: EvidenceSpan | None = None
    version: str = RULE_VERSION

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


_FLAGS = re.IGNORECASE


def _patterns(*values: str) -> tuple[Pattern[str], ...]:
    return tuple(re.compile(value, _FLAGS) for value in values)


# ---- report sections ------------------------------------------------------------------
# Text under these headers states why the scan was ordered, not what it showed
# ("INDICATION: evaluate for pulmonary embolism"), so it is never evidence either way.
_SECTION_HEADER = re.compile(
    r"(?:^|(?<=[.;]))[ \t]*(?P<name>[A-Za-z][A-Za-z /&()-]{1,40}?)[ \t]*:", re.MULTILINE
)
_NON_FINDING_SECTIONS = frozenset({
    "indication", "indications", "clinical indication", "clinical indications", "history",
    "clinical history", "history of present illness", "clinical information", "clinical data",
    "clinical details", "clinical", "reason for exam", "reason for examination",
    "reason for study", "reason for referral", "reason", "exam", "examination", "comparison",
    "comparisons", "technique", "procedure", "protocol", "order",
})
# Headers that open the radiologist's answer; text before the first one is the preamble
# (exam title, indication), where "suspected PE" is why the scan was ordered.
_FINDING_SECTIONS = frozenset({
    "finding", "findings", "impression", "impressions", "conclusion", "conclusions",
    "summary", "opinion", "interpretation", "diagnosis", "result", "results",
})
# A sentence end that is not a decimal point ("1.5") or a date ("01.02.2020").
_SENTENCE_END = re.compile(r"[.!?](?!\d)")


def _section_name(header: re.Match[str]) -> str:
    return " ".join(header.group("name").lower().split())


def _finding_text(text: str) -> tuple[str, int]:
    """``text`` with non-finding sections blanked out (offsets and newlines preserved),
    and the offset where the first finding section starts (0 when there is none).

    A masked section runs to the next header; without a later header it ends at the next
    newline, or, on a one-line report, at the end of its first sentence ("Comparison:
    None. Pulmonary embolism in the right lower lobe." keeps the second sentence).
    """
    headers = list(_SECTION_HEADER.finditer(text))
    masked = list(text)
    preamble_end = next(
        (header.start("name") for header in headers if _section_name(header) in _FINDING_SECTIONS), 0
    )
    for index, header in enumerate(headers):
        if _section_name(header) not in _NON_FINDING_SECTIONS:
            continue
        if index + 1 < len(headers):
            end = headers[index + 1].start("name")
        else:
            newline = text.find("\n", header.end())
            if newline >= 0:
                end = newline
            else:
                sentence = _SENTENCE_END.search(text, header.end())
                end = len(text) if sentence is None else sentence.end()
        for position in range(header.start("name"), end):
            if masked[position] != "\n":
                masked[position] = " "
    return "".join(masked), preamble_end


# ---- mention context (NegEx-style) ----------------------------------------------------
# A cue only reaches mentions in its own clause: sentence punctuation (a period not
# followed by a digit, so "1.5" stays whole but "image 112." ends the sentence; common
# abbreviations such as "vs." and "Dr." do not end it), a newline, or a contrastive
# conjunction ends the clause.
_CLAUSE_BREAK = re.compile(
    r"(?<!\bdr)(?<!\bvs)(?<!\bcf)(?<!\bneg)(?<!\be)(?<!\be\.g)(?<!\bi)(?<!\bi\.e)(?<!\bapprox)"
    r"\.(?!\d)|[;!\n]|\b(?:but|however|although|though|except|whereas|otherwise|"
    r"aside from|apart from|other than)\b",
    _FLAGS,
)
# A clause longer than this is cut to the words nearest the mention (NegEx uses a window
# of a few words); without the cap an unpunctuated report is rescanned per mention.
_MAX_WINDOW_CHARS = 300
_NEGATION = re.compile(
    r"\b(?:no|not|nor|neither|without|negative|neg(?=\.?\s+for\b)|absence\s+of|absent|"
    r"free\s+of|resolution\s+of|resolved|clear\s+of)\b",
    _FLAGS,
)
# Negation words that do not negate the finding ("no change in the right lobar embolus").
_PSEUDO_NEGATION = re.compile(
    r"(?:no|not|without)\s+(?:(?:significant|interval|appreciable|substantial|definite)\s+)?"
    r"(?:change|increase|decrease|improvement|worsening|progression)\b|not\s+only\b",
    _FLAGS,
)
# "Partially resolved", "near complete resolution of": the finding is still there.
_PARTIAL_RESOLUTION = re.compile(
    r"\b(?:partial(?:ly)?|partly|incomplete(?:ly)?|near(?:ly)?(?:[- ]+complete(?:ly)?)?|"
    r"almost(?:\s+complete(?:ly)?)?|mostly|largely|predominantly|substantially|some|"
    r"slight(?:ly)?|minimal(?:ly)?|improving|decreas(?:ed|ing)|progressive(?:ly)?|"
    r"not\s+(?:yet\s+)?(?:fully|completely|entirely))\s+(?:interval\s+)?$",
    _FLAGS,
)
# "... with residual clot": a resolution cue in the same clause is only partial.
_RESIDUAL = re.compile(
    r"\b(?:residual|persistent|persisting|remaining)\b(?:\s+[\w-]+){0,3}?\s+(?:clots?|thromb\w*|"
    r"embol\w*|filling\s+defects?|PEs?)\b|\b(?:clots?|thromb\w*|embol\w*|filling\s+defects?)\b"
    r"[^.;\n]{0,30}?\b(?:persists?|remains?|is\s+still|are\s+still)\b",
    _FLAGS,
)
_RESIDUAL_NEGATED = re.compile(r"\b(?:no|not|without)\s+(?:[\w-]+\s+)?$", _FLAGS)
_NEGATION_POST = re.compile(
    r"[\s,:]*(?:(?:is|are|was|were|has|have|had)\s+(?:been\s+)?)?(?:"
    r"(?:not|no\s+longer)\s+(?:(?:definitely|clearly|convincingly|well|currently)\s+)?"
    r"(?:identified|seen|present|demonstrated|visuali[sz]ed|evident|detected|noted|"
    r"appreciated|apparent|shown|depicted)|absent|resolved|excluded|ruled\s+out|none|negative)\b",
    _FLAGS,
)
# Structured "Finding: answer" lines ("Pulmonary embolism: No.", "PE: no evidence").
_ANSWER_NEGATIVE = re.compile(
    r"\s*:\s*(?:no(?:\s+(?:definite\s+)?evidence)?|none(?:\s+(?:seen|identified))?|neg(?:ative)?|"
    r"absent|nil|not\s+(?:seen|identified|present|evident|detected|demonstrated))\.?\s*(?:,|$)",
    _FLAGS,
)
_ANSWER_POSITIVE = re.compile(r"\s*:\s*(?:yes|present|positive)\s*(?:,|$)", _FLAGS)
_UNCERTAIN = re.compile(
    r"\b(?:cannot|can\s+not|can't|could\s+not)\s+(?:be\s+)?(?:entirely\s+|completely\s+|"
    r"definitively\s+)?(?:exclude|excluded|rule\s+out|ruled\s+out)\b|"
    r"\bnot\s+(?:be\s+)?(?:entirely\s+|completely\s+)?(?:excluded|ruled\s+out)\b|"
    r"\b(?:may|might|could)\s+(?:represent|reflect|be)\b|"
    r"\b(?:possible|possibly|probable|probably|likely|unlikely|questionable|question\s+of|"
    r"questioned|equivocal|suspicious\s+for|suspected|suspicion|concerning\s+for|concern\s+for|"
    r"worrisome\s+for|suggestive\s+of|differential|versus|vs|presumed|nondiagnostic|"
    r"non-diagnostic|limited\s+evaluation|limited\s+assessment|suboptimal|no\s+new)\b|"
    r"(?<!age-)(?<!age )\bindeterminate\b(?![- ]+(?:age|acuity|chronicity))|\?",
    _FLAGS,
)
# Hedges written after the finding: "pulmonary embolism cannot be excluded".
_UNCERTAIN_POST = re.compile(
    r"[^,]{0,25}?(?:\b(?:cannot|can\s+not|can't|could\s+not|not)\s+(?:be\s+)?(?:entirely\s+|"
    r"completely\s+|definitively\s+)?(?:excluded|ruled\s+out)\b|\b(?:is|are)\s+(?:possible|"
    r"possibly|suspected|likely|probable|questioned|favou?red)\b|\b(?:versus|vs|unlikely|"
    r"less\s+likely)\b|\?)",
    _FLAGS,
)
# A past finding, not a statement about this examination ("Patient with h/o PE").
_HISTORICAL = re.compile(
    r"\b(?:(?:known|remote|prior|previous|past)\s+)?(?:history\s+of|h/o|hx(?:\s+of)?|"
    r"previously\s+(?:seen|noted|described|demonstrated|identified|reported|treated|diagnosed)|"
    r"status\s+post|s/p)(?![\w/])",
    _FLAGS,
)
# "Pulmonary emboli previously seen in the right lower lobe ..."
_HISTORICAL_POST = re.compile(
    r"[\s,]*previously\s+(?:seen|noted|described|demonstrated|identified|reported)\b", _FLAGS
)
# "Prior pulmonary embolism", "previous right lower lobe PE": a past-tense adjective at
# most three words before the mention; "known" only counts for an episodic finding
# (a known PE is history, a known emphysema is still present). "70-year-old" is not "old".
_HISTORICAL_WORDS = r"prior|previous|remote|(?<!year )(?<!years )(?<!-)old"
# Intermediate words may not start a new finding or name the comparison exam ("prior CT").
_HISTORICAL_ADJ_TAIL = (
    r"\s+(?:(?!(?:and|with|now|new|acute|but|which|that|cta?|scan|study|studies|exam\w*|"
    r"imaging|comparison|show\w*|demonstrat\w*)\b)[\w/-]+\s+){0,3}$"
)
_HISTORICAL_ADJ = re.compile(rf"\b(?:{_HISTORICAL_WORDS}){_HISTORICAL_ADJ_TAIL}", _FLAGS)
_HISTORICAL_ADJ_EPISODIC = re.compile(rf"\b(?:known|{_HISTORICAL_WORDS}){_HISTORICAL_ADJ_TAIL}", _FLAGS)
# Why the scan was ordered, written into the impression without a section header.
_INDICATION = re.compile(
    r"\b(?:evaluat(?:e|ed|ion|ing)\s+for|eval\s+for|assess(?:ed|ing|ment)?\s+for|"
    r"to\s+(?:evaluate|assess|exclude|rule\s+out)|rule\s+out|r/o|query|indication|"
    r"reason\s+for|ordered\s+for|protocol|clinical\s+(?:history|indication|suspicion))(?![\w/])",
    _FLAGS,
)
_INDICATION_POST = re.compile(
    r"\s*(?:protocol|study|exam(?:ination)?|evaluation|work-?up|cta|ct)\b", _FLAGS
)
# Exam titles and orders that name the finding directly: "Eval for PE", "Evaluate PE",
# "CT PULMONARY ANGIOGRAM FOR PE", "CTA chest (PE)".
_INDICATION_ADJACENT = re.compile(
    r"(?:\b(?:eval(?:uat(?:e|ed|es|ion|ing))?|assess(?:ed|ing|ment)?|(?<!not )(?<!cannot )"
    r"(?<!can't )(?:r/o|rule\s+out))(?:\s+(?:for|of))?|"
    r"\b(?:ct|cta|ctpa|angiogra(?:m|phy)|chest|study|exam(?:ination)?|scan|imaging)\s+for)"
    r"(?:\s+(?:possible|suspected|presumed|acute|an?))?\s+$"
    r"|\b(?:ct|cta|ctpa|angiogra(?:m|phy)|chest|study|exam(?:ination)?)\s*[(\[]\s*$",
    _FLAGS,
)
# In the preamble a suspicion is the indication ("Concern for PE." above IMPRESSION).
_SUSPICION = re.compile(
    r"\b(?:suspected|suspicion|concern(?:ing)?\s+for|query|question\s+of|r/o|rule\s+out)"
    r"(?![\w/])|\?",
    _FLAGS,
)
_SUSPICION_POST = re.compile(r"\s*\?", _FLAGS)
# "No A, small B": after a list separator a size/side/grade word starts a new, affirmed
# finding, so the earlier "no" does not reach B.
_ITEM_SEPARATOR = re.compile(r",|\b(?:and|with|plus)\b", _FLAGS)
_PRESENCE_MODIFIER = re.compile(
    r"\s*(?:there\s+(?:is|are)\b|(?:(?:a|an)\s+)?(?:trace|tiny|minimal|small|mild|moderate|"
    r"large|severe|massive|extensive|new|increased|increasing|enlarging|persistent|stable|"
    r"unchanged|residual|bilateral|left|right|loculated|layering|moderate-sized|"
    r"small-to-moderate|moderate-to-large)\b)",
    _FLAGS,
)
# "No A and the B is enlarged": a coordinated item with its own verb is its own clause.
_FINITE_VERB = re.compile(
    r"\b(?:is|are|was|were|appears?|remains?|demonstrates?|shows?|measures?)\b", _FLAGS
)
_FINITE_VERB_AFTER = re.compile(
    r"\s*(?:is|are|was|were|appears?|remains?|measures?)\b(?!\s+(?:not|no)\b)", _FLAGS
)
_COORDINATION = re.compile(r"\b(?:or|nor)\b", _FLAGS)
# Beyond this many words the reach of a negation cue is a guess.
_MAX_SCOPE_WORDS = 8
# A negation limited to acute PE ("No acute pulmonary embolism") says nothing about a
# chronic clot.
_ACUTE = re.compile(r"\bacute\b", _FLAGS)
_CHRONIC = re.compile(r"\bchronic\b", _FLAGS)


@dataclass(frozen=True)
class _Mention:
    status: str  # affirmed | negated | uncertain | ignored
    start: int
    end: int
    # A negation that only covers acute PE.
    acute_only: bool = False


class _Context:
    """Clause boundaries of one (section-masked) report, computed once per call."""

    def __init__(self, text: str, preamble_end: int = 0):
        self.text = text
        self.preamble_end = preamble_end
        breaks = list(_CLAUSE_BREAK.finditer(text))
        self._ends = [match.end() for match in breaks]
        self._starts = [match.start() for match in breaks]

    def clause(self, start: int, end: int) -> tuple[int, int]:
        before = bisect_right(self._ends, start)
        clause_start = self._ends[before - 1] if before else 0
        after = bisect_left(self._starts, end)
        clause_end = self._starts[after] if after < len(self._starts) else len(self.text)
        # Cut an overlong clause at a word boundary inside the window.
        if start - clause_start > _MAX_WINDOW_CHARS:
            cut = self.text.find(" ", start - _MAX_WINDOW_CHARS, start)
            clause_start = cut + 1 if cut >= 0 else start - _MAX_WINDOW_CHARS
        if clause_end - end > _MAX_WINDOW_CHARS:
            cut = self.text.rfind(" ", end, end + _MAX_WINDOW_CHARS)
            clause_end = cut if cut > end else end + _MAX_WINDOW_CHARS
        return clause_start, clause_end


def _has_residual(text: str) -> bool:
    return any(
        not _RESIDUAL_NEGATED.search(text, 0, match.start()) for match in _RESIDUAL.finditer(text)
    )


def _negation_cues(text: str, residual: bool = False) -> list[re.Match[str]]:
    cues = []
    for cue in _NEGATION.finditer(text):
        if _PSEUDO_NEGATION.match(text, cue.start()):
            continue
        if cue.group().lower().startswith("resol") and (
            residual or _PARTIAL_RESOLUTION.search(text, 0, cue.start())
        ):
            continue
        cues.append(cue)
    return cues


def _post_negation(post: str, residual: bool) -> re.Match[str] | None:
    match = _NEGATION_POST.match(post) or _ANSWER_NEGATIVE.match(post)
    if match is not None and residual and "resol" in match.group().lower():
        return None  # "emboli have resolved, with residual clot" is partial
    return match


def _negation_scope(gap: str, post: str, interior: str = "") -> str:
    """Whether a negation cue ``gap`` words before a mention reaches it.

    "negated", "broken" (a new affirmed finding started in between) or "ambiguous".
    ``interior`` is the mention's own gap text ("right ventricle <is> enlarged").
    """
    if len(gap.split()) > _MAX_SCOPE_WORDS:
        return "ambiguous"
    separators = list(_ITEM_SEPARATOR.finditer(gap))
    if not separators:
        return "negated"
    tail = gap[separators[-1].end():]
    if _PRESENCE_MODIFIER.match(tail):
        return "broken"
    # "No PE and the right ventricle is enlarged": unless the list is coordinated by or/nor
    # ("no A, B or C is seen"), an item with its own finite verb starts a new clause.
    if not _COORDINATION.search(gap) and (
        _FINITE_VERB.search(tail) or _FINITE_VERB.search(interior) or _FINITE_VERB_AFTER.match(post)
    ):
        return "broken"
    commas = [separator for separator in separators if separator.group() == ","]
    if not commas:
        return "negated"  # "no A and B" stays negated
    # "no A, B or C": a comma list is negated only when it is coordinated by or/nor.
    if _COORDINATION.search(gap[commas[0].end():]) or _COORDINATION.search(post):
        return "negated"
    return "ambiguous"


def _classify(
    context: _Context,
    match: re.Match[str],
    partial: Pattern[str] | None = None,
    episodic: bool = False,
) -> _Mention:
    """Affirmed, negated, uncertain or ignored, judged inside the mention's clause.

    ``gap`` (when the pattern has that group) is interior text between the two parts
    of a mention ("main pulmonary artery <is patent without> thrombus") and is checked
    like the pre-window. ``partial`` names qualifiers that restrict a negation to part
    of the finding ("no segmental pulmonary embolism" is not "no pulmonary embolism").
    ``episodic`` findings (PE) are history when "known" ("known PE").

    Historical mentions and exam titles/indications are ignored (no evidence either way)
    rather than uncertain, so "H/o PE. No pulmonary embolism." still resolves.
    """
    text = context.text
    start, end = match.start(), match.end()
    clause_start, clause_end = context.clause(start, end)
    pre = text[clause_start:start]
    post = text[end:clause_end]
    gap = (match.group("gap") or "") if "gap" in match.re.groupindex else ""
    historical_adj = _HISTORICAL_ADJ_EPISODIC if episodic else _HISTORICAL_ADJ
    if (
        _HISTORICAL.search(pre) or _HISTORICAL.search(gap) or historical_adj.search(pre)
        or _HISTORICAL_POST.match(post)
    ):
        return _Mention("ignored", start, end)
    if _INDICATION_ADJACENT.search(pre) or (
        start < context.preamble_end and (_SUSPICION.search(pre) or _SUSPICION_POST.match(post))
    ):
        return _Mention("ignored", start, end)
    if _UNCERTAIN.search(pre) or _UNCERTAIN.search(gap) or _UNCERTAIN_POST.match(post):
        return _Mention("uncertain", start, end)
    if _ANSWER_POSITIVE.match(post):
        return _Mention("affirmed", start, end)

    residual = _has_residual(post)
    negated_from: int | None = None
    negated_to = end
    post_cue = False
    if _negation_cues(gap, residual):
        negated_from = start
        post_cue = True
    elif (post_negation := _post_negation(post, residual)) is not None:
        negated_from, negated_to = start, end + post_negation.end()
        post_cue = True
    else:
        cues = _negation_cues(pre, residual)
        if cues:
            cue = cues[-1]
            scope = _negation_scope(pre[cue.end():], post, gap)
            if scope == "ambiguous":
                return _Mention("uncertain", start, end)
            if scope == "negated":
                negated_from = clause_start + cue.start()
    if negated_from is not None:
        if partial is not None and partial.search(text[negated_from:end]):
            return _Mention("ignored", start, end)
        # The words a negation covers: cue to mention, or for "Acute PE is not seen" the
        # adjectives right before the mention.
        covered = text[max(clause_start, start - 20):end] if post_cue else text[negated_from:end]
        acute_only = bool(_ACUTE.search(covered)) and not _CHRONIC.search(covered)
        return _Mention("negated", negated_from, negated_to, acute_only)
    if _INDICATION.search(pre) or _INDICATION_POST.match(post):
        return _Mention("ignored", start, end)
    return _Mention("affirmed", start, end)


# ---- target vocabularies --------------------------------------------------------------
_EMBOLUS = r"(?:thrombo)?embol(?:us|i|ism|isms)"
_PE_ABBREV = r"(?-i:PEs?)"
_CLOT = rf"(?:{_EMBOLUS}|{_PE_ABBREV}|thromb(?:us|i)|clots?|filling\s+defects?)"
_CLOT_MODIFIERS = (
    r"(?:(?:pulmonary|arterial|artery|acute|chronic|subacute|bilateral|occlusive|"
    r"nonocclusive|non-occlusive|partially|small|large)\s+)*"
)
_LEVEL = r"(?:central|lobar|segmental|subsegmental)"
# "lobar and segmental", "lobar, segmental, or subsegmental": one mention, several levels.
_LEVELS = (
    rf"(?P<levels>{_LEVEL}(?:\s*(?:,\s*(?:and\s+|or\s+)?|/|\band/or\b|\band\b|\bor\b|\bto\b|"
    rf"\bthrough\b|-)\s*{_LEVEL})*)"
)
_GAP = r"(?P<gap>[^.;,\n]{0,60}?)"
_LEVEL_BEFORE_CLOT = rf"\b{_LEVELS}\s+{_CLOT_MODIFIERS}{_CLOT}\b"
_CLOT_BEFORE_LEVEL = (
    rf"\b{_CLOT}\b{_GAP}\b{_LEVELS}\s+(?:(?:pulmonary|lower|upper|middle|lobe|right|left|"
    rf"bilateral)\s+)*(?:arter(?:y|ies|ial)|branch(?:es)?|vessels?|levels?)\b"
)
_MAIN_ARTERY = r"(?:main\s+(?:right\s+|left\s+)?pulmonary\s+arter(?:y|ies|ial)|pulmonary\s+trunk)"
_RATIO = (
    r"(?:rv\s*[/:-]\s*lv|right\s+vent(?:ricular|ricle)[- /]+(?:to[- ]+)?left\s+"
    r"vent(?:ricular|ricle))\s+(?:diameter\s+)?ratio"
)
_RATIO_GAP = r"(?P<gap>(?:[^.;\n]|(?<=\d)\.(?=\d)){0,25}?)"
# A stated bound on the ratio ("> 1", "less than 1.5"). The number is compared with 1.0
# in _bound_ok, so "< 1.5" (which allows 1.2) is no evidence either way.
_RATIO_NUMBER = r"\s*(?P<num>one|\d+(?:\.\d+)?)(?![\d.]*\d)"
_LOWER_BOUND = (
    r"(?:\b(?:greater|more|higher|larger)\s+than(?:\s+or\s+equal\s+to)?|\babove|"
    r"\bexceed(?:s|ing)?|>=?|\u2265)"
)
_UPPER_BOUND = (
    r"(?:\b(?:less|lower|smaller)\s+than(?:\s+or\s+equal\s+to)?|\bbelow|\bunder|<=?|\u2264)"
)


@dataclass(frozen=True)
class _TargetRules:
    positive: tuple[Pattern[str], ...]
    # Only evidence when negated ("no filling defect in the pulmonary arteries").
    negative_only: tuple[Pattern[str], ...] = ()
    # Affirmed statements of absence ("RV/LV ratio is normal").
    explicit_negative: tuple[Pattern[str], ...] = ()
    # For patterns with a ``levels`` group: the level word this target needs in it.
    keyword: str | None = None
    partial: Pattern[str] | None = None
    # The finding is the mention itself (any mention of an RV/LV ratio counts).
    mention_only: bool = False
    # An event rather than a standing finding: "known PE" is history (see _classify).
    episodic: bool = False
    # A negation that only covers acute PE ("No acute pulmonary embolism") is dropped when
    # the report affirms a clot elsewhere: an affirmed positive or negative_only mention,
    # or one of these chronic-clot descriptions.
    acute_limited: bool = False
    clot_evidence: tuple[Pattern[str], ...] = ()


_PE_PARTIAL = re.compile(
    r"\b(?:central|main|lobar|interlobar|segmental|subsegmental|saddle|proximal|large)\b", _FLAGS
)
_PE_RULES = _TargetRules(
    positive=_patterns(
        rf"\bpulmonary\s+(?:arter(?:y|ial)\s+)?{_EMBOLUS}\b",
        rf"\b{_PE_ABBREV}\b",
        rf"\b(?:saddle|central|lobar|segmental|subsegmental)\s+{_CLOT_MODIFIERS}{_EMBOLUS}\b",
        r"\bchronic\s+(?:pulmonary\s+)?thromboembolic\s+(?:disease|pulmonary\s+hypertension)\b",
    ),
    negative_only=_patterns(
        r"\b(?:(?:intraluminal\s+)?filling\s+defects?|thromb(?:us|i)|clots?)\b"
        r"(?P<gap>[^.;\n]{0,45}?)\bpulmonary\s+arter(?:y|ies|ial)\b",
    ),
    partial=_PE_PARTIAL,
    episodic=True,
    acute_limited=True,
    clot_evidence=_patterns(
        r"\bchronic\b(?P<gap>[^.;\n]{0,40}?)\b(?:thromb(?:us|i)|clots?|filling\s+defects?|webs?|"
        r"bands?|emboli|embolus|thromboembolic)\b",
        r"\b(?:intraluminal\s+)?(?:webs?|bands?)\b(?P<gap>[^.;\n]{0,30}?)\bpulmonary\s+arter",
    ),
)


def _location(keyword: str, *extra: str) -> _TargetRules:
    return _TargetRules(
        positive=_patterns(_LEVEL_BEFORE_CLOT, _CLOT_BEFORE_LEVEL, *extra), keyword=keyword,
        episodic=True,
    )


_TARGET_RULES: dict[str, _TargetRules] = {
    "pe_present": _PE_RULES,
    "saddle": _TargetRules(
        positive=_patterns(
            rf"\bsaddle\s+{_CLOT_MODIFIERS}(?:{_EMBOLUS}|{_PE_ABBREV}|thromb(?:us|i)|clots?)\b",
            rf"\b{_CLOT}\b(?P<gap>[^.;,\n]{{0,40}}?)\bsaddl(?:e|es|ing)\b",
        ),
        episodic=True,
    ),
    "central": _location(
        "central",
        rf"\b{_CLOT}\b{_GAP}\b{_MAIN_ARTERY}\b",
        rf"\b{_MAIN_ARTERY}\b{_GAP}\b{_CLOT}\b",
    ),
    "lobar": _location("lobar"),
    "segmental": _location("segmental"),
    "subsegmental": _location("subsegmental"),
    "rv_enlargement": _TargetRules(
        positive=_patterns(
            r"\bright\s+vent(?:ricle|ricular)\b(?P<gap>[^.;\n]{0,25}?)\b(?:enlarged|enlargement|"
            r"dilated|dilation|dilatation)\b",
            r"\b(?:enlarged|dilated)\s+right\s+ventricle\b",
            r"\b(?:rv|right\s+heart)\s+(?:enlargement|dilation|dilatation)\b",
            r"\b(?:enlargement|dilation|dilatation)\s+of\s+the\s+right\s+ventricle\b",
        )
    ),
    "rv_lv_ratio_mentioned": _TargetRules(positive=_patterns(rf"\b{_RATIO}\b"), mention_only=True),
    "rv_lv_ratio_abnormal": _TargetRules(
        positive=_patterns(
            rf"\b{_RATIO}{_RATIO_GAP}\b(?:elevated|abnormal|increased)\b",
            rf"\b{_RATIO}{_RATIO_GAP}{_LOWER_BOUND}{_RATIO_NUMBER}",
            rf"\b(?:elevated|abnormal|increased)\s+{_RATIO}\b",
        ),
        explicit_negative=_patterns(
            rf"\b{_RATIO}{_RATIO_GAP}\b(?:normal|within\s+normal\s+limits)\b",
            rf"\b{_RATIO}{_RATIO_GAP}{_UPPER_BOUND}{_RATIO_NUMBER}",
            rf"\bnormal\s+{_RATIO}\b",
        ),
    ),
    "septal_bowing": _TargetRules(
        positive=_patterns(
            r"\b(?:interventricular\s+)?septal\s+bowing\b",
            r"\bbowing\s+of\s+the\s+(?:interventricular\s+)?septum\b",
            r"\b(?:interventricular\s+)?septum\s+bow(?:s|ing)\b",
        )
    ),
    "contrast_reflux": _TargetRules(
        positive=_patterns(
            r"\b(?:contrast\s+)?reflux(?:es|ed)?\b(?P<gap>[^.;\n]{0,45}?)\b(?:ivc|inferior\s+vena\s+"
            r"cava|hepatic\s+veins?)\b",
            r"\breflux\s+of\s+contrast\b",
        ),
        negative_only=_patterns(r"\b(?:contrast\s+)?reflux\b"),
    ),
    "pleural_effusion": _TargetRules(
        positive=_patterns(
            r"\b(?P<levels>(?:pleural|pericardial)(?:\s*(?:,|/|\band\b|\bor\b)\s*(?:pleural|"
            r"pericardial))*)\s+effusions?\b"
        ),
        keyword="pleural",
    ),
    "pericardial_effusion": _TargetRules(
        positive=_patterns(
            r"\b(?P<levels>(?:pleural|pericardial)(?:\s*(?:,|/|\band\b|\bor\b)\s*(?:pleural|"
            r"pericardial))*)\s+effusions?\b"
        ),
        keyword="pericardial",
    ),
    "malignancy_related_finding": _TargetRules(
        positive=_patterns(
            r"\b(?:metastatic\s+(?:disease|lesions?|nodules?|deposits?|(?:lymph)?adenopathy)|"
            r"metastas[ie]s|known\s+malignancy|malignant\s+(?:mass|neoplasm|lesions?|tumou?r|"
            r"nodules?|effusion))\b",
            r"\b(?:mass|lesion|nodule|tumou?r)\b(?P<gap>[^.;\n]{0,40}?)\b(?:suspicious\s+for|"
            r"consistent\s+with|concerning\s+for|worrisome\s+for)\s+(?:malignan(?:cy|t)|"
            r"metasta\w*|neoplasm|carcinoma)\b",
        ),
        negative_only=_patterns(r"\bmalignan(?:cy|t\s+disease)\b"),
    ),
    "chronic_lung_disease": _TargetRules(
        positive=_patterns(
            r"\bchronic\s+(?:obstructive\s+)?(?:lung|pulmonary)\s+disease\b",
            r"\bcopd\b",
        )
    ),
    "fibrosis": _TargetRules(
        positive=_patterns(
            r"\b(?:pulmonary|interstitial|lung)\s+fibrosis\b",
            r"\bfibrotic\s+(?:lung\s+)?(?:changes?|disease|scarring)\b",
        ),
        negative_only=_patterns(r"\bfibrosis\b"),
    ),
    "emphysema": _TargetRules(
        # Subcutaneous/mediastinal emphysema is air in soft tissue, not lung disease.
        positive=_patterns(
            r"(?<!subcutaneous )(?<!mediastinal )(?<!soft tissue )(?<!chest wall )"
            r"\bemphysema(?:tous)?\b"
        )
    ),
}

# "RV/LV ratio is 1.2", "RV/LV ratio of 1.2", and the bare "Heart: RV/LV ratio 1.4."
_RATIO_VALUE = re.compile(
    rf"\b{_RATIO}\s*(?:(?:is|=|:|of|measures?|measuring)\s*)?(?:approximately\s+)?"
    r"([0-9]+(?:\.[0-9]+)?)\b",
    _FLAGS,
)
_ACUITY_TERM = (
    rf"(?:(?:pulmonary\s+)?(?:arter(?:y|ial)\s+)?{_EMBOLUS}|{_PE_ABBREV}|thromboembolic\s+disease|"
    # "Chronic thrombus in the right main pulmonary artery" describes the PE's acuity too.
    r"(?:thromb(?:us|i)|clots?)(?=[^.;\n]{0,45}?\bpulmonary\s+(?:arter|trunk)))"
)
# Up to six words between the acuity and the embolus ("acute right lower lobe pulmonary
# emboli"), drawn from a closed list of location/extent words, so an acuity of another
# noun ("chronic lung disease with acute PE") does not reach the embolus. The words are
# checked for negation and hedging like any interior gap.
_ACUITY_WORD = (
    r"(?:and|or|on|superimposed|appearing|likely|probably|predominantly|mostly|acute|subacute|"
    r"chronic|right|left|bilateral|unilateral|upper|middle|lower|lobes?|lobar|interlobar|"
    r"segmental|subsegmental|central|main|saddle|proximal|distal|peripheral|tiny|small|large|"
    r"massive|extensive|multiple|multifocal|scattered|few|several|numerous|occlusive|"
    r"nonocclusive|non-occlusive|partially|completely|new|additional|lingular|rll|rml|rul|lll|"
    r"lul|mild|moderate|small-volume|large-volume|volume)"
)
_ACUITY_FILLER = rf"(?P<gap>(?:(?:\s*,\s*|\s+|-){_ACUITY_WORD}(?![\w-])){{0,6}}?)"
_ACUITY_PATTERNS: tuple[tuple[str, Pattern[str]], ...] = (
    ("acute_on_chronic", re.compile(
        rf"\bacute(?:[- ]on[- ]|\s+and\s+)chronic{_ACUITY_FILLER}\s+{_ACUITY_TERM}\b", _FLAGS)),
    ("acute", re.compile(rf"\b(?:sub)?acute{_ACUITY_FILLER}\s+{_ACUITY_TERM}\b", _FLAGS)),
    ("chronic", re.compile(rf"\bchronic{_ACUITY_FILLER}\s+{_ACUITY_TERM}\b", _FLAGS)),
    ("uncertain", re.compile(
        rf"\b(?:age[- ]indeterminate|indeterminate[- ](?:age|acuity)){_ACUITY_FILLER}\s+"
        rf"{_ACUITY_TERM}\b", _FLAGS)),
    ("uncertain", re.compile(
        rf"\b{_ACUITY_TERM}\b(?P<gap>[^.;\n]{{0,30}}?)\bof\s+(?:indeterminate|uncertain)\s+"
        r"(?:age|acuity|chronicity)\b", _FLAGS)),
)


def _span(original: str, start: int, end: int) -> EvidenceSpan:
    return EvidenceSpan(text=original[start:end], start=start, end=end)


def _dedupe(matches: list[re.Match[str]]) -> list[re.Match[str]]:
    """Overlapping hits of one target's patterns are one mention: keep the leftmost-longest."""
    kept: list[re.Match[str]] = []
    # Sorted by start, so a hit overlaps a kept one exactly when it starts before the last
    # kept hit ends (linear rather than comparing against every kept hit).
    for match in sorted(matches, key=lambda item: (item.start(), -item.end())):
        if not kept or match.start() >= kept[-1].end():
            kept.append(match)
    return kept


def _keyword_ok(match: re.Match[str], keyword: str | None) -> bool:
    if keyword is None or "levels" not in match.re.groupindex:
        return True
    return re.search(rf"\b{keyword}\b", match.group("levels"), _FLAGS) is not None


def _bound_ok(match: re.Match[str], upper: bool) -> bool:
    """Whether a stated RV/LV bound decides "ratio > 1": "> 1.2" and "< 0.9" do, "< 1.5" does not."""
    if "num" not in match.re.groupindex:
        return True
    raw = match.group("num").lower()
    value = 1.0 if raw == "one" else float(raw)
    return value <= 1.0 if upper else value >= 1.0


def _boolean_decision(
    original: str, context: _Context, target: str, rules: _TargetRules
) -> RuleDecision:
    text = context.text
    found = _dedupe([
        match
        for pattern in rules.positive
        for match in pattern.finditer(text)
        if _keyword_ok(match, rules.keyword) and _bound_ok(match, upper=False)
    ])
    if rules.mention_only:
        if found:
            return RuleDecision(
                True, True, f"explicit_positive_{target}", _span(original, found[0].start(), found[0].end())
            )
        return RuleDecision(False, None, f"no_explicit_evidence_{target}")

    found_starts = [match.start() for match in found]

    def outside(match: re.Match[str]) -> bool:
        # ``found`` is sorted and non-overlapping: only its neighbours can overlap ``match``.
        index = bisect_left(found_starts, match.end())
        return index == 0 or found[index - 1].end() <= match.start()

    positive: list[_Mention] = []
    negative: list[_Mention] = []
    uncertain: list[_Mention] = []
    # Affirmed clot descriptions that are not positive evidence on their own.
    clots: list[_Mention] = []
    for match in found:
        mention = _classify(context, match, rules.partial, rules.episodic)
        {"affirmed": positive, "negated": negative, "uncertain": uncertain}.get(mention.status, []).append(mention)
    for match in _dedupe([m for p in rules.negative_only for m in p.finditer(text) if outside(m)]):
        mention = _classify(context, match, rules.partial, rules.episodic)
        {"negated": negative, "uncertain": uncertain, "affirmed": clots}.get(mention.status, []).append(mention)
    for match in _dedupe([
        m for p in rules.explicit_negative for m in p.finditer(text)
        if outside(m) and _bound_ok(m, upper=True)
    ]):
        mention = _classify(context, match)
        # "the ratio is not normal" is a double negative: no explicit answer.
        {"affirmed": negative, "negated": uncertain, "uncertain": uncertain}.get(mention.status, []).append(mention)
    if rules.acute_limited and any(mention.acute_only for mention in negative):
        clots += [
            mention
            for pattern in rules.clot_evidence
            for match in pattern.finditer(text)
            if (mention := _classify(context, match, None, rules.episodic)).status == "affirmed"
        ]
        # "No acute PE. Chronic thrombus in the right main pulmonary artery.": the negation
        # leaves the chronic clot standing, so it is not evidence of "no PE".
        if positive or clots:
            negative = [mention for mention in negative if not mention.acute_only]

    if uncertain:
        first = uncertain[0]
        return RuleDecision(False, None, f"context_uncertain_{target}", _span(original, first.start, first.end))
    if positive and negative:
        first = min(positive + negative, key=lambda mention: mention.start)
        return RuleDecision(False, None, f"conflicting_explicit_{target}", _span(original, first.start, first.end))
    if positive:
        return RuleDecision(
            True, True, f"explicit_positive_{target}", _span(original, positive[0].start, positive[0].end)
        )
    if negative:
        return RuleDecision(
            True, False, f"explicit_negative_{target}", _span(original, negative[0].start, negative[0].end)
        )
    return RuleDecision(False, None, f"no_explicit_evidence_{target}")


def _acuity_decision(original: str, context: _Context) -> RuleDecision:
    affirmed: list[tuple[str, _Mention]] = []
    for value, pattern in _ACUITY_PATTERNS:
        for match in pattern.finditer(context.text):
            mention = _classify(context, match, episodic=True)
            if mention.status == "uncertain":
                return RuleDecision(
                    False, None, f"context_uncertain_acuity:{value}", _span(original, mention.start, mention.end)
                )
            if mention.status == "affirmed":
                affirmed.append((value, mention))
    # A negated acuity ("no evidence of acute pulmonary embolism") is not an acuity.
    if not affirmed:
        return RuleDecision(False, None, "acuity_unresolved")
    values = {value for value, _ in affirmed}
    if "acute_on_chronic" in values:
        values.discard("acute")
        values.discard("chronic")
    if len(values) != 1:
        first = affirmed[0][1]
        return RuleDecision(False, None, "conflicting_explicit_acuity", _span(original, first.start, first.end))
    value = next(iter(values))
    mention = next(mention for candidate, mention in affirmed if candidate == value)
    return RuleDecision(True, value, f"explicit_{value}", _span(original, mention.start, mention.end))


def _ratio_value_decision(original: str, context: _Context) -> RuleDecision:
    matches = []
    for match in _RATIO_VALUE.finditer(context.text):
        status = _classify(context, match).status
        if status in {"uncertain", "negated"}:
            return RuleDecision(
                False, None, "context_uncertain_rv_lv_ratio_value", _span(original, match.start(), match.end())
            )
        if status == "affirmed":
            matches.append(match)
    if not matches:
        return RuleDecision(False, None, "rv_lv_ratio_value_unresolved")
    values = {float(match.group(1)) for match in matches}
    if len(values) != 1:
        return RuleDecision(
            False, None, "conflicting_rv_lv_ratio_values", _span(original, matches[0].start(), matches[0].end())
        )
    value = next(iter(values))
    return RuleDecision(
        True, value, "explicit_numeric_rv_lv_ratio", _span(original, matches[0].start(), matches[0].end())
    )


def apply_rule(report: str, target: str) -> RuleDecision:
    """Explicit regex decision for one target; unresolved (abstain) whenever in doubt.

    A target resolves only when every mention in the finding sections agrees: one
    hedged mention, or an affirmed and a negated mention together, leaves it unresolved.
    """
    canonical = canonical_target(target)
    text = str(report)
    if not text.strip():
        return RuleDecision(False, None, "empty_report")
    context = _Context(*_finding_text(text))
    if canonical == "acuity":
        return _acuity_decision(text, context)
    if canonical == "rv_lv_ratio_value":
        return _ratio_value_decision(text, context)
    return _boolean_decision(text, context, canonical, _TARGET_RULES[canonical])
