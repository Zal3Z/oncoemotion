"""Model-as-classifier: constrained scoring of the PRO-CTCAE terms at point E.

The deterministic mapper (``mapping/pipeline.py``) never sees the model's internal
state, so role/emotion cannot affect its label. To ask *"does emotionality change
what the model would code?"* we let the MODEL choose the term: after the identical
teacher-forced prefix ``{"pro_ctcae":{"term":"`` we score every candidate PRO term
by its length-normalised continuation log-probability and take the arg-max.

This is a measurement of the model's committed term, independent of and comparable
to the safe deterministic mapper. Nothing here changes the clinical mapping.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Candidate:
    canonical_id: str
    term_en: str
    surfaces: list[str] = field(default_factory=list)          # surface strings
    surface_ids: list[list[int]] = field(default_factory=list)  # tokenised


@dataclass
class GenPrediction:
    generated: str            # raw generated continuation
    term_str: str             # extracted term (up to the closing quote)
    top1_id: str | None       # mapped PRO id (None if no confident map)
    top1_term: str | None
    map_score: float          # fuzzy match score of term_str to the mapped term
    logprob: float            # mean token log-prob of the generated term (confidence)
    matched: bool             # map_score >= floor and a term was mapped
    kind: str                 # mapped | abstained | non_answer | unmapped


# Explicit "no PRO term" markers a model emits when it (correctly) abstains.
# The multi-word Italian forms were missing, so a model declining properly -- "nessun
# evento avverso correlato al trattamento" -- was counted as a mapping failure. On
# the real corpus those phrases alone accounted for several hundred rows.
ABSTAIN_MARKERS = {
    "n/a", "na", "n.a.", "none", "no", "nan", "null", "nulla", "niente", "nessuno",
    "nessun sintomo", "nessun termine", "non applicabile", "no applicabile",
    "non specificato", "non pertinente", "sconosciuto", "unknown", "-", "0", "",
    "nessun evento avverso", "nessun evento avverso rilevante",
    "nessun evento avverso correlato al trattamento", "nessun effetto collaterale",
    "nessuna reazione", "nessun disturbo", "nessun problema", "assente",
    "non riferito", "non valutabile", "no adverse event", "no symptoms",
    "not applicable", "not specified", "not reported", "nessuna",
}

# Codes, identifier fragments and digit runs are not abstentions: the model failed
# to produce a clinical answer.  Keep this separate from a deliberate refusal to
# code, otherwise a real failure mode disappears inside the generic "unmapped"
# bucket.
NON_ANSWER_PATTERN = re.compile(
    r"^(?:[\d\s.\-_/]+|[A-Z]\d{2}(?:\.\d+)?|CTCAE[_\s].*|PRO[_\s]?\d+)$", re.I
)

# Clinical and technical surfaces that ARE PRO-CTCAE concepts under another name.
# Taken from what the models actually emitted and the fuzzy matcher then discarded:
# it answered "Disfagia", which IS PRO_002 "difficolta a deglutire", and was scored
# wrong. Across 13 models this silently threw away 22-59% of answers (median ~30%),
# and the accuracy metric was largely measuring the surface list's vocabulary.
CLINICAL_SUPPLEMENT = {
    "PRO_001": ["xerostomia", "secchezza delle fauci", "dry mouth"],
    "PRO_002": ["disfagia", "dysphagia", "difficolta di deglutizione", "swallowing difficulty"],
    "PRO_003": ["mucosite", "mucosite orale", "mucositis", "oral mucositis", "stomatite",
                "afte", "stomatitis"],
    "PRO_005": ["alterazione della voce", "voice alteration", "disfonia", "dysphonia"],
    "PRO_006": ["raucedine", "hoarseness"],
    "PRO_007": ["disgeusia", "dysgeusia", "gusto metallico", "gusto alterato",
                "alterazione del gusto", "metallic taste", "taste alteration", "ageusia"],
    "PRO_008": ["anoressia", "anorexia", "appetito ridotto", "inappetenza",
                "riduzione dell'appetito", "loss of appetite", "decreased appetite"],
    "PRO_012": ["flatulenza", "meteorismo", "flatulence"],
    "PRO_013": ["distensione addominale", "gonfiore addominale", "abdominal distension",
                "meteorismo addominale"],
    "PRO_016": ["diarrhoea", "diarrea", "alvo diarroico"],
    "PRO_017": ["dolore addominale", "abdominal pain", "algia addominale"],
    "PRO_019": ["dispnea", "dyspnoea", "dyspnea", "affanno", "fame d'aria"],
    "PRO_022": ["edema", "edema periferico", "peripheral edema", "tumefazione",
                "gonfiore degli arti", "linfedema", "oedema"],
    "PRO_023": ["tachicardia", "palpitazioni", "cardiopalmo", "tachycardia", "palpitations"],
    "PRO_024": ["eritema", "erythema", "esantema", "rash cutaneo"],
    "PRO_025": ["xerosi", "xerosi cutanea", "secchezza cutanea", "dry skin"],
    "PRO_027": ["alopecia", "alopecia da chemioterapia"],
    "PRO_028": ["prurito", "pruritus", "itching"],
    "PRO_030": ["sindrome mano-piede", "hand-foot syndrome", "eritrodisestesia palmo-plantare"],
    "PRO_039": ["neuropatia periferica", "peripheral neuropathy", "parestesie",
                "paresthesia", "neuropatia sensitiva", "formicolio"],
    "PRO_040": ["vertigini", "vertigine", "capogiri", "dizziness", "instabilita posturale"],
    "PRO_041": ["visione offuscata", "offuscamento visivo", "visual disturbance",
                "visual disturbances", "blurred vision", "annebbiamento visivo"],
    "PRO_044": ["epifora", "lacrimazione", "watery eyes"],
    "PRO_045": ["tinnito", "tinnitus", "acufene", "acufeni", "ronzio auricolare"],
    "PRO_046": ["deficit di concentrazione", "cognitive impairment", "difficolta di concentrazione"],
    "PRO_047": ["deficit mnesico", "memory impairment", "amnesia"],
    "PRO_048": ["dolore", "pain", "algia", "dolore generalizzato"],
    "PRO_049": ["cefalea", "headache", "emicrania", "migraine"],
    "PRO_050": ["mialgia", "mialgie", "myalgia", "dolore muscolare"],
    "PRO_051": ["artralgia", "artralgie", "arthralgia", "dolore articolare"],
    "PRO_052": ["insonnia", "insomnia", "disturbi del sonno"],
    "PRO_053": ["astenia", "asthenia", "fatigue", "affaticamento", "spossatezza"],
    "PRO_054": ["ansia", "stato ansioso", "anxiety"],
    "PRO_056": ["depressione", "umore depresso", "depression"],
    "PRO_061": ["disuria", "dysuria", "stranguria"],
    "PRO_062": ["urgenza minzionale", "urinary urgency"],
    "PRO_063": ["pollachiuria", "urinary frequency"],
    "PRO_065": ["incontinenza urinaria", "urinary incontinence"],
    "PRO_073": ["ecchimosi", "ematoma", "bruising", "lividi"],
    "PRO_074": ["brividi", "chills"],
    "PRO_075": ["iperidrosi", "hyperhidrosis", "sudorazione profusa", "sudorazione notturna"],
    "PRO_077": ["vampate", "vampate di calore", "hot flushes", "hot flashes"],
    "PRO_078": ["epistassi", "epistaxis", "sanguinamento nasale"],
}


def _fuzzy_best(term: str, surfaces: list[str]) -> float:
    """Best fuzzy score of ``term`` vs a list of surfaces (0..1).

    Uses rapidfuzz WRatio (handles morphological/partial variants such as
    "Anxiety" vs "Anxious") when available, else difflib.
    """
    try:
        from rapidfuzz import fuzz
        return max((fuzz.WRatio(term, s) for s in surfaces), default=0.0) / 100.0
    except Exception:
        from difflib import SequenceMatcher
        return max((SequenceMatcher(None, term, s).ratio() for s in surfaces), default=0.0)


# English lay/noun synonyms not in the terminology (whose canonical_english is an
# adjective, e.g. "Anxious"), for models that answer in English. Focused on the
# affect terms central to this study; other terms rely on the library surfaces.
EN_SUPPLEMENT = {
    "PRO_054": ["anxiety", "nervousness", "worry", "worried", "nervous"],
    "PRO_055": ["discouragement", "hopelessness", "demoralization", "hopeless"],
    "PRO_056": ["sadness", "depression", "depressed mood", "low mood", "unhappy"],
    "PRO_053": ["tiredness", "exhaustion", "tired", "lack of energy", "weakness"],
}


def build_term_matcher(library):
    """Return ``match(term_str) -> (canonical_id|None, term_en|None, score)``.

    Maps a free-text term the model GENERATED (e.g. "nausea", "ansia", "Anxiety",
    "Fatigue") to a canonical PRO id by fuzzy-matching against ALL library surface
    forms (official Italian, canonical English, reviewed/auto/synthetic synonyms,
    patient phrases) plus a small English affect supplement. Explicit abstention
    markers ("N/A", "None", ...) return no match.
    """
    per_term = []  # (canonical_id, term_en, [surfaces])
    for t in library:
        surf = [e.surface.lower() for e in t.match_entries()]
        if t.official_italian_labels:
            surf += _italian_surfaces(t.official_italian_labels[0])
        surf += EN_SUPPLEMENT.get(t.canonical_id, [])
        surf += CLINICAL_SUPPLEMENT.get(t.canonical_id, [])
        surf = [s.strip() for s in dict.fromkeys(surf) if s and s.strip()]
        per_term.append((t.canonical_id, t.canonical_english, surf))

    def match(term_str: str):
        term = (term_str or "").lower().strip().strip('.,;:"\'')
        if term in ABSTAIN_MARKERS or len(term) < 2:
            return (None, None, 0.0)
        best = (None, None, 0.0)
        for cid, en, surf in per_term:
            r = _fuzzy_best(term, surf)
            if r > best[2]:
                best = (cid, en, r)
        return best

    return match


def classify_generated_term(term_str: str, matcher, floor: float = 0.72):
    """Classify a free-generated answer without conflating four outcomes.

    Returns ``(canonical_id, kind, score)`` where ``kind`` is one of ``mapped``,
    ``abstained``, ``non_answer`` or ``unmapped``.  This function is shared by the
    GPU run and the offline legacy-row rescoring path, so their semantics cannot
    drift apart again.
    """
    term = (term_str or "").strip()
    folded = term.lower().strip('.,;:"\' ')
    if not term or NON_ANSWER_PATTERN.fullmatch(term):
        return None, "non_answer", 0.0
    if folded in ABSTAIN_MARKERS:
        return None, "abstained", 0.0
    canonical_id, _term_en, score = matcher(term)
    if canonical_id is not None and score >= floor:
        return canonical_id, "mapped", float(score)
    return None, "unmapped", float(score)


def predict_generative(adapter, prefix_ids, matcher, max_new_tokens: int = 10,
                       floor: float = 0.72) -> GenPrediction:
    """Greedy-generate the term after point E, then map it to a PRO id.

    Faithful to what the model *commits to* under the current role/ablation, and
    naturally abstains when the generated term maps to nothing (score < floor).
    """
    import torch

    model = adapter.model
    tok = adapter.tokenizer
    with torch.no_grad():
        out = model.generate(
            prefix_ids, max_new_tokens=max_new_tokens, do_sample=False,
            temperature=None, top_p=None, output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id),
        )
    gen = out.sequences[0][prefix_ids.shape[1]:]
    text = tok.decode(gen, skip_special_tokens=True)
    # term = text up to the first closing quote or brace
    term_str = text.split('"')[0].split('}')[0].strip().strip(',').strip()
    # confidence: mean log-prob over the term's own tokens
    n_term_tok = max(1, len(tok(term_str, add_special_tokens=False).input_ids)) if term_str else 1
    lps = []
    for i, logits in enumerate(out.scores[:n_term_tok]):
        lp = torch.log_softmax(logits[0].float(), dim=-1)
        lps.append(float(lp[gen[i]]))
    mean_lp = float(np.mean(lps)) if lps else 0.0
    cid, kind, ms = classify_generated_term(term_str, matcher, floor=floor)
    matched = kind == "mapped"
    en = None
    if matched:
        # ``matcher`` also supplies the canonical display term.  Calling it again
        # is cheap and keeps the public classifier's return value minimal.
        _cid, en, _score = matcher(term_str)
    return GenPrediction(
        generated=text.strip(), term_str=term_str,
        top1_id=cid if matched else None, top1_term=en if matched else None,
        map_score=round(ms, 3), logprob=round(mean_lp, 3), matched=matched,
        kind=kind)


@dataclass
class LabelPrediction:
    top1_id: str
    top1_term: str
    top1_score: float          # length-normalised mean log-prob of the term
    margin: float              # top1 - top2 (concept-level)
    softmax_top1: float        # softmax over concept scores (confidence proxy)
    entropy: float             # entropy of the concept softmax
    ranking: list[tuple]       # [(id, term, score), ...] top-k
    concept_scores: dict       # canonical_id -> best (max over surfaces) score


def _italian_surfaces(label: str, max_words: int = 5) -> list[str]:
    """Short, natural lowercase surface forms from an official PRO label.

    The model, prompted in Italian, emits a SHORT lowercase term after the forced
    ``"term":"`` (empirically e.g. ``nausea``). Official labels are long/UPPERCASE,
    so we derive: the head before the first comma (parentheticals removed) AND the
    parenthetical content itself (often the common word, e.g. ``diarrea`` in
    "FECI MOLLI O ACQUOSE (DIARREA)"). Forms longer than ``max_words`` are dropped.
    """
    lab = label.strip().lower()
    paren = re.findall(r"\(([^)]+)\)", lab)
    head = re.sub(r"\([^)]*\)", "", lab).split(",")[0].strip()
    out = [head] + [p.strip() for p in paren]
    return [f for f in out if f and len(f.split()) <= max_words]


def build_candidates(adapter, library, forms=("it", "en", "phrase")) -> list[Candidate]:
    """Candidate surface strings per PRO term, pre-tokenised (all lowercase).

    The concept score is the MAX over its variants, so the classifier is robust to
    whether the model emits the Italian term, the English canonical, or a common
    patient phrase. ``forms`` selects which families to include.
    """
    tok = adapter.tokenizer
    cands: list[Candidate] = []
    for t in library:
        surfaces: list[str] = []
        if "it" in forms and t.official_italian_labels:
            surfaces += _italian_surfaces(t.official_italian_labels[0])
        if "en" in forms and t.canonical_english:
            surfaces.append(t.canonical_english.lower())
        if "phrase" in forms and t.common_patient_phrases:
            surfaces.append(t.common_patient_phrases[0].strip().lower())
        # de-dup while keeping order
        seen, uniq = set(), []
        for s in surfaces:
            if s and s not in seen:
                seen.add(s); uniq.append(s)
        ids = [tok(s, add_special_tokens=False).input_ids for s in uniq]
        cands.append(Candidate(t.canonical_id, t.canonical_english, uniq, ids))
    return cands


def _score_surface_ids_cached(adapter, prefix_ids, flat_ids: list[list[int]],
                              chunk: int) -> list[float] | None:
    """Same scores as :func:`_score_surface_ids`, with the prompt computed once.

    Every candidate is scored after the *identical* prefix, so recomputing it for
    each one is the whole cost: 163 surfaces after a 93-token prompt is 15k tokens
    of forward pass per item instead of 93 + 163*3. Measured at 164s per item on a
    small GPU, which makes constrained scoring unusable on a real corpus.

    Returns None if the model's cache does not expand cleanly, so the caller can
    fall back to the slow path rather than risk different numbers.
    """
    import torch

    model = adapter.model
    dev = adapter.device
    tok = adapter.tokenizer
    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id or 0
    prefix = prefix_ids[0]
    T = int(prefix.shape[0])

    try:
        with torch.no_grad():
            base = model(input_ids=prefix.unsqueeze(0),
                         attention_mask=torch.ones(1, T, device=dev),
                         use_cache=True)
        past = base.past_key_values
        last_logits = base.logits[:, -1, :].float()          # predicts the 1st cand token
    except Exception:
        return None

    def _expand(pkv, b):
        """Repeat a batch-1 cache to batch b, across the cache APIs in the wild."""
        try:
            import copy
            c = copy.deepcopy(pkv)
            if hasattr(c, "batch_repeat_interleave"):
                c.batch_repeat_interleave(b)
                return c
            if hasattr(c, "key_cache"):
                c.key_cache = [k.expand(b, *k.shape[1:]).contiguous() for k in c.key_cache]
                c.value_cache = [v.expand(b, *v.shape[1:]).contiguous() for v in c.value_cache]
                return c
            return tuple(tuple(t.expand(b, *t.shape[1:]).contiguous() for t in layer)
                         for layer in pkv)
        except Exception:
            return None

    out: list[float] = []
    for i in range(0, len(flat_ids), chunk):
        batch = [list(c) if c else [pad_id] for c in flat_ids[i:i + chunk]]
        clens = [len(c) for c in batch]
        maxc, B = max(clens), len(batch)
        cand = torch.tensor([c + [pad_id] * (maxc - len(c)) for c in batch],
                            device=dev, dtype=prefix.dtype)
        pkv = _expand(past, B)
        if pkv is None:
            return None
        attn = torch.cat([torch.ones(B, T, device=dev),
                          (torch.arange(maxc, device=dev).unsqueeze(0)
                           < torch.tensor(clens, device=dev).unsqueeze(1)).float()], dim=1)
        try:
            with torch.no_grad():
                res = model(input_ids=cand, attention_mask=attn,
                            past_key_values=pkv, use_cache=False)
        except Exception:
            return None
        # logit for candidate token j comes from the prefix (j=0) or from position j-1
        logits = torch.cat([last_logits.unsqueeze(1).expand(B, 1, -1),
                            res.logits[:, :maxc - 1, :].float()], dim=1) if maxc > 1 \
            else last_logits.unsqueeze(1).expand(B, 1, -1)
        logp = torch.log_softmax(logits, dim=-1)
        tok_lp = logp.gather(-1, cand.unsqueeze(-1)).squeeze(-1)
        mask = (torch.arange(maxc, device=dev).unsqueeze(0)
                < torch.tensor(clens, device=dev).unsqueeze(1)).float()
        clen_t = torch.tensor(clens, device=dev, dtype=torch.float32)
        out.extend(((tok_lp * mask).sum(1) / clen_t).detach().cpu().tolist())
    return out


def _score_surface_ids(adapter, prefix_ids, flat_ids: list[list[int]], chunk: int) -> list[float]:
    """Mean per-token continuation log-prob for each id-list after ``prefix_ids``."""
    import torch

    model = adapter.model
    dev = adapter.device
    tok = adapter.tokenizer
    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id or 0
    prefix = prefix_ids[0]                      # [T]
    T = int(prefix.shape[0])
    out_scores: list[float] = []
    for i in range(0, len(flat_ids), chunk):
        batch = flat_ids[i:i + chunk]
        clens = [max(1, len(c)) for c in batch]
        maxc = max(clens)
        seqs, attns = [], []
        for c in batch:
            c = list(c) if c else [pad_id]
            pad = maxc - len(c)
            seqs.append(torch.cat([prefix,
                                   torch.tensor(c + [pad_id] * pad, device=dev, dtype=prefix.dtype)]))
            attns.append(torch.cat([torch.ones(T + len(c), device=dev),
                                    torch.zeros(pad, device=dev)]))
        input_ids = torch.stack(seqs)           # [B, T+maxc]
        attn = torch.stack(attns)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attn,
                           use_cache=False).logits
        # positions T-1 .. T-2+maxc predict candidate tokens at T .. T-1+maxc
        sub = logits[:, T - 1:T - 1 + maxc, :].float()
        logp = torch.log_softmax(sub, dim=-1)   # [B, maxc, V]
        targets = input_ids[:, T:T + maxc]       # [B, maxc]
        tok_lp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, maxc]
        pos = torch.arange(maxc, device=dev).unsqueeze(0)
        clen_t = torch.tensor(clens, device=dev).unsqueeze(1)
        mask = (pos < clen_t).float()
        mean_lp = (tok_lp * mask).sum(1) / clen_t.squeeze(1).float()
        out_scores.extend(mean_lp.detach().cpu().tolist())
    return out_scores


def predict_label(adapter, prefix_ids, candidates: list[Candidate],
                  chunk: int = 32, top_k: int = 5) -> LabelPrediction:
    """Score all candidate terms at point E and return the model's committed term."""
    # flatten all surfaces, remember which concept each belongs to
    flat_ids, owner = [], []
    for ci, cand in enumerate(candidates):
        for sid in cand.surface_ids:
            flat_ids.append(sid); owner.append(ci)
    # cached prefix first; the slow path stays as the reference and the fallback
    surf_scores = _score_surface_ids_cached(adapter, prefix_ids, flat_ids, chunk)
    if surf_scores is None:
        surf_scores = _score_surface_ids(adapter, prefix_ids, flat_ids, chunk)

    concept_best = [-1e30] * len(candidates)
    for s, ci in zip(surf_scores, owner):
        if s > concept_best[ci]:
            concept_best[ci] = s
    scores = np.array(concept_best, dtype=float)
    order = np.argsort(-scores)
    ranking = [(candidates[i].canonical_id, candidates[i].term_en, float(scores[i]))
               for i in order[:top_k]]
    # softmax over concept scores as a confidence proxy
    z = scores - scores.max()
    p = np.exp(z); p = p / p.sum()
    ent = float(-(p * np.log(np.clip(p, 1e-12, None))).sum())
    i0 = int(order[0]); i1 = int(order[1]) if len(order) > 1 else i0
    return LabelPrediction(
        top1_id=candidates[i0].canonical_id,
        top1_term=candidates[i0].term_en,
        top1_score=float(scores[i0]),
        margin=float(scores[i0] - scores[i1]),
        softmax_top1=float(p[i0]),
        entropy=ent,
        ranking=ranking,
        concept_scores={candidates[i].canonical_id: float(scores[i]) for i in range(len(candidates))},
    )
