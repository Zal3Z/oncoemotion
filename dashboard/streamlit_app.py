"""oncoemotion dashboard (spec section 15).

Panels:
  * Clinical mapping (always): PRO candidates, CTCAE fallback, abstention,
    evidence spans, safety flag — runs the baseline mapper, NEVER steering.
  * Emotion analysis (research, optional): point-E emotion-like scores, an
    emotion x layer heatmap, and a baseline-vs-steering comparison. Requires the
    ML stack + built vectors; degrades gracefully otherwise.

Run:  streamlit run dashboard/streamlit_app.py

IMPORTANT: emotion scores are 'emotion-like internal representations', NOT
diagnoses and NOT conscious emotions. This is a research/support tool.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oncoemotion.factory import build_default_mapper  # noqa: E402
from oncoemotion.schemas import MapRequest  # noqa: E402

st.set_page_config(page_title="oncoemotion", layout="wide")


@st.cache_resource
def get_mapper():
    return build_default_mapper()


@st.cache_resource
def get_model_and_vectors():
    """Load the open-weight model + emotion vectors (heavy; research only)."""
    import json
    from oncoemotion.config import ModelConfig
    from oncoemotion.models.base import load_adapter

    vec_path = ROOT / "outputs/checkpoints/emotion_vectors.npz"
    val_path = ROOT / "outputs/reports/vector_validation.json"
    if not vec_path.exists():
        return None
    V = np.load(vec_path, allow_pickle=True)
    val = json.loads(val_path.read_text(encoding="utf-8")) if val_path.exists() else {"concepts": {}}
    adapter = load_adapter(None, ModelConfig())
    adapter.load()
    return {"adapter": adapter, "V": V, "val": val}


def render_mapping(mapper, text: str):
    resp = mapper.map(MapRequest(record_id="dash", text=text))
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("PRO-CTCAE")
        st.metric("status", resp.pro_ctcae.status)
        for p in resp.pro_ctcae.predictions:
            st.write(f"**{p.term}** (`{p.canonical_id}`) — p={p.probability:.2f}")
            st.caption(f"attributi: {', '.join(p.applicable_attributes)} · evidenza: {p.evidence_spans} · {p.explanation}")
        if resp.abstain:
            st.warning(f"Astensione: {resp.abstention_reason}")
    with c2:
        st.subheader("CTCAE fallback")
        st.metric("status", resp.ctcae.status)
        for p in resp.ctcae.predictions:
            st.write(f"**{p.term}** — p={p.probability:.2f}")
        st.subheader("Safety")
        if resp.safety.urgent_human_review:
            st.error(f"URGENT human review: {resp.safety.reason}")
        else:
            st.success("Nessun flag di rischio urgente")
    with st.expander("Menzioni cliniche + JSON completo"):
        st.json([m.model_dump() for m in resp.clinical_mentions])
        st.json(resp.model_dump())
    return resp


def render_emotion(bundle, text: str):
    from oncoemotion.clinical.prompt import build_decision_prompt, NEUTRAL_FILLER
    from oncoemotion.clinical.measure import point_e_hidden, project_scores, zscore
    from oncoemotion.emotion_vectors.seeds import EMOTION_SEEDS, NEUTRAL
    from oncoemotion.steering.runtime import SteeringRuntime

    adapter, V, val = bundle["adapter"], bundle["V"], bundle["val"]
    method = "diff_of_means"
    emotions = [c for c in EMOTION_SEEDS if (f"{c}|{method}|resid" in V or f"{c}|{method}" in V)]

    def key(c):
        return f"{c}|{method}|resid" if f"{c}|{method}|resid" in V else f"{c}|{method}"

    vectors = {c: V[key(c)] for c in emotions}
    best = {c: val["concepts"].get(c, {}).get("best_layer", vectors[c].shape[0] // 2) for c in emotions}

    # cached neutral baseline
    @st.cache_data
    def baseline():
        proj = {c: [] for c in emotions}
        for t in NEUTRAL[:12]:
            h = point_e_hidden(adapter, build_decision_prompt(t))
            s = project_scores(h, vectors, best)
            for c in emotions:
                proj[c].append(s[c])
        return ({c: float(np.mean(proj[c])) for c in emotions},
                {c: float(np.std(proj[c]) + 1e-9) for c in emotions})

    bmean, bstd = baseline()
    h = point_e_hidden(adapter, build_decision_prompt(text))
    z = zscore(project_scores(h, vectors, best), bmean, bstd)

    st.subheader("Emotion-like signal at the decision point (z vs neutral)")
    st.caption("NON è una diagnosi né un'emozione cosciente: rappresentazione interna emotion-like.")
    st.bar_chart({"z-score": {c: round(z[c], 2) for c in emotions}})

    # emotion x layer heatmap (raw projection magnitude)
    with st.expander("Heatmap emotion × layer (proiezione grezza)"):
        n_layers = next(iter(vectors.values())).shape[0]
        M = np.zeros((len(emotions), n_layers))
        for i, c in enumerate(emotions):
            for l in range(n_layers):
                v = vectors[c][l]; n = np.linalg.norm(v)
                M[i, l] = float(h[l] @ (v / n)) if n > 0 else 0.0
        st.write("righe = emozioni, colonne = layer")
        st.dataframe({f"L{l}": {emotions[i]: round(M[i, l], 2) for i in range(len(emotions))}
                      for l in range(0, n_layers, max(1, n_layers // 12))})

    # steering comparison (research)
    with st.expander("Confronto baseline vs steering (research)"):
        concept = st.selectbox("direzione", emotions, index=0)
        alpha = st.slider("alpha (norm-scaled)", -0.10, 0.10, 0.05, 0.01)
        if st.button("Applica steering"):
            rt = SteeringRuntime(adapter)
            r = rt.steer_and_summarize(build_decision_prompt(text), best[concept], vectors[concept], alpha)
            st.write(f"Δ entropia = {r['delta_entropy']:+.3f} · Δ margine = {r['delta_margin']:+.3f} · "
                     f"top-1 cambiato = {r['top1_changed']}")
            st.caption("Confronta sempre con vettore random / emozione opposta (vedi scripts/run_steering.py).")


def main():
    st.title("oncoemotion — PRO-CTCAE/CTCAE mapping + emotion-concept research")
    st.caption("Strumento di ricerca e supporto. Non effettua diagnosi autonome né sostituisce la revisione clinica. "
               "Gli emotion score sono rappresentazioni interne emotion-like, non emozioni coscienti.")

    text = st.text_area("Testo del paziente (campo libero)", "ho un dolore lancinante e insopportabile che non passa")
    if not text.strip():
        st.stop()

    render_mapping(get_mapper(), text)

    st.divider()
    if st.checkbox("Abilita analisi emotion (research, richiede modello + vettori)"):
        bundle = get_model_and_vectors()
        if bundle is None:
            st.info("Vettori emotivi non trovati. Esegui prima: python scripts/build_vectors.py")
        else:
            render_emotion(bundle, text)


if __name__ == "__main__":
    main()
