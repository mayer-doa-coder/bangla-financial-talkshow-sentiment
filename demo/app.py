"""Minimal Gradio demonstration for Bangla financial transcript retrieval.

Run locally:
    python demo/app.py --data-root datasets

Run in Google Colab/remote environments:
    python demo/app.py --data-root /content/drive/MyDrive/ML_Project/datasets --share
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from search_core import (
    ALL_COLLECTIONS,
    ALL_EPISODES,
    ALL_SPEAKERS,
    SEMANTIC_MODEL,
    CorpusIndex,
    SearchHit,
    highlight_query,
)
from showcase_core import ProjectShowcase


APP_TITLE = "অর্থকথা অনুসন্ধান"
APP_SUBTITLE = "Speaker-attributed search across Bangla financial talk shows"


CSS = r"""
:root {
  --ink: #17201c;
  --muted: #627168;
  --paper: #fbfcfa;
  --panel: #ffffff;
  --line: #e3e9e4;
  --green: #0b6b4f;
  --green-soft: #e8f4ef;
  --gold: #b77a13;
}
html, body, .gradio-container, .dark, .dark .gradio-container {
  background: var(--paper) !important;
  color: var(--ink) !important;
  color-scheme: light;
}
.gradio-container {
  max-width: 1240px !important;
  margin: 0 auto !important;
  font-family: "Noto Sans Bengali", "Hind Siliguri", Inter, system-ui, sans-serif !important;
}
/* Gradio token overrides so components never inherit dark-theme text/surfaces. */
:root, .dark {
  --body-background-fill: var(--paper);
  --body-text-color: var(--ink);
  --body-text-color-subdued: var(--muted);
  --background-fill-primary: var(--panel);
  --background-fill-secondary: #f5f8f6;
  --block-background-fill: var(--panel);
  --block-label-background-fill: var(--green-soft);
  --block-label-text-color: var(--green);
  --block-title-text-color: var(--ink);
  --block-border-color: var(--line);
  --border-color-primary: var(--line);
  --border-color-accent: var(--green);
  --input-background-fill: var(--panel);
  --input-border-color: var(--line);
  --input-placeholder-color: var(--muted);
  --panel-background-fill: var(--panel);
  --table-even-background-fill: var(--panel);
  --table-odd-background-fill: #f7faf8;
  --table-border-color: var(--line);
  --table-text-color: var(--ink);
  --link-text-color: var(--green);
  --link-text-color-hover: var(--green);
  --neutral-950: #17201c;
}
.gradio-container h1, .gradio-container h2, .gradio-container h3,
.gradio-container h4, .gradio-container p, .gradio-container li,
.gradio-container span, .gradio-container label, .gradio-container td,
.gradio-container th, .gradio-container summary, .gradio-container .prose {
  color: var(--ink);
}
.gradio-container .prose p, .gradio-container .prose li { color: var(--ink); }
.gradio-container input, .gradio-container textarea, .gradio-container select {
  background: var(--panel) !important;
  color: var(--ink) !important;
}
.gradio-container table, .gradio-container .table-wrap, .gradio-container .cell-wrap {
  background: var(--panel) !important;
  color: var(--ink) !important;
}
.gradio-container thead th, .gradio-container .svelte-virtual-table-viewport thead th {
  background: var(--green-soft) !important;
  color: var(--green) !important;
}
.gradio-container .tab-nav button { color: var(--muted); }
.gradio-container .tab-nav button.selected {
  color: var(--green);
  border-bottom-color: var(--green);
}
.hero {
  padding: 28px 30px;
  border: 1px solid var(--line);
  border-radius: 24px;
  background:
    radial-gradient(circle at 90% 0%, rgba(11,107,79,.15), transparent 38%),
    linear-gradient(145deg, #ffffff 0%, #f3f8f5 100%);
  margin-bottom: 16px;
  box-shadow: 0 14px 40px rgba(25, 56, 43, .07);
}
.hero-kicker { color: var(--green); text-transform: uppercase; letter-spacing: .12em; font-size: 12px; font-weight: 800; }
.hero h1 { margin: 8px 0 6px; font-size: clamp(30px, 5vw, 54px); line-height: 1.08; letter-spacing: -.035em; }
.hero p { color: var(--muted); font-size: 16px; max-width: 760px; margin: 0; }
.stat-grid { display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 10px; margin: 12px 0 20px; }
.stat-card { background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 15px 16px; }
.stat-value { font-size: 25px; font-weight: 760; letter-spacing: -.03em; }
.stat-label { color: var(--muted); font-size: 12px; margin-top: 3px; }
.search-panel { border: 1px solid var(--line); border-radius: 20px; background: var(--panel); padding: 6px; }
.result-list { display: grid; gap: 11px; margin-top: 12px; }
.result-card { border: 1px solid var(--line); border-radius: 17px; padding: 17px 18px; background: #fff; box-shadow: 0 7px 20px rgba(20,40,31,.045); }
.result-card:hover { border-color: #b9cec3; box-shadow: 0 10px 28px rgba(20,40,31,.075); }
.result-top { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px; }
.result-meta { display: flex; flex-wrap: wrap; align-items: center; gap: 7px; }
.pill { border-radius: 999px; padding: 4px 9px; font-size: 12px; font-weight: 700; background: var(--green-soft); color: var(--green); }
.pill.neutral { background: #f0f2f0; color: #4e5b54; }
.pill.score { background: #fff4de; color: #8b5b08; }
.result-text { font-size: 17px; line-height: 1.75; margin: 12px 0 5px; }
.result-context { color: var(--muted); font-size: 13px; line-height: 1.55; margin-top: 7px; }
mark { background: #ffdfa3; color: inherit; padding: 0 2px; border-radius: 3px; }
.method-note { border-left: 3px solid var(--green); padding: 9px 12px; color: var(--muted); background: #f5f8f6; border-radius: 0 10px 10px 0; }
.episode-header { border: 1px solid var(--line); padding: 18px; border-radius: 16px; background: #fff; }
.empty-state { border: 1px dashed #bbc8c0; border-radius: 16px; padding: 28px; text-align: center; color: var(--muted); background: #fbfdfb; }
.footer-note { color: var(--muted); font-size: 12px; line-height: 1.6; padding: 14px 2px; }
.evidence-banner { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 8px 0 16px; }
.evidence-state { border-radius: 15px; padding: 13px 15px; border: 1px solid var(--line); background: #fff; font-size: 13px; }
.evidence-state strong { display: block; margin-bottom: 3px; }
.evidence-state.done { border-left: 4px solid #16805f; }
.evidence-state.exploratory { border-left: 4px solid #c38720; }
.evidence-state.unavailable { border-left: 4px solid #8a9690; }
@media (max-width: 900px) { .stat-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 700px) { .evidence-banner { grid-template-columns: 1fr; } }
"""

FORCE_LIGHT_JS = """
() => {
  const url = new URL(window.location.href);
  if (url.searchParams.get('__theme') !== 'light') {
    url.searchParams.set('__theme', 'light');
    window.location.replace(url.href);
    return;
  }
  document.documentElement.classList.remove('dark');
  document.body.classList.remove('dark');
}
"""

APP_THEME = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="amber",
    neutral_hue="slate",
    radius_size="lg",
)


def _summary_html(index: CorpusIndex) -> str:
    summary = index.corpus_summary()
    values = [
        (f"{summary['contributors']:,}", "Roll folders found"),
        (f"{summary['result_bundles']:,}", "Result bundles"),
        (f"{summary['source_audio_files']:,}", "Source audio files"),
        (f"{summary['episodes']:,}", "Processed episodes"),
        (f"{summary['hours']:.2f} h", "Processed audio"),
        (f"{summary['utterances']:,}", "Utterances"),
        (f"{summary['words']:,}", "ASR words"),
        (f"{summary['assignment_rate'] * 100:.2f}%", "Speaker assigned"),
    ]
    cards = "".join(
        f'<div class="stat-card"><div class="stat-value">{value}</div>'
        f'<div class="stat-label">{label}</div></div>'
        for value, label in values
    )
    return f'<div class="stat-grid">{cards}</div>'


def _warning_html(index: CorpusIndex) -> str:
    if not index.warnings:
        return ""
    unique = list(dict.fromkeys(index.warnings))
    shown = unique[:8]
    suffix = f"<li>…and {len(unique) - len(shown)} more.</li>" if len(unique) > len(shown) else ""
    return (
        "<details><summary>Corpus loading notes</summary><ul>"
        + "".join(f"<li>{html.escape(item)}</li>" for item in shown)
        + suffix
        + "</ul></details>"
    )


def _target_badges(hit: SearchHit) -> str:
    if not hit.targets:
        return ""
    return f'<span class="pill neutral">{html.escape(hit.targets)}</span>'


def _result_cards(hits: list[SearchHit], query: str) -> str:
    if not hits:
        return (
            '<div class="empty-state"><strong>No matching utterance found.</strong><br>'
            "Try Fuzzy, Semantic, a lower relevance threshold, or a wider search scope.</div>"
        )
    cards = []
    for rank, hit in enumerate(hits, start=1):
        confidence = (
            f" · ASR {hit.mean_word_confidence * 100:.0f}%"
            if hit.mean_word_confidence is not None
            else ""
        )
        context = ""
        if hit.context and hit.context != f"Match: {hit.text}":
            previous = hit.context.replace(f"Match: {hit.text}", "")
            context = f'<div class="result-context">{html.escape(previous[:620])}</div>'
        cards.append(
            f"""
            <article class="result-card">
              <div class="result-top">
                <div class="result-meta">
                  <span class="pill">#{rank} · {html.escape(hit.global_episode_id)}</span>
                  <span class="pill">Speaker {hit.speaker_id}</span>
                  <span class="pill neutral">{hit.timestamp}</span>
                  {_target_badges(hit)}
                </div>
                <span class="pill score">{hit.match_type.title()} · {hit.score * 100:.1f}%{confidence}</span>
              </div>
              <div class="result-text">{highlight_query(hit.text, query)}</div>
              {context}
            </article>
            """
        )
    return '<div class="result-list">' + "".join(cards) + "</div>"


def _hits_dataframe(hits: list[SearchHit]) -> pd.DataFrame:
    columns = [
        "Rank",
        "Roll",
        "Episode",
        "Speaker ID",
        "Timestamp",
        "Match",
        "Relevance",
        "Transcript",
        "Targets",
        "Audio",
    ]
    rows = []
    for rank, hit in enumerate(hits, start=1):
        rows.append(
            {
                "Rank": rank,
                "Roll": hit.collection_id,
                "Episode": hit.episode_id,
                "Speaker ID": hit.speaker_id,
                "Timestamp": hit.timestamp,
                "Match": hit.match_type,
                "Relevance": round(hit.score, 3),
                "Transcript": hit.text,
                "Targets": hit.targets,
                "Audio": "Yes" if hit.audio_available else "No",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _write_search_csv(hits: list[SearchHit]) -> str | None:
    if not hits:
        return None
    directory = Path(tempfile.gettempdir()) / "bangla_financial_search_exports"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "speaker_search_results.csv"
    rows = []
    for rank, hit in enumerate(hits, start=1):
        rows.append(
            {
                "rank": rank,
                "roll": hit.collection_id,
                "episode_id": hit.episode_id,
                "speaker_id": hit.speaker_id,
                "speaker_reference": hit.speaker_reference,
                "start_sec": hit.start_sec,
                "end_sec": hit.end_sec,
                "timestamp": hit.timestamp,
                "match_type": hit.match_type,
                "relevance": hit.score,
                "transcript": hit.text,
                "targets": hit.targets,
            }
        )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def _method_note(method: str) -> str:
    notes = {
        "Smart": "Smart search prioritizes exact matches, then complete-token and fuzzy matches.",
        "Exact": "Exact search requires the normalized word or phrase to appear in the transcript.",
        "All words": "All query words must occur in the same utterance; their order may differ.",
        "Any word": "At least one query word must occur; results with more matched words rank higher.",
        "Fuzzy": "Fuzzy search tolerates Bangla spelling differences and ASR transcription variation.",
        "Semantic": f"Semantic search uses {SEMANTIC_MODEL} and ranks meaning rather than exact wording.",
        "Hybrid": "Hybrid search combines multilingual semantic similarity with lexical evidence.",
    }
    return notes.get(method, "")


def _speaker_plot(index: CorpusIndex, episode: str):
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    rows = index.speaker_rows(episode)
    seconds = []
    for row in rows:
        selected = [
            record
            for record in index.records
            if record.global_episode_id == episode and record.speaker_id == row["Speaker ID"]
        ]
        seconds.append(sum(max(0.0, record.end_sec - record.start_sec) for record in selected) / 60)
    figure = go.Figure(
        go.Bar(
            x=[row["Speaker"] for row in rows],
            y=seconds,
            marker_color="#0b6b4f",
            hovertemplate="%{x}<br>%{y:.1f} minutes<extra></extra>",
        )
    )
    figure.update_layout(
        title="Attributed speaking time",
        xaxis_title="Episode-local speaker",
        yaxis_title="Minutes",
        template="plotly_white",
        height=360,
        margin=dict(l=40, r=20, t=55, b=40),
        font=dict(family="Noto Sans Bengali, Arial"),
    )
    return figure


def _gate_plot(showcase: ProjectShowcase):
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    counts: dict[str, int] = {"PASS": 0, "FAIL": 0, "SKIPPED": 0}
    for row in showcase.gate_rows():
        status = str(row.get("Status", "")).upper()
        counts[status] = counts.get(status, 0) + 1
    figure = go.Figure(
        go.Bar(
            x=list(counts),
            y=list(counts.values()),
            marker_color=["#16805f", "#bd4d3a", "#8a9690"],
            text=list(counts.values()),
            textposition="auto",
            hovertemplate="%{x}: %{y}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Recorded evaluation gates",
        yaxis_title="Gate count",
        template="plotly_white",
        height=330,
        margin=dict(l=40, r=20, t=55, b=40),
    )
    return figure


def _profile_plot(showcase: ProjectShowcase):
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    rows = showcase.profile_rows()
    labels = [f"{row['Roll']}::{row['Episode']} / S{row['Speaker ID']}" for row in rows]
    figure = go.Figure()
    figure.add_bar(name="Positive", x=labels, y=[row["Positive"] for row in rows], marker_color="#16805f")
    figure.add_bar(name="Negative", x=labels, y=[row["Negative"] for row in rows], marker_color="#bd4d3a")
    figure.update_layout(
        title="Exploratory weak-labelled sentiment snapshot",
        barmode="stack",
        yaxis_title="Target labels",
        template="plotly_white",
        height=390,
        margin=dict(l=40, r=20, t=55, b=95),
        xaxis_tickangle=-35,
    )
    return figure


def build_app(index: CorpusIndex, semantic_cache_dir: Path) -> gr.Blocks:
    summary = index.corpus_summary()
    showcase = ProjectShowcase(index)
    episode_choices = index.global_episodes
    collection_choices = [ALL_COLLECTIONS, *index.collections]
    speaker_choices: list[Any] = [ALL_SPEAKERS, *[str(value) for value in index.speaker_ids]]
    holder = {"index": index}

    def run_search(
        query: str,
        query_type: str,
        method: str,
        collection: str,
        episode: str,
        speaker: str,
        include_unassigned: bool,
        minimum_score: float,
        top_k: int,
        include_context: bool,
    ):
        if not str(query or "").strip():
            empty = _hits_dataframe([])
            return (
                "### Enter a Bangla or English word, sentence, or concept.",
                _result_cards([], ""),
                empty,
                [],
                gr.Dropdown(choices=[], value=None),
                None,
            )
        effective_method = "Hybrid" if query_type == "Concept / question" and method == "Smart" else method
        try:
            hits = holder["index"].search(
                query,
                query_type=query_type,
                method=effective_method,
                collection=collection,
                episode=episode,
                speaker=speaker,
                include_unassigned=include_unassigned,
                minimum_score=minimum_score,
                top_k=top_k,
                include_context=include_context,
                semantic_cache_dir=semantic_cache_dir,
            )
            note = _method_note(effective_method)
            status = (
                f"### {len(hits)} result{'s' if len(hits) != 1 else ''} · {effective_method}\n"
                f"<div class='method-note'>{note} Relevance is a ranking score, not accuracy.</div>"
            )
        except Exception as exc:  # Surface optional semantic setup failures cleanly.
            hits = []
            status = f"### Search could not run\n`{type(exc).__name__}: {exc}`"
        choices = [
            (
                f"#{rank} · {hit.global_episode_id} · Speaker {hit.speaker_id} · {hit.timestamp}",
                hit.key,
            )
            for rank, hit in enumerate(hits, start=1)
        ]
        first = choices[0][1] if choices else None
        state = [hit.key for hit in hits]
        return (
            status,
            _result_cards(hits, query),
            _hits_dataframe(hits),
            state,
            gr.Dropdown(choices=choices, value=first),
            _write_search_csv(hits),
        )

    def preview_result(key: str | None, _state: list[str]):
        if not key:
            return None, "Select a result to inspect its speaker-attributed audio segment."
        record = holder["index"].record_by_key(key)
        if record is None:
            return None, "The selected result is no longer available."
        clip = holder["index"].create_audio_clip(key)
        details = (
            f"### {record.global_episode_id} · Speaker {record.speaker_id}\n"
            f"**Time:** {record.timestamp}  \n"
            f"**Utterance ID:** `{record.utterance_id}`  \n\n"
            f"{record.text}"
        )
        if clip is None:
            details += "\n\n_Audio preview is unavailable. Confirm the source audio path and ffmpeg installation._"
        return str(clip) if clip else None, details

    def explore_episode(episode: str):
        selected = holder["index"].episodes[episode]
        details = (
            f'<div class="episode-header"><strong>{html.escape(episode)}</strong><br>'
            f'{html.escape(selected.programme or "Programme not specified")} · '
            f'{selected.speaker_count} speakers · {selected.utterance_count} utterances · '
            f'{selected.word_count:,} words · {selected.assignment_rate * 100:.2f}% assigned'
            f'</div>'
        )
        return (
            details,
            pd.DataFrame(holder["index"].speaker_rows(episode)),
            pd.DataFrame(holder["index"].transcript_rows(episode)),
            _speaker_plot(holder["index"], episode),
            showcase.waveform_path(episode),
        )

    def build_semantic():
        try:
            status = holder["index"].build_semantic_index(semantic_cache_dir)
            return f"✅ {status}"
        except Exception as exc:
            return f"❌ {type(exc).__name__}: {exc}"

    def download_evidence():
        return showcase.create_evidence_zip()

    with gr.Blocks(title=APP_TITLE) as app:
        gr.HTML(
            f"""
            <section class="hero">
              <div class="hero-kicker">RTV Business Talk · Bangla financial discourse</div>
              <h1>{APP_TITLE}</h1>
              <p>{APP_SUBTITLE}. The scanner keeps each roll folder separate, reports both
              group totals and individual contributions, and preserves roll, episode,
              speaker, timestamp, transcript, and audio provenance for every result.</p>
            </section>
            {_summary_html(index)}
            """
        )

        with gr.Tabs():
            with gr.Tab("Project overview", id="overview"):
                gr.HTML(
                    """
                    <div class="evidence-banner">
                      <div class="evidence-state done"><strong>Completed automatic evidence</strong>Audio inventory, diarization, ASR, timestamp fusion, annotation drafts and retrieval.</div>
                      <div class="evidence-state exploratory"><strong>Exploratory output</strong>Weak-labelled profiles are shown with their tier and alignment status.</div>
                      <div class="evidence-state unavailable"><strong>Gold-dependent evaluation</strong>WER, DER, sentiment F1 and error propagation remain unavailable until gold references exist.</div>
                    </div>
                    """
                )
                gr.Markdown(
                    f"""
                    ## End-to-end project evidence

                    The current scan found **{summary['contributors']:,} roll folder(s)**,
                    **{summary['source_audio_files']:,} source audio file(s)** and
                    **{summary['result_bundles']:,} result bundle(s)**. Completed bundles contribute
                    **{summary['episodes']:,} processed episodes**, **{summary['hours']:.2f} hours**,
                    **{summary['utterances']:,} fused utterances**, and **{summary['words']:,} timestamped words**.
                    Every figure below is read from durable pipeline artifacts; unavailable evaluation is displayed
                    explicitly rather than replaced with estimated accuracy.
                    """
                )
                gr.Markdown("### Group and individual contribution summary")
                gr.Dataframe(
                    pd.DataFrame(index.contributor_rows()),
                    interactive=False,
                    wrap=True,
                    label="Roll folders, source audio and completed result bundles",
                )
                gr.Dataframe(
                    pd.DataFrame(showcase.stage_summary_rows()),
                    interactive=False,
                    wrap=True,
                    label="Pipeline completion and evidence map",
                )
                gr.Markdown("### Episode-level corpus summary")
                gr.Dataframe(pd.DataFrame(index.episode_rows()), interactive=False, wrap=True)
                gr.Markdown("### Models and reproducibility provenance")
                gr.Dataframe(pd.DataFrame(showcase.model_rows()), interactive=False, wrap=True)
                gr.HTML(_warning_html(index))

            with gr.Tab("Search & evidence", id="search"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=7, elem_classes="search-panel"):
                        query = gr.Textbox(
                            label="What do you want to find?",
                            placeholder="উদাহরণ: ব্যাংক · ব্যাংক খাতের সংকট · রেমিট্যান্স কেন কমছে?",
                            lines=2,
                            autofocus=True,
                        )
                        with gr.Row():
                            query_type = gr.Dropdown(
                                ["Auto", "Word", "Sentence", "Concept / question"],
                                value="Auto",
                                label="Query type",
                            )
                            method = gr.Dropdown(
                                ["Smart", "Exact", "All words", "Any word", "Fuzzy", "Semantic", "Hybrid"],
                                value="Smart",
                                label="Search method",
                            )
                        with gr.Row():
                            collection = gr.Dropdown(collection_choices, value=ALL_COLLECTIONS, label="Roll / contributor")
                            episode = gr.Dropdown([ALL_EPISODES, *episode_choices], value=ALL_EPISODES, label="Episode scope")
                            speaker = gr.Dropdown(speaker_choices, value=ALL_SPEAKERS, label="Episode-local speaker ID")
                        # Kept outside the accordion: episodes whose diarization
                        # covered little speech are almost invisible until this
                        # is ticked, so the operator must be able to see it.
                        include_unassigned = gr.Checkbox(
                            False,
                            label="Include words with no speaker (-1)",
                            info="Needed to search episodes with low diarization coverage.",
                        )
                        with gr.Accordion("Ranking and display options", open=False):
                            with gr.Row():
                                minimum_score = gr.Slider(0.0, 1.0, value=0.55, step=0.01, label="Minimum relevance")
                                top_k = gr.Slider(5, 100, value=20, step=5, label="Maximum results")
                            include_context = gr.Checkbox(True, label="Show surrounding context")
                        search_button = gr.Button("Search transcripts", variant="primary", size="lg")
                        gr.Examples(
                            examples=[["ব্যাংক"], ["ব্যাংক খাতের সংকট"], ["রেমিট্যান্স"], ["মূল্যস্ফীতি কেন বাড়ছে"]],
                            inputs=[query],
                            label="Try an example",
                        )
                    with gr.Column(scale=3):
                        gr.Markdown(
                            """
                            ### Search options

                            - **Word / Exact:** literal mentions.
                            - **Fuzzy:** spelling and ASR variation.
                            - **Semantic:** same idea, different wording.
                            - **Hybrid:** meaning plus lexical evidence.

                            Speaker IDs are **episode-local predictions**, not names or identities.
                            """
                        )
                        semantic_button = gr.Button("Prepare semantic search", variant="secondary")
                        semantic_status = gr.Markdown("Semantic search loads lazily; keyword and fuzzy search work immediately.")

                search_status = gr.Markdown("### Ready to search")
                result_cards = gr.HTML()
                with gr.Accordion("Result table and export", open=False):
                    result_table = gr.Dataframe(interactive=False, wrap=True)
                    download = gr.DownloadButton("Download search results as CSV")

                with gr.Row():
                    with gr.Column(scale=4):
                        preview_choice = gr.Dropdown(label="Choose a result to play", choices=[])
                        preview_audio = gr.Audio(label="Matching audio excerpt", type="filepath")
                    with gr.Column(scale=6):
                        preview_details = gr.Markdown("Select a result after searching.")
                result_state = gr.State([])

            with gr.Tab("Explore an episode", id="episode"):
                selected_episode = gr.Dropdown(
                    episode_choices,
                    value=episode_choices[0],
                    label="Episode",
                )
                episode_details = gr.HTML()
                with gr.Row():
                    with gr.Column(scale=6):
                        speaker_table = gr.Dataframe(interactive=False, wrap=True, label="Speaker summary")
                        speaker_plot = gr.Plot(label="Attributed speaking time")
                    with gr.Column(scale=4):
                        waveform = gr.Image(label="Audio waveform report", type="filepath")
                transcript_table = gr.Dataframe(interactive=False, wrap=True, label="Speaker-attributed transcript")

            with gr.Tab("Pipeline results", id="pipeline"):
                gr.Markdown(
                    """
                    ## Evidence from every completed processing stage

                    “Ready” means a durable, non-empty output is present. Some resumed ASR files retain a
                    `0/total` chunk counter even though their segments and fused outputs exist; therefore the
                    interface displays the counter as metadata and judges readiness from the actual output.
                    """
                )
                gr.Dataframe(pd.DataFrame(showcase.stage_rows()), interactive=False, wrap=True, label="End-to-end episode matrix")
                with gr.Tabs():
                    with gr.Tab("Audio inventory"):
                        gr.Dataframe(pd.DataFrame(showcase.audio_rows()), interactive=False, wrap=True)
                    with gr.Tab("Diarization"):
                        gr.Markdown("Episode-local speaker turns, model provider, post-processing thresholds and RTTM evidence.")
                        gr.Dataframe(pd.DataFrame(showcase.diarization_rows()), interactive=False, wrap=True)
                    with gr.Tab("ASR"):
                        gr.Markdown("Long-form Bangla decoding configuration, completed segments and runtime provenance.")
                        gr.Dataframe(pd.DataFrame(showcase.asr_rows()), interactive=False, wrap=True)
                    with gr.Tab("Fusion"):
                        gr.Markdown("Word timestamps fused with diarization turns using midpoint then maximum-overlap assignment.")
                        gr.Dataframe(pd.DataFrame(showcase.fusion_rows()), interactive=False, wrap=True)
                    with gr.Tab("Annotation drafts"):
                        gr.Markdown("ASR-fused rows prepared for human correction; these are drafts, not gold annotations.")
                        gr.Dataframe(pd.DataFrame(showcase.annotation_rows()), interactive=False, wrap=True)

            with gr.Tab("Financial analysis", id="financial"):
                gr.Markdown(
                    """
                    ## Target-aware financial discourse design

                    Sentiment is only meaningful with its financial target. The project therefore separates
                    MARKET, SECTOR, COMPANY, REGULATOR and MACRO_INDICATOR and retains positive/negative polarity.
                    The current fused drafts contain no final target labels. Any older weak-labelled profile below
                    remains useful as an exploratory experiment but is not presented as current validated output.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=5):
                        gr.Dataframe(pd.DataFrame(showcase.target_policy_rows()), interactive=False, wrap=True, label="Target taxonomy")
                    with gr.Column(scale=5):
                        gr.Dataframe(pd.DataFrame(showcase.entity_rows()), interactive=False, wrap=True, label="Financial entity lexicon")
                gr.Plot(_profile_plot(showcase), label="Weak-labelled profile visualisation")
                gr.Dataframe(pd.DataFrame(showcase.profile_rows()), interactive=False, wrap=True, label="Exploratory discourse-profile snapshot")

            with gr.Tab("Evaluation & downloads", id="evaluation"):
                gr.Markdown(
                    """
                    ## Honest evaluation status

                    The submission gate report is preserved in full. A **SKIPPED** gate records a planned
                    experiment whose required gold reference does not exist; it is not a zero score. A **FAIL**
                    may also describe project-scope requirements rather than a broken automatic output. Snapshot
                    inconsistencies are flagged when an older aggregate disagrees with current fused files.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=4):
                        gr.Plot(_gate_plot(showcase))
                    with gr.Column(scale=6):
                        gr.Markdown(
                            """
                            ### Metrics that cannot be truthfully computed yet

                            - **WER:** needs a human-corrected transcript.
                            - **DER:** needs a human speaker-turn RTTM reference.
                            - **Sentiment macro-F1:** needs verified target/polarity labels.
                            - **Error propagation:** needs the paired gold and automatic streams above.

                            Retrieval, timestamps and audio evidence remain demonstrable without inventing these values.
                            """
                        )
                gr.Dataframe(pd.DataFrame(showcase.gate_rows()), interactive=False, wrap=True, label="Submission and evaluation gates")
                gr.Markdown("### Durable artifact inventory")
                gr.Dataframe(pd.DataFrame(showcase.artifact_summary_rows()), interactive=False, wrap=True)
                with gr.Accordion("Show every exported artifact", open=False):
                    gr.Dataframe(pd.DataFrame(showcase.artifact_rows()), interactive=False, wrap=True)
                evidence_button = gr.Button("Build downloadable evidence package", variant="primary")
                evidence_download = gr.File(label="Project evidence ZIP")

            with gr.Tab("Methodology", id="methodology"):
                gr.Markdown(
                    """
                    ## Research-informed cascaded methodology

                    1. **Inventory and audio hygiene:** preserve source provenance and prepare consistent audio.
                    2. **Speaker diarization:** pyannote Community-1/Wespeaker-based automatic turns with
                       episode-local opaque IDs and first-speaker-wins overlap policy.
                    3. **Long-form Bangla ASR:** Tugstugi Whisper-medium derivative, sub-30-second chunks,
                       word timestamps, beam search, selective Demucs and previous-text conditioning disabled.
                    4. **Fusion:** assign each timestamped ASR word by midpoint, then maximum temporal overlap;
                       construct speaker-attributed utterances while retaining unassigned evidence.
                    5. **Target-aware analysis:** binary polarity is always tied to MARKET, SECTOR, COMPANY,
                       REGULATOR or MACRO_INDICATOR rather than reported as context-free sentiment.
                    6. **Aggregation:** preserve speaker and target separately when building discourse profiles.
                    7. **Error-propagation design:** compare gold/automatic diarization and transcript streams
                       through the same classifier once gold references exist.
                    8. **Interactive retrieval:** exact evidence, ASR-tolerant fuzzy matching, multilingual
                       semantic search and hybrid ranking with direct return to source audio.

                    ### Scientific interpretation

                    - A result such as `2107006::ep002 / Speaker 3` does not identify a real person.
                    - Relevance is a retrieval ranking score, not accuracy or probability.
                    - Transcript and speaker labels are automatic and may contain ASR/diarization errors.
                    - Semantic search uses multilingual E5 with `query:` and `passage:` retrieval prefixes.
                    - No cross-episode speaker identification is performed.
                    - Automatic coverage is not WER or DER; those require human references.
                    """
                )

        gr.HTML(
            """
            <div class="footer-note">
              Built for the CSE 4112 Bangla financial discourse project. Search results preserve
              provenance down to roll, result bundle, episode, utterance, speaker and timestamp.
            </div>
            """
        )

        search_inputs = [
            query,
            query_type,
            method,
            collection,
            episode,
            speaker,
            include_unassigned,
            minimum_score,
            top_k,
            include_context,
        ]
        search_outputs = [
            search_status,
            result_cards,
            result_table,
            result_state,
            preview_choice,
            download,
        ]
        search_button.click(run_search, search_inputs, search_outputs)
        query.submit(run_search, search_inputs, search_outputs)
        preview_choice.change(
            preview_result,
            inputs=[preview_choice, result_state],
            outputs=[preview_audio, preview_details],
        )
        semantic_button.click(build_semantic, outputs=[semantic_status])
        evidence_button.click(download_evidence, outputs=[evidence_download])
        selected_episode.change(
            explore_episode,
            inputs=[selected_episode],
            outputs=[episode_details, speaker_table, transcript_table, speaker_plot, waveform],
        )
        app.load(
            explore_episode,
            inputs=[selected_episode],
            outputs=[episode_details, speaker_table, transcript_table, speaker_plot, waveform],
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bangla financial speaker-attributed search demo")
    parser.add_argument(
        "--data-root",
        default=os.environ.get("BFT_DATA_ROOT", "datasets"),
        help="Root recursively containing one or more data/fused result bundles",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("BFT_SEARCH_CACHE", ""),
        help="Semantic embedding and audio-clip cache (default: temporary directory)",
    )
    parser.add_argument(
        "--server-name",
        default="127.0.0.1",
        help="Bind address (use 0.0.0.0 only when access from other hosts is required)",
    )
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    cache_dir = (
        Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir
        else Path(tempfile.gettempdir()) / "bangla_financial_search"
    )
    print(f"Loading result bundles under: {data_root}")
    index = CorpusIndex(data_root)
    summary = index.corpus_summary()
    print(
        f"Loaded {summary['contributors']} roll folder(s), "
        f"{summary['source_audio_files']} source audio file(s), "
        f"{summary['result_bundles']} result bundle(s), "
        f"{summary['episodes']} processed episode(s), "
        f"{summary['utterances']} utterances, {summary['hours']:.2f} hours"
    )
    app = build_app(index, cache_dir)
    # Gradio 6 takes theme/css/js at launch time, not on the Blocks constructor.
    app.queue(default_concurrency_limit=4).launch(
        theme=APP_THEME,
        css=CSS,
        js=FORCE_LIGHT_JS,
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        debug=args.debug,
        show_error=True,
    )


if __name__ == "__main__":
    main()
