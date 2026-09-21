"""Build the worked-example figure: one moment of audio, all the way through.

The report needs to show what the data actually looks like, and that means
showing Bengali script. pdfLaTeX cannot typeset Bengali, and neither
matplotlib nor this machine's Pillow does complex-script shaping, so the
conjuncts and dependent vowel signs would come out wrong. This script writes
an HTML page instead and the figure is captured from a browser, which shapes
the script correctly.

Everything on the page is read from the released files. Nothing is retyped,
so the figure cannot drift away from the corpus.

    python -X utf8 speaker_sentiment/make_sample_figure.py
    # then screenshot runs/sample/sample_figure.html at 1400px wide

Example shown: 2107006::ep005, the moment at 2328.6 s, and the speaker unit
it belongs to. That speaker is labelled `mixed` overall while the turn shown
is `negative`, which is the point the figure exists to make.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "runs" / "sample"

EPISODE = "2107006::ep005"
SPEAKER_KEY = "2107006::ep005::S3"
TURN_INDEX = 24
WINDOW_ID = "ep005_utt00100"
FUSED = ROOT.parent / "2107006" / "results" / "fused" / "ep005.json"
RTTM = ROOT.parent / "2107006" / "results" / "diarization" / "ep005.rttm"

PALETTE = {
    "sig": ("#DCE8F6", "#3E6FA8"),
    "dat": ("#DDEEE1", "#3C7D52"),
    "mod": ("#FBE6D4", "#B5701F"),
    "out": ("#EAE1F2", "#6E4B9E"),
    "ask": ("#E7EAF0", "#6B7483"),
}
LABEL_COLOUR = {
    "negative": "#B5432A",
    "neutral": "#6B7483",
    "positive": "#3C7D52",
    "mixed": "#B5701F",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def esc(text: str) -> str:
    return html.escape(str(text))


def highlight(text: str, span: str) -> str:
    """Mark the judge's evidence span inside the turn text, if it is there."""
    if span and span in text:
        before, _, after = text.partition(span)
        return f"{esc(before)}<mark>{esc(span)}</mark>{esc(after)}"
    return esc(text)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    speaker = next(
        row for row in read_jsonl(ROOT / "data" / "speakers_labelled.jsonl")
        if row["speaker_key"] == SPEAKER_KEY
    )
    turns = [
        row for row in read_jsonl(ROOT / "data" / "turns_labelled.jsonl")
        if row["speaker_key"] == SPEAKER_KEY
    ]
    turn = next(row for row in turns if row["turn_index"] == TURN_INDEX)

    fused = json.loads(FUSED.read_text(encoding="utf-8"))
    window = next(u for u in fused["utterances"] if u["utterance_id"] == WINDOW_ID)
    # The window immediately before this one, which belongs to someone else.
    # Taking the first match rather than the last would reach back to the top
    # of the episode and report the wrong speaker.
    neighbour = max(
        (u for u in fused["utterances"]
         if u["end_sec"] <= window["start_sec"] + 0.01
         and u["speaker_id"] != window["speaker_id"]),
        key=lambda u: u["end_sec"],
    )

    rttm_lines = [
        line.split()
        for line in RTTM.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rttm = [
        parts for parts in rttm_lines
        if 2300.0 < float(parts[3]) < 2340.0
    ]

    words = (window.get("words") or [])[:6]
    word_chips = "".join(
        f'<span class="chip"><b>{esc(str(w.get("word", "")).strip())}</b>'
        f'<i>{float(w.get("start", 0)):.2f}s</i></span>'
        for w in words
    )

    rttm_rows = "".join(
        f"<tr><td>{esc(p[7])}</td><td>{float(p[3]):.2f}s</td>"
        f"<td>{float(p[4]):.2f}s</td></tr>"
        for p in rttm
    )

    tally = (f'{speaker["n_negative"]} negative &middot; {speaker["n_neutral"]} neutral '
             f'&middot; {speaker["n_positive"]} positive')

    other_turns = "".join(
        f'<li><span class="pill" style="background:{LABEL_COLOUR[t["label"]]}">'
        f'{esc(t["label"])}</span> turn {t["turn_index"]}, {t["n_words"]} words</li>'
        for t in sorted(turns, key=lambda r: r["turn_index"])
    )

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Dataset sample</title>
<style>
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; }}
  body {{
    width: 1400px; margin: 0; padding: 26px 30px 30px;
    background: #fff; color: #14171c;
    font-family: "Segoe UI", Arial, Helvetica, sans-serif;
  }}
  .bn {{ font-family: "Nirmala UI", "Noto Sans Bengali", sans-serif;
         font-size: 17px; line-height: 1.85; }}
  .stage {{
    border: 1px solid; border-radius: 6px; padding: 12px 15px; margin-bottom: 11px;
    position: relative;
  }}
  .stage h3 {{
    margin: 0 0 7px; font-size: 13px; letter-spacing: .055em; text-transform: uppercase;
  }}
  .stage .meta {{ font-size: 12.5px; color: #4a5059; margin-bottom: 6px; }}
  .num {{
    position: absolute; top: -11px; left: 13px; width: 22px; height: 22px;
    border-radius: 50%; background: #fff; border: 1px solid #9aa1ab;
    font-size: 12px; font-weight: 700; text-align: center; line-height: 20px;
  }}
  table {{ border-collapse: collapse; font-size: 12.5px; }}
  td {{ padding: 2px 16px 2px 0; font-variant-numeric: tabular-nums; }}
  .chip {{
    display: inline-block; margin: 2px 5px 2px 0; padding: 3px 8px;
    border: 1px solid #b9c2cd; border-radius: 4px; background: #fff;
  }}
  .chip b {{ font-family: "Nirmala UI","Noto Sans Bengali",sans-serif; font-size: 15px; }}
  .chip i {{ font-style: normal; color: #6b7483; font-size: 11px; margin-left: 6px; }}
  mark {{ background: #FFE9A8; padding: 1px 2px; border-radius: 2px; }}
  .pill {{
    display: inline-block; padding: 1px 9px; border-radius: 10px;
    color: #fff; font-size: 11.5px; font-weight: 700; letter-spacing: .03em;
  }}
  .cols {{ display: flex; gap: 16px; }}
  .cols > div {{ flex: 1; }}
  ul {{ margin: 4px 0 0; padding-left: 17px; font-size: 12.5px; line-height: 1.75; }}
  .verdict {{ font-size: 21px; font-weight: 700; }}
  .note {{ font-size: 12px; color: #4a5059; margin-top: 7px; font-style: italic; }}
</style></head><body>

<div class="stage" style="background:{PALETTE['sig'][0]};border-color:{PALETTE['sig'][1]}">
  <div class="num">1</div>
  <h3 style="color:{PALETTE['sig'][1]}">Broadcast audio, then diarization</h3>
  <div class="meta">Episode <b>{esc(EPISODE)}</b> &middot; pyannote decides who holds the floor.
    Speaker ids are local to this episode.</div>
  <table><tr><td><b>speaker</b></td><td><b>starts</b></td><td><b>lasts</b></td></tr>
  {rttm_rows}</table>
  <div class="note">The floor changes hands at 2328.8 s. Everything below follows the
    speaker who starts there.</div>
</div>

<div class="stage" style="background:{PALETTE['sig'][0]};border-color:{PALETTE['sig'][1]}">
  <div class="num">2</div>
  <h3 style="color:{PALETTE['sig'][1]}">Speech recognition, in fixed windows</h3>
  <div class="meta">Whisper-medium Bangla decodes the audio in windows of about 28 seconds,
    with a timestamp on every word. Window <b>{esc(WINDOW_ID)}</b>,
    {window['start_sec']:.1f}&ndash;{window['end_sec']:.1f} s, speaker {window['speaker_id']}.</div>
  <div>{word_chips}<span class="chip" style="border-style:dashed">&hellip;</span></div>
  <div class="bn" style="margin-top:8px">{esc(window['text_verbatim'])}</div>
  <div class="note">The previous window ({neighbour['start_sec']:.1f}&ndash;{neighbour['end_sec']:.1f} s)
    belongs to speaker {neighbour['speaker_id']}, so a window boundary is not a speaker boundary.</div>
</div>

<div class="stage" style="background:{PALETTE['dat'][0]};border-color:{PALETTE['dat'][1]}">
  <div class="num">3</div>
  <h3 style="color:{PALETTE['dat'][1]}">One speaker turn, rebuilt</h3>
  <div class="meta">Consecutive windows from the same speaker are merged back into a turn, and
    fragments cut off mid-word are dropped. <b>{esc(turn['turn_id'])}</b> &middot;
    {turn['start_sec']:.1f}&ndash;{turn['end_sec']:.1f} s &middot; {turn['n_words']} words &middot;
    built from {turn.get('n_windows', 1)} window(s) &middot; role: {esc(turn['role'])}</div>
  <div class="bn">{highlight(turn['text'], turn['evidence'])}</div>
</div>

<div class="stage" style="background:{PALETTE['mod'][0]};border-color:{PALETTE['mod'][1]}">
  <div class="num">4</div>
  <h3 style="color:{PALETTE['mod'][1]}">Level one: what this turn expresses</h3>
  <div class="meta">
    <span class="pill" style="background:{LABEL_COLOUR[turn['label']]}">{esc(turn['label'])}</span>
    &nbsp;judge confidence: {esc(turn['judge_confidence'])} &middot; split: {esc(turn['split'])}
  </div>
  <div style="font-size:13px">Evidence the judge cited, highlighted above:
    <span class="bn" style="font-size:16px"><mark>{esc(turn['evidence'])}</mark></span>
    &nbsp;&mdash;&nbsp;<i>&ldquo;no real change has happened&rdquo;</i></div>
</div>

<div class="stage" style="background:{PALETTE['out'][0]};border-color:{PALETTE['out'][1]}">
  <div class="num">5</div>
  <h3 style="color:{PALETTE['out'][1]}">Level two: where this speaker stands in the whole episode</h3>
  <div class="cols">
    <div>
      <div class="meta"><b>{esc(SPEAKER_KEY)}</b> &middot; {esc(speaker['role'])} &middot;
        {speaker['n_turns']} turns &middot; {speaker['n_words']:,} words</div>
      <div class="verdict" style="color:{LABEL_COLOUR[speaker['judge_label']]}">
        {esc(speaker['judge_label'])}
        <span style="font-size:14px;color:#4a5059">
          &nbsp;stance {speaker['judge_stance_score']:+.2f} on a &minus;1 to +1 scale</span>
      </div>
      <div class="meta" style="margin-top:5px">Turn labels for this speaker: {tally}</div>
      <ul>{other_turns}</ul>
    </div>
    <div>
      <div class="meta"><b>Why the verdict is not simply the majority turn label</b></div>
      <div style="font-size:13px;line-height:1.65">{esc(speaker['rationale'])}</div>
      <div class="meta" style="margin-top:8px"><b>What the speaker talked about</b></div>
      <div style="font-size:12.5px">{esc(', '.join(speaker['dominant_topics']))}</div>
    </div>
  </div>
</div>

</body></html>
"""

    out = OUT_DIR / "sample_figure.html"
    out.write_text(page, encoding="utf-8")
    print("wrote", out)
    print(f"example: {SPEAKER_KEY}, turn {TURN_INDEX} is "
          f"{turn['label']} while the speaker verdict is {speaker['judge_label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
