"""Render the pipeline report to PDF in the same format as the reference
project report.

Format is taken from measurements of Meowtropolis-Project-Report.pdf:
US Letter, Arial, 22 mm margins, section headings in #145214, subsection
headings in #1a7a1a, body in #222222, table headers white on dark green.

The reference sets body text at 11 pt. This deck lifts the floor to 12 pt as
requested, so every size here is scaled up from the reference rather than
copied.

    python build_report_pdf.py
"""

from __future__ import annotations

import base64
import subprocess
import sys
from pathlib import Path

import pymupdf

HERE = Path(__file__).parent
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

GREEN_DARK = "#145214"   # section headings and table header fill
GREEN = "#1a7a1a"        # subsection headings
INK = "#222222"
MUTED = "#555555"
RULE = "#8fae8f"
BAND = "#f1f7f1"         # alternating table row
BORDER = "#c9d6c9"


def logo_data_uri() -> str:
    p = HERE / "report_assets" / "kuet_logo.png"
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


CSS = f"""
@page {{ size: Letter; margin: 22mm 21mm 20mm 22mm; }}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; background: #fff; }}
body {{
  font-family: Arial, Helvetica, sans-serif;
  font-size: 12pt; line-height: 1.5; color: {INK};
}}
p {{ margin: 0 0 9pt; text-align: justify; }}
strong {{ font-weight: bold; }}

h2 {{
  font-size: 16pt; font-weight: bold; color: {GREEN_DARK};
  margin: 20pt 0 8pt; padding-bottom: 4pt;
  border-bottom: 1pt solid {RULE};
  page-break-after: avoid; break-after: avoid;
}}
h3 {{
  font-size: 13.5pt; font-weight: bold; color: {GREEN};
  margin: 14pt 0 6pt;
  page-break-after: avoid; break-after: avoid;
}}

ul, ol {{ margin: 0 0 9pt; padding-left: 26pt; }}
li {{ margin-bottom: 4pt; text-align: justify; }}

code {{ font-family: Consolas, "Courier New", monospace; font-size: 12pt; }}
.bn {{ font-family: "Nirmala UI", "Noto Serif Bengali", sans-serif; font-size: 12.5pt; }}

pre {{
  font-family: Consolas, "Courier New", monospace; font-size: 12pt;
  line-height: 1.42; background: {BAND}; border: 0.75pt solid {BORDER};
  border-left: 3pt solid {GREEN}; padding: 8pt 11pt; margin: 10pt 0;
  white-space: pre; overflow: hidden;
  page-break-inside: avoid; break-inside: avoid;
}}
pre .c {{ color: {MUTED}; }}

table {{
  width: 100%; border-collapse: collapse; margin: 9pt 0 0;
  font-size: 12pt; page-break-inside: avoid; break-inside: avoid;
}}
th {{
  background: {GREEN_DARK}; color: #fff; font-weight: bold;
  text-align: left; padding: 5pt 7pt; border: 0.75pt solid {GREEN_DARK};
}}
td {{ padding: 5pt 7pt; border: 0.75pt solid {BORDER}; vertical-align: top; }}
tbody tr:nth-child(even) td {{ background: {BAND}; }}
td.n, th.n {{ text-align: right; }}
tr.total td {{ font-weight: bold; background: #e2eee2; }}

figure {{ margin: 12pt 0 14pt; page-break-inside: avoid; break-inside: avoid; }}
figcaption {{
  font-size: 12pt; color: {INK}; text-align: center; margin-top: 7pt;
}}

.callout {{
  border: 0.75pt solid {BORDER}; border-left: 3pt solid {GREEN};
  background: {BAND}; padding: 8pt 11pt; margin: 10pt 0;
  page-break-inside: avoid; break-inside: avoid;
}}
.callout h4 {{ margin: 0 0 5pt; font-size: 12.5pt; font-weight: bold; color: {GREEN_DARK}; }}
.callout p:last-child {{ margin-bottom: 0; }}

.quote {{
  border: 0.75pt solid {BORDER}; border-left: 3pt solid {GREEN};
  padding: 8pt 11pt; margin: 10pt 0; page-break-inside: avoid; break-inside: avoid;
}}
.quote .gloss {{ font-size: 12pt; color: {MUTED}; margin: 5pt 0 0; text-align: left; }}

/* five stage pipeline figure */
.flow {{ display: table; width: 100%; table-layout: fixed; border-collapse: collapse; }}
.flow > div {{
  display: table-cell; border: 0.75pt solid {BORDER}; background: {BAND};
  padding: 7pt 6pt; text-align: center; vertical-align: top;
}}
.flow .s {{ display: block; font-size: 12pt; font-weight: bold; color: {GREEN}; }}
.flow .n {{ display: block; font-size: 12pt; font-weight: bold; margin-top: 3pt; }}
.flow .v {{ display: block; font-size: 12pt; color: {MUTED}; margin-top: 3pt; }}

/* ---------------- title page ---------------- */
.cover {{ page-break-after: always; break-after: page; text-align: center; padding-top: 0; }}
.cover img {{ width: 108pt; height: auto; display: block; margin: 0 auto; }}
.cover .uni {{ font-size: 17pt; font-weight: bold; color: {GREEN_DARK}; margin-top: 12pt; }}
.cover .dept {{ font-size: 13pt; color: {INK}; margin-top: 5pt; }}
.cover .rule {{ border-top: 1pt solid {RULE}; margin: 15pt 0; }}
.cover .title {{ font-size: 24pt; font-weight: bold; color: {GREEN_DARK}; line-height: 1.2; }}
.cover .sub {{ font-size: 13.5pt; color: {INK}; margin-top: 9pt; }}
.cover table {{ margin-top: 0; }}
.cover td {{ text-align: left; padding: 7pt 11pt; }}
.cover .lbl {{ font-weight: bold; color: {GREEN_DARK}; font-size: 12pt; }}
.cover .val {{ font-size: 12pt; }}
.cover .hd {{
  font-weight: bold; color: {GREEN_DARK}; font-size: 12.5pt;
  text-decoration: underline; margin-bottom: 6pt; display: block;
}}
.cover .person {{ font-weight: bold; font-size: 12pt; margin-top: 4pt; }}
.cover .role {{ font-size: 12pt; color: {MUTED}; }}
.spacer {{ height: 13pt; }}
"""


def build_html() -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Pipeline Report</title>
<style>{CSS}</style></head><body>

<div class="cover">
  <img src="{logo_data_uri()}" alt="KUET">
  <div class="uni">Khulna University of Engineering &amp; Technology</div>
  <div class="dept">Department of Computer Science and Engineering</div>
  <div class="rule"></div>
  <div class="title">Speaker-Attributed Financial Discourse<br>Analysis of Bangla Talk Shows</div>
  <div class="sub">A Five Stage Diarization, Recognition and Target-Aware Sentiment Pipeline</div>
  <div class="rule"></div>

  <table><tr>
    <td width="50%">
      <span class="lbl">Course No:</span><br><span class="val">CSE 4112</span><br>
      <span class="lbl">Course Title:</span><br><span class="val">Machine Learning Laboratory</span>
    </td>
    <td width="50%">
      <span class="lbl">Submission Date:</span><br><span class="val">September 7, 2026</span><br>
      <span class="lbl">Corpus:</span><br><span class="val">26 episodes, 16.88 hours</span>
    </td>
  </tr></table>

  <div class="spacer"></div>

  <table><tr>
    <td width="50%">
      <span class="hd">Submitted By</span>
      <div class="person">Roll: 2107001</div>
      <div class="person">Roll: 2107004</div>
      <div class="person">Roll: 2107006</div>
      <div class="person">Roll: 2107009</div>
      <div class="person">Roll: 2107010</div>
      <div class="person">Roll: 2107015</div>
    </td>
    <td width="50%">
      <span class="hd">Submitted To</span>
      <div class="person">Dr. Muhammad Aminul Haque Akhand</div>
      <div class="role">Professor<br>Dept. of CSE, KUET</div>
      <div class="person">Nabil Faiyaz Sadi</div>
      <div class="role">Lecturer<br>Dept. of CSE, KUET</div>
    </td>
  </tr></table>
</div>

<h2>1. Abstract</h2>
<p>This report documents a five stage pipeline that turns raw Bangla business talk show audio into
speaker-attributed polarity judgements about named financial targets. The speech path conditions
broadcast audio to 16 kHz mono at a controlled loudness, segments it by speaker with pyannote,
transcribes it with a Bengali adapted Whisper model in 28 second chunks, and fuses words to speakers by
timestamp overlap, producing 3,532 speaker-attributed utterances. The language path detects financial
targets with a 52 entry stem lexicon, assigns polarity with a target conditioned rule set, and trains a
classifier on the resulting labels.</p>

<p>Evaluation uses 180 target pairs read and labelled by hand. The rule set reaches a macro F1 of 0.584
and a Cohen kappa of 0.392 against those labels, while a character n-gram classifier trained on its
output reaches 0.466 on held out episodes, below the rule set that supervised it. The report also
records two transcript defects found by direct inspection, affecting 11.3 and 1.3 percent of
utterances, and identifies the label rules rather than transcript noise as the current bottleneck.</p>

<h2>2. System Architecture</h2>
<p>The system is a cascade of five stages. Each stage reads the durable output of the previous one and
writes its own, so any stage can be rerun, inspected or replaced without disturbing the others. No
state is held in memory between stages.</p>

<figure>
  <div class="flow">
    <div><span class="s">Stage 1</span><span class="n">Audio</span><span class="v">16 kHz mono</span></div>
    <div><span class="s">Stage 2</span><span class="n">Diarization</span><span class="v">1,867 turns</span></div>
    <div><span class="s">Stage 3</span><span class="n">Recognition</span><span class="v">130,750 words</span></div>
    <div><span class="s">Stage 4</span><span class="n">Fusion</span><span class="v">3,532 utterances</span></div>
    <div><span class="s">Stage 5</span><span class="n">Polarity</span><span class="v">3,109 pairs</span></div>
  </div>
  <figcaption>Fig 1: The five stage cascade</figcaption>
</figure>

<p>Stages 2 and 3 are independent of each other: diarization never sees the transcript and recognition
never sees the speaker labels. This is a deliberate design property. A failure in one does not corrupt
the other, and either can be rerun in isolation. It also means the two stages disagree about time
boundaries, and reconciling that disagreement is the entire job of stage 4, which is the first point
where both are required and therefore the point where errors compound.</p>

<h2>3. Data</h2>
<p>The corpus is 26 full episodes of Bangla business and economics talk shows drawn from six
broadcasters, contributed by six group members. Episodes run 24 to 48 minutes. The material is
deliberately difficult: panel discussion with frequent crosstalk, studio music between segments, guests
joining by telephone at reduced bandwidth, and heavy code switching in which English finance terms
appear inside Bangla sentences.</p>

<figure>
  <table>
    <thead><tr><th>Roll</th><th class="n">Episodes</th><th class="n">Words</th><th>Programmes</th></tr></thead>
    <tbody>
      <tr><td>2107001</td><td class="n">4</td><td class="n">21,500</td><td>Business Talk, ATN</td></tr>
      <tr><td>2107004</td><td class="n">6</td><td class="n">29,383</td><td>Star Biz, Dhaka Stream</td></tr>
      <tr><td>2107006</td><td class="n">5</td><td class="n">25,736</td><td>RTV Business Talk</td></tr>
      <tr><td>2107009</td><td class="n">5</td><td class="n">23,447</td><td>Money Talks, Desh TV</td></tr>
      <tr><td>2107010</td><td class="n">1</td><td class="n">2,960</td><td>Star Biz Dialogue</td></tr>
      <tr><td>2107015</td><td class="n">5</td><td class="n">27,724</td><td>ATN, Gtv</td></tr>
      <tr class="total"><td>Total</td><td class="n">26</td><td class="n">130,750</td><td>16.88 hours</td></tr>
    </tbody>
  </table>
  <figcaption>Table 1: Corpus composition by contributor</figcaption>
</figure>

<p>Episode identifiers are namespaced by roll number, so <code>2107001::ep001</code> and
<code>2107006::ep001</code> remain distinct entities throughout. Speaker identifiers are episode local
by construction and are never merged across recordings: speaker 2 in one episode has no relationship to
speaker 2 in another. They are cluster indices, not people.</p>

<h2>4. Preprocessing: Audio Conditioning</h2>
<p>Both downstream speech models expect 16 kHz mono at a predictable loudness. Broadcast MP3 satisfies
neither condition, and levels vary substantially between channels. Stage 1 resamples 44.1 and 48 kHz
stereo sources to 16 kHz mono, then measures integrated loudness to the EBU R128 standard against a
target of -16 LUFS. Measured values across the corpus span -28.3 to -16.0 LUFS.</p>

<p>Gain is applied selectively rather than universally. An episode is adjusted only when it is genuinely
off target and its true peak leaves headroom; roughly one third of measured episodes were changed and
the rest were left untouched. Normalising every file blindly would clip the loud ones and introduce
distortion that the speech models would then have to work around.</p>

<div class="callout">
  <h4>Why loudness matters two stages later</h4>
  <p>Diarization clusters speaker embeddings by distance. If one episode arrives 12 dB quieter than
  another, its embeddings shift and a clustering threshold tuned on the first recording stops working on
  the second. Controlling loudness first is what allows one threshold configuration to apply across all
  26 episodes rather than requiring per episode tuning.</p>
</div>

<h2>5. Speaker Diarization and Postprocessing</h2>
<p>Stage 2 uses the pyannote speaker-diarization-community-1 pipeline with the default speaker embedding
replaced by wespeaker-voxceleb-resnet34-LM, which separates voices more reliably on noisy broadcast
audio. Voice activity uses a minimum off duration of 0.1 s and clustering uses a minimum cluster size of
20.</p>

<p>The raw output of any diarizer on broadcast audio is fragmented and contains spurious speakers, so a
postprocessing pass is applied. Both the raw and the postprocessed results are retained, so the effect
of each threshold can be inspected rather than assumed.</p>

<figure>
  <table>
    <thead><tr><th>Postprocessing step</th><th class="n">Threshold</th><th>Purpose</th></tr></thead>
    <tbody>
      <tr><td>Minimum segment duration</td><td class="n">0.75 s</td><td>A cough or a jingle does not become a turn</td></tr>
      <tr><td>Gap stitching</td><td class="n">0.17 s</td><td>One sentence does not split into three turns</td></tr>
      <tr><td>Segment merge threshold</td><td class="n">3.79</td><td>Adjacent turns of one speaker are combined</td></tr>
      <tr><td>Minimum speaker airtime</td><td class="n">9.0 s</td><td>Spurious clusters are deleted entirely</td></tr>
      <tr><td>Scoring collar</td><td class="n">0.25 s</td><td>Standard allowance for boundary imprecision</td></tr>
    </tbody>
  </table>
  <figcaption>Table 2: Diarization postprocessing thresholds</figcaption>
</figure>

<p>The stage produces 1,867 speaker turns across the corpus, with a median of 3 speakers per episode and
a range of 1 to 7, which matches the expected format of a host with two or three panellists.</p>

<div class="callout">
  <h4>No diarization error rate is reported</h4>
  <p>Diarization error rate requires a human made reference of who spoke when. No such reference exists
  for this corpus, and scoring automatic output against itself would be meaningless. The figure is left
  blank rather than estimated. Section 12 gives the coverage statistic reported in its place and states
  precisely what that statistic does and does not claim.</p>
</div>

<h2>6. Speech Recognition</h2>
<p>Stage 3 uses a Bengali adapted Whisper medium checkpoint converted to CTranslate2 and run in float16
on GPU. Whisper reads a fixed 30 second receptive field, so transcribing a 40 minute episode is
fundamentally a chunking and stitching problem wrapped around a short context model. The corpus was
decoded in 2,184 chunks producing 4,006 segments and 130,750 words with word level timestamps.</p>

<figure>
  <table>
    <thead><tr><th>Setting</th><th class="n">Value</th><th>Reason</th></tr></thead>
    <tbody>
      <tr><td>Chunk length</td><td class="n">28.0 s</td><td>Leaves headroom inside the 30 s field, so the window edge never sits at the model limit</td></tr>
      <tr><td>Language</td><td class="n">bn, forced</td><td>Stops the detector switching to Hindi or Assamese on a noisy chunk</td></tr>
      <tr><td>Beam size</td><td class="n">5</td><td>Standard accuracy and speed compromise</td></tr>
      <tr><td>Condition on previous text</td><td class="n">False</td><td>Stops a hallucination in one chunk entering the next as context</td></tr>
      <tr><td>Source separation</td><td class="n">On detection</td><td>Spectral flux is measured per chunk; music is separated only when present</td></tr>
      <tr><td>Repetition penalty</td><td class="n">0.8</td><td>Below 1.0, which encourages repetition rather than suppressing it. See section 7.2</td></tr>
    </tbody>
  </table>
  <figcaption>Table 3: Decoding configuration as recorded in the run manifest</figcaption>
</figure>

<p>Disabling conditioning on previous text is the most consequential of these choices. With it enabled, a
single hallucinated phrase is fed forward as context and the decoder can produce a corrupted minute of
transcript from one bad chunk. Disabling it costs some fluency across boundaries but bounds the damage
of any single failure to the chunk that produced it.</p>

<p>The choice of a Bengali adapted rather than stock multilingual checkpoint matters most for the
financial vocabulary the rest of the system depends on. It produces <span class="bn">মূল্যস্ফীতি</span>,
<span class="bn">খেলাপি ঋণ</span> and <span class="bn">বৈদেশিক মুদ্রার রিজার্ভ</span> as clean tokens
rather than phonetic approximations, which the target lexicon in section 9 requires in order to match
anything at all.</p>

<h2>7. Transcript Quality Audit</h2>
<p>Every stage reported success, so the transcripts were inspected directly rather than trusted. Two
systematic defects were found and quantified across all 3,532 utterances.</p>

<h3>7.1 Chunk boundary truncation</h3>
<p>Bangla orthography never begins a word with a dependent vowel sign. When a segment does, the chunk
stitcher has removed the opening consonant. This makes detection mechanical rather than a matter of
judgement, so the count is exact: 400 utterances, or 11.3 percent of the corpus, begin with a token that
is not a valid word.</p>

<div class="quote">
  <div class="bn">ুক্তা উন্নতি লক্ষ্য করা যাচ্ছে</div>
  <p class="gloss">The opening word should be <span class="bn">কিছুটা</span>. The leading consonant has
  been consumed at a chunk boundary. Worst affected episode: 21.2 percent of its utterances.</p>
</div>

<p>The cause is that chunks are cut back to back with no overlap. Cutting with a 2 to 3 second overlap
and stitching on the shared region removes the boundary word entirely. This is a rerun configuration
change, not new code.</p>

<h3>7.2 Decoder repetition loops</h3>
<p>In 46 utterances, or 1.3 percent, the decoder locks onto a single token and emits it for many seconds.
Detection uses two signals together: the longest run of one repeated token, and the unique token ratio
across a long span. The run catches a single repeated token and the ratio catches an alternating loop.</p>

<div class="quote">
  <div class="bn">বাজার বাজার বাজার বাজার বাজার বাজার</div>
  <p class="gloss">These carry no readable meaning. They are excluded from supervision rather than
  quietly labelled neutral, which would teach the model that meaningless text is non evaluative.</p>
</div>

<div class="callout">
  <h4>Probable cause</h4>
  <p>The run sets a repetition penalty of 0.8. A penalty above 1.0 suppresses tokens that have already
  been produced; a value below 1.0 does the opposite and makes repeating them more likely. Raising the
  value above 1.0, or adding a no repeat n-gram constraint, is the direct remedy. This has not yet been
  confirmed by a rerun, so it is reported as the probable rather than the established cause.</p>
</div>

<h2>8. Fusion: Attaching Words to Speakers</h2>
<p>Diarization produces time intervals with speaker labels. Recognition produces words with timestamps.
Neither knows about the other, so stage 4 must decide which speaker owns each word.</p>

<ol>
  <li>Take the midpoint of the word. If it falls inside one speaker interval, that speaker owns the word.</li>
  <li>If the midpoint falls in a gap between intervals, fall back to the interval with the largest
  temporal overlap with the word.</li>
  <li>If two speaker intervals overlap in time, the speaker who started first keeps the word, so no word
  is ever counted twice.</li>
  <li>Consecutive words from the same speaker are grouped into an utterance, which becomes the unit of
  analysis for everything downstream.</li>
</ol>

<p>The midpoint is used rather than the word start because start timestamps drift at chunk boundaries,
where the truncation defect of section 7.1 also originates, and the midpoint is the more stable of the
two anchors. The result is 3,532 speaker-attributed utterances carrying speaker, timing and text
together, which is what allows any later polarity judgement to be traced back to a specific person at a
specific second of a specific episode.</p>

<p>Word to speaker attribution reaches 99.5 percent across the rolls with diarization completed. This is
a coverage statistic: it states that a word was attached to some automatic speaker interval. It is not a
claim that the attributed speaker is correct, which would require the human reference discussed in
section 5.</p>

<h2>9. Target Detection</h2>
<p>Polarity is meaningless without a target, so stage 5 first identifies the financial entity or
indicator under discussion. Bangla morphology makes this harder than a string match, because case and
definiteness are attached as suffixes. The word for bank appears in the corpus as six distinct surface
forms totalling 555 occurrences; exact matching treats them as six unrelated words.</p>

<p>Matching therefore operates on stem prefixes over whitespace tokens, so one stem covers the entire
inflectional family. A prefix rule is blunt, so each lexicon entry carries an exclusion list. Without
one, the interest rate stem <span class="bn">সুদ</span> matches the country name
<span class="bn">সুদান</span>, and the stem for good, <span class="bn">ভালো</span>, matches the word for
love, <span class="bn">ভালোবাসা</span>. Longer stems claim their spans before shorter ones, so a specific
target is preferred over a generic word nested inside it.</p>

<figure>
  <table>
    <thead><tr><th>Target type</th><th class="n">Pairs</th><th>Examples</th></tr></thead>
    <tbody>
      <tr><td>MACRO_INDICATOR</td><td class="n">1,121</td><td class="bn">মূল্যস্ফীতি, রিজার্ভ, বিনিয়োগ</td></tr>
      <tr><td>REGULATOR</td><td class="n">980</td><td class="bn">বাংলাদেশ ব্যাংক, বাজেট</td></tr>
      <tr><td>SECTOR</td><td class="n">901</td><td class="bn">গ্যাস, কৃষি, শিল্প</td></tr>
      <tr><td>MARKET</td><td class="n">107</td><td class="bn">শেয়ারবাজার, পুঁজিবাজার</td></tr>
      <tr class="total"><td>Total</td><td class="n">3,109</td><td>52 lexicon entries, 48 seen in corpus</td></tr>
    </tbody>
  </table>
  <figcaption>Table 4: Target detection output</figcaption>
</figure>

<h2>10. Polarity Assignment</h2>

<h3>10.1 Target conditioned direction</h3>
<p>The central modelling decision is that direction words are not polar on their own. Ordinary sentiment
analysis treats an increase word as positive, which in economic discourse is wrong roughly half the
time. Each target therefore carries an orientation, and a direction cue is multiplied by it.</p>

<figure>
  <table>
    <thead><tr><th class="n">Orientation</th><th class="n">Targets</th><th>Meaning</th><th>Examples</th></tr></thead>
    <tbody>
      <tr><td class="n">+1</td><td class="n">14</td><td>A rise is good news</td><td>reserves, remittance, GDP, exports, investment, employment</td></tr>
      <tr><td class="n">-1</td><td class="n">9</td><td>A rise is bad news</td><td>inflation, consumer prices, bad loans, foreign debt, capital flight</td></tr>
      <tr><td class="n">0</td><td class="n">29</td><td>An entity, no direction</td><td>banks, the government, the energy sector</td></tr>
    </tbody>
  </table>
  <figcaption>Table 5: Target orientation</figcaption>
</figure>

<p>So <span class="bn">রিজার্ভ বেড়েছে</span>, reserves have increased, resolves positive, while
<span class="bn">মূল্যস্ফীতি বেড়েছে</span>, inflation has increased, resolves negative from the same cue.
Because each target is scored independently, one utterance can carry two different labels at once. A
sentence level model cannot represent that distinction.</p>

<h3>10.2 Scoring and postprocessing</h3>
<p>Supervision comes from a written rule set rather than a black box, so every resulting label can be
traced to the tokens that produced it. Each of 104 polarity cue stems carries a weight, a kind
(evaluative or direction) and a negation flag.</p>

<pre><span class="c"># every cue token within 12 tokens of the target</span>
polarity = cue.polarity
if cue.kind == "direction":
    if orientation == 0: skip
    polarity *= orientation
if cue.negatable and negator within 3 tokens:
    polarity *= -1
decay = 1 / (1 + distance / 4)
score += polarity * cue.weight * decay</pre>

<p>The score is then thresholded: above +0.55 is positive, below -0.55 is negative, and the band between
is non evaluative. Two postprocessing rules proved necessary and were found by reading the output.</p>

<ul>
  <li><strong>Negation has scope.</strong> A first version flipped any cue near a negator and scored
  crisis as positive in <span class="bn">সংকট কাটছে না</span>, the crisis is not passing. The particle
  negates the verb, not the noun, and the sentence is still bad news. Only adjectives and verbs are now
  marked negatable; polar nouns are not.</li>
  <li><strong>Bangla builds a negated verb complex.</strong> The particle can sit three tokens from the
  word it negates, as in <span class="bn">স্থিতিশীল করতে পারবেন না</span>, you will not be able to
  stabilise it. The negation window was widened from one token to three.</li>
</ul>

<p>Utterances flagged as repetition loops, and those under six words, are excluded from supervision
rather than labelled neutral. That removes 90 of 3,109 pairs and leaves 3,019 labelled pairs, of which
284 are negative at 9.4 percent, 2,583 are non evaluative at 85.6 percent, and 152 are positive at 5.0
percent. This imbalance governs the choice of metric in section 12 and the results in section 13.</p>

<h2>11. Dataset Construction</h2>
<p>The corpus is partitioned by episode, never by row. Two utterances from the same talk show share the
panel, the topic, the host's phrasing and the same recognition error pattern. A random row split would
place utterances from one recording on both sides of the partition, allowing a model to recognise the
episode rather than the sentiment. The reported score would rise and none of that rise would survive
contact with a new recording.</p>

<figure>
  <table>
    <thead><tr><th>Split</th><th class="n">Episodes</th><th class="n">Pairs</th><th class="n">Human</th><th class="n">Negative</th><th class="n">Non eval</th><th class="n">Positive</th></tr></thead>
    <tbody>
      <tr><td>Train</td><td class="n">19</td><td class="n">2,026</td><td class="n">99</td><td class="n">164</td><td class="n">1,771</td><td class="n">91</td></tr>
      <tr><td>Validation</td><td class="n">3</td><td class="n">320</td><td class="n">22</td><td class="n">24</td><td class="n">278</td><td class="n">18</td></tr>
      <tr class="total"><td>Test</td><td class="n">4</td><td class="n">673</td><td class="n">59</td><td class="n">96</td><td class="n">534</td><td class="n">43</td></tr>
    </tbody>
  </table>
  <figcaption>Table 6: Partition by episode</figcaption>
</figure>

<p>The split routine asserts that no episode identifier appears in more than one partition, so an editing
mistake fails loudly rather than silently inflating a score. Test episodes were selected to carry as
much of the hand labelled reference set as possible, so the headline figure is measured on human labels
from recordings the model has never seen. Validation is used only to select the stopping epoch and is
never fitted on.</p>

<h2>12. Evaluation Methodology</h2>
<p>No reference standard existed, so one was created: a stratified sample of 180 target pairs was read in
context and labelled by hand, of which 59 fall inside the held out test episodes. The sample is
stratified, so its accuracy is not corpus accuracy; macro F1 is the comparable figure.</p>

<p>The class balance governs metric choice. A model that answers non evaluative for every row scores 85.6
percent accuracy on this corpus while having learned nothing, so accuracy is not used as a headline. The
reported metrics are:</p>

<ul>
  <li><strong>Macro F1</strong>, the headline. It averages F1 across the three classes with equal weight,
  so the 152 positive rows count as much as the 2,583 neutral ones. The all neutral model scores 0.31.</li>
  <li><strong>Per class precision and recall.</strong> Macro F1 conceals which class is failing.
  Separating it is what revealed that positive precision is the weak point.</li>
  <li><strong>Cohen kappa</strong>, agreement corrected for chance. Raw agreement of 59.4 percent sounds
  considerably better than the corrected value of 0.392, which is fair agreement rather than good.</li>
  <li><strong>Confusion matrix</strong>, the only view that shows the direction of the errors. A single
  F1 cannot distinguish a model that misses negatives from one that invents positives.</li>
</ul>

<p>Word error rate, diarization error rate and speaker attribution accuracy are deliberately not
reported. Each requires a human corrected reference that this project does not have, and computing them
from automatic output would compare the system against itself.</p>

<h2>13. Results</h2>

<figure>
  <table>
    <thead><tr><th>System</th><th class="n">Macro F1</th><th>Notes</th></tr></thead>
    <tbody>
      <tr><td>Majority class baseline</td><td class="n">0.177</td><td>Always answers non evaluative</td></tr>
      <tr><td>Character n-gram classifier</td><td class="n">0.466</td><td>Trained on rule labels, 2,026 rows</td></tr>
      <tr class="total"><td>Rule labeller</td><td class="n">0.566</td><td>The system that produced the training labels</td></tr>
      <tr><td>BanglaBERT pair model</td><td class="n">not run</td><td>Implemented, not yet trained</td></tr>
    </tbody>
  </table>
  <figcaption>Table 7: Macro F1 on the 59 held out human labelled pairs</figcaption>
</figure>

<p>The trained classifier scores below the rule set that generated its training labels, 0.466 against
0.566. This is the most important result in the report. A student model trained on noisy labels cannot
exceed its teacher unless it generalises past the teacher's mistakes, and with 2,026 training rows
containing only 91 positive examples it has no basis on which to do so.</p>

<figure>
  <table>
    <thead><tr><th>Class</th><th class="n">Precision</th><th class="n">Recall</th><th class="n">F1</th><th class="n">Support</th></tr></thead>
    <tbody>
      <tr><td>negative</td><td class="n">0.850</td><td class="n">0.607</td><td class="n">0.708</td><td class="n">84</td></tr>
      <tr><td>non evaluative</td><td class="n">0.517</td><td class="n">0.484</td><td class="n">0.500</td><td class="n">64</td></tr>
      <tr><td>positive</td><td class="n">0.417</td><td class="n">0.781</td><td class="n">0.543</td><td class="n">32</td></tr>
      <tr class="total"><td>Macro average</td><td class="n">0.594</td><td class="n">0.624</td><td class="n">0.584</td><td class="n">180</td></tr>
    </tbody>
  </table>
  <figcaption>Table 8: Rule labeller per class, all 180 pairs. Accuracy 0.594, Cohen kappa 0.392</figcaption>
</figure>

<p>Negative precision is high at 0.850: when the system reports negative sentiment it is usually correct.
Positive precision at 0.417 is where the failure concentrates. The confusion matrix locates it
precisely.</p>

<figure>
  <table>
    <thead><tr><th>Human label</th><th class="n">Said negative</th><th class="n">Said non eval</th><th class="n">Said positive</th></tr></thead>
    <tbody>
      <tr><td>is negative</td><td class="n">51</td><td class="n">22</td><td class="n">11</td></tr>
      <tr><td>is non evaluative</td><td class="n">9</td><td class="n">31</td><td class="n">24</td></tr>
      <tr><td>is positive</td><td class="n">0</td><td class="n">7</td><td class="n">25</td></tr>
    </tbody>
  </table>
  <figcaption>Table 9: Confusion matrix. 33 pairs read by a human as negative or neutral were reported positive</figcaption>
</figure>

<h2>14. Error Analysis</h2>
<p>The natural hypothesis is that recognition noise propagates downstream and degrades the sentiment
stage. This was tested by stratifying the 180 reference pairs and does not hold.</p>

<figure>
  <table>
    <thead><tr><th>Stratum</th><th class="n">n</th><th class="n">Macro F1</th></tr></thead>
    <tbody>
      <tr><td>Recognition confidence above median</td><td class="n">90</td><td class="n">0.550</td></tr>
      <tr><td>Recognition confidence below median</td><td class="n">90</td><td class="n">0.619</td></tr>
      <tr><td>No leading truncation</td><td class="n">161</td><td class="n">0.584</td></tr>
      <tr><td>Leading truncation present</td><td class="n">19</td><td class="n">0.562</td></tr>
      <tr><td>Short utterance, under 40 words</td><td class="n">37</td><td class="n">0.522</td></tr>
      <tr><td>Long utterance, 40 words or more</td><td class="n">143</td><td class="n">0.602</td></tr>
    </tbody>
  </table>
  <figcaption>Table 10: Performance by transcript quality stratum</figcaption>
</figure>

<p>The cleaner half scores no better than the noisier half. At 90 pairs per side the difference sits
inside the noise, so the honest reading is that recognition confidence does not separate sentiment
accuracy on this sample. Utterance length does correlate, which is expected: a 12 token window around
the target finds fewer cues in a short utterance. The bottleneck is the label rules, not the
transcripts. Three specific causes account for the errors.</p>

<h3>14.1 Prescription read as evaluation</h3>
<p>The largest single class, 24 of 73 errors. Bangla economic commentary is dense with prescription. A
speaker states what should happen and the rule set counts the favourable word as a judgement about the
present state.</p>

<div class="quote">
  <div class="bn">অর্থনীতি স্থিতিশীল করতে হবে</div>
  <p class="gloss">The economy must be stabilised. This is a demand, not a compliment. The system reads
  the word for stable and reports positive.</p>
</div>

<h3>14.2 Negative meaning with no cue word</h3>
<p>22 errors. The cue set holds 104 stems, and plain Bangla frequently conveys bad news without using any
of them.</p>

<div class="quote">
  <div class="bn">বিনিয়োগ বাইশ পয়েন্ট চার পার্সেন্টে নেমে গেছে</div>
  <p class="gloss">Investment has come down to 22.4 percent. Unambiguously negative, and not one word of
  it appears in the cue list.</p>
</div>

<h3>14.3 Negation carried by a suffix</h3>
<p>Bangla negates the past tense by attaching <span class="bn">নি</span> directly to the verb, so no
separate particle exists for the negation window to detect. In
<span class="bn">বাজেট বাস্তবায়ন হয়নি</span>, the budget was not implemented, the system sees the word
for implementation and reports positive.</p>

<h3>14.4 Structural cause</h3>
<p>Underlying all three is class imbalance. With 85.6 percent of rows non evaluative and only 91 positive
examples reaching the training set, there is far too little signal for a 110 million parameter model and
thin signal even for logistic regression.</p>

<h2>15. Limitations</h2>
<ul>
  <li>No word error rate, diarization error rate or speaker attribution accuracy, as no human reference
  transcript or speaker annotation exists. The 99.5 percent figure is attribution coverage, not accuracy.</li>
  <li>The 180 reference labels were assigned by a single reader. A second annotator and an inter
  annotator agreement figure are required before these can properly be called gold labels.</li>
  <li>The BanglaBERT pair model is implemented and its checkpoint loads and tokenises correctly, but it
  has not been trained. No neural result is claimed.</li>
  <li>The rule labels measure agreement with a rule set, not correctness. Only the 180 human labels
  measure correctness.</li>
  <li>The reference sample is stratified, so its accuracy figure is not corpus accuracy.</li>
  <li>26 episodes is a small corpus, and the positive class at 152 rows is its thinnest part.</li>
  <li>The repetition penalty diagnosis in section 7.2 is inferred from the configuration and has not been
  confirmed by a rerun.</li>
</ul>

<h2>16. Future Work</h2>
<p>Ordered by expected gain relative to effort.</p>
<ol>
  <li><strong>Hand label 800 to 1,000 pairs with two annotators.</strong> This removes the ceiling rather
  than raising it, because training on real labels frees the model from the teacher's bias. Two
  annotators also yield the agreement figure the project currently lacks. Six members at roughly 150
  pairs each is a single working session.</li>
  <li><strong>Correct the two recognition settings.</strong> Raise the repetition penalty above 1.0 to
  suppress decoder loops, and cut chunks with a 2 to 3 second overlap to eliminate the truncated first
  word in 11.3 percent of utterances. Both are configuration changes that clean the input for every
  subsequent stage.</li>
  <li><strong>Separate prescription from evaluation.</strong> The largest single error class. Bangla marks
  obligation with a small closed set of endings, chiefly <span class="bn">করতে হবে</span>,
  <span class="bn">উচিত</span>, <span class="bn">দরকার</span> and <span class="bn">প্রয়োজন</span>.
  Detecting these and suppressing the polarity claim is a short rule.</li>
  <li><strong>Handle suffix negation.</strong> Matching a verb final <span class="bn">নি</span> catches a
  class of sign inversions that currently enter the training labels uncorrected.</li>
  <li><strong>Train the sentence pair model.</strong> It reads whole phrases rather than character
  n-grams, so it can learn that a fall in investment is negative without that phrase appearing in a cue
  list. Worth running once real labels exist.</li>
  <li><strong>Extend the corpus.</strong> The pipeline runs end to end on new material without code
  changes, which makes additional episodes the cheapest route to strengthening the positive class.</li>
</ol>

<h2>17. References</h2>
<ol>
  <li>H. M. S. Tabib et al., Bengali-Loop: Community Benchmarks for Long-Form Bangla ASR and Speaker
  Diarization, 2026. Used for the long-form protocol, episode local speaker identifiers, first speaker
  overlap handling and evaluation stance.</li>
  <li>A. Radford et al., Robust Speech Recognition via Large-Scale Weak Supervision, 2022. Used for the
  Whisper architecture and its 30 second input constraint.</li>
  <li>M. S. Chowdhury and A. F. Chowdhury, Robust Long-Form Bangla Speech Processing, 2026. Used as
  evidence for selective source separation and diarization postprocessing.</li>
  <li>S. Hasan et al., Make It Hard to Hear, Easy to Learn, 2026. Used for the postprocessing thresholds
  represented in stage 2.</li>
  <li>A. Chowdhury et al., WhisperAlign, 2026. Used for long-form chunking and timestamp fusion choices.</li>
  <li>Bangla Diarizz, 2026. Used for diarization embedding, clustering and stitching guidance.</li>
  <li>csebuetnlp, BanglaBERT. Pretrained encoder used for the target aware sentence pair classifier.</li>
</ol>

</body></html>"""


def stamp_page_numbers(pdf: Path) -> int:
    """Chrome cannot draw a page number, so add them afterwards in the same
    position the reference report uses: bottom right, inside the margin."""
    doc = pymupdf.open(pdf)
    for i, page in enumerate(doc, start=1):
        r = page.rect
        page.insert_text(
            pymupdf.Point(r.width - 60, r.height - 40),
            str(i), fontname="helv", fontsize=12,
            color=(0.13, 0.13, 0.13),
        )
    n = len(doc)
    doc.saveIncr()
    doc.close()
    return n


def main() -> None:
    html = HERE / "report_print.html"
    html.write_text(build_html(), encoding="utf-8")

    pdf = HERE / "Bangla_Pipeline_Project_Report.pdf"
    if pdf.exists():
        try:
            pdf.unlink()
        except PermissionError:
            sys.exit(f"{pdf.name} is open in a viewer. Close it and rerun.")

    subprocess.run([
        str(CHROME), "--headless", "--disable-gpu", "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw", "--virtual-time-budget=12000",
        f"--print-to-pdf={pdf}", html.as_uri(),
    ], check=True, capture_output=True)

    pages = stamp_page_numbers(pdf)
    size = pdf.stat().st_size / 1024
    print(f"wrote {pdf.name}: {pages} pages, {size:.0f} KB, US Letter, 12 pt minimum")


if __name__ == "__main__":
    main()
