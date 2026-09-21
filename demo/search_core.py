"""Corpus discovery and Bangla speaker-attributed transcript search.

The module is deliberately UI-independent so it can be tested from the
command line and reused by a Colab/Gradio application.  It treats speaker IDs
as episode-local identifiers and namespaces every result by collection and
episode to prevent accidental cross-episode identity claims.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


SEMANTIC_MODEL = "intfloat/multilingual-e5-small"
SEMANTIC_MODEL_REVISION = "c007d7ef6fd86656326059b28395a7a03a7c5846"

ALL_COLLECTIONS = "All rolls"
ALL_EPISODES = "All episodes"
ALL_SPEAKERS = "All speakers"

AUDIO_SUFFIXES = {
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg",
    ".aac",
    ".mpeg",
    ".mpg",
    ".mpga",
    ".opus",
    ".wma",
    ".m4b",
}

# A stored episode duration and a probed file duration are treated as the same
# recording when they agree this closely.
DURATION_TOLERANCE_SEC = 2.0

# Below this share of speaker-attributed words an episode gets a loader warning.
LOW_ASSIGNMENT_WARNING = 0.75

ROLL_PATTERN = re.compile(r"\d{6,}")

# Superseded pipeline runs are archived next to the live bundle (the roll folders
# keep them under _source_runs/). They contain a complete fused/ stage, so
# indexing them registers a second bundle for the same roll and double-counts
# every episode. Skip archives and tooling directories when discovering bundles.
ARCHIVED_DIR_NAMES = frozenset({
    "_source_runs",
    "__pycache__",
    ".git",
    ".ipynb_checkpoints",
})


def _is_archived(path: Path) -> bool:
    return any(part in ARCHIVED_DIR_NAMES for part in path.parts)


# Exported bundles differ between members: some keep the canonical
# ``<bundle>/data/<stage>`` tree, others exported the stage folders directly
# under ``<bundle>/``. A few stage folders were also renamed on export.
STAGE_ALIASES: dict[str, tuple[str, ...]] = {
    "annotation_drafts": ("annotation_drafts", "drafts"),
    "weak_labels": ("weak_labels", "weak"),
}


SEARCH_REPLACEMENTS = {
    "ব্যাঙ্ক": "ব্যাংক",
    "ব্যাংকিং": "ব্যাংক",
    "রেমিটেন্স": "রেমিট্যান্স",
    "রেমিটেন্যান্স": "রেমিট্যান্স",
    "banking sector": "ব্যাংক খাত",
    "bank sector": "ব্যাংক খাত",
    "banking": "ব্যাংক",
    "remittances": "remittance",
    "remittance": "রেমিট্যান্স",
    "central bank": "বাংলাদেশ ব্যাংক",
    "inflation": "মূল্যস্ফীতি",
    "imports": "আমদানি",
    "import": "আমদানি",
    "exports": "রপ্তানি",
    "export": "রপ্তানি",
}


def normalize_text(text: str) -> str:
    """Normalize Bangla/English transcript text for retrieval only."""

    value = unicodedata.normalize("NFC", str(text or "")).casefold()
    for source, target in SEARCH_REPLACEMENTS.items():
        value = value.replace(source, target)
    value = re.sub(r"[^\w\u0980-\u09ff]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def format_timestamp(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _speaker_label(speaker_id: int) -> str:
    return "Unassigned" if speaker_id < 0 else f"Speaker {speaker_id}"


@dataclass(slots=True)
class UtteranceRecord:
    key: str
    collection_id: str
    episode_id: str
    global_episode_id: str
    utterance_id: str
    speaker_id: int
    start_sec: float
    end_sec: float
    text: str
    text_verbatim: str
    normalized_text: str
    programme: str = ""
    channel: str = ""
    audio_path: str = ""
    diarization_provider: str = ""
    asr_model: str = ""
    word_count: int = 0
    mean_word_confidence: float | None = None
    targets: list[dict[str, Any]] = field(default_factory=list)
    previous_text: str = ""
    next_text: str = ""

    @property
    def speaker_reference(self) -> str:
        return f"{self.global_episode_id} / {_speaker_label(self.speaker_id)}"

    @property
    def timestamp(self) -> str:
        return f"{format_timestamp(self.start_sec)}–{format_timestamp(self.end_sec)}"


@dataclass(slots=True)
class EpisodeSummary:
    collection_id: str
    episode_id: str
    global_episode_id: str
    programme: str
    channel: str
    duration_sec: float
    speaker_count: int
    utterance_count: int
    word_count: int
    unassigned_words: int
    assignment_rate: float
    diarization_provider: str
    asr_model: str
    audio_path: str


@dataclass(slots=True)
class SearchHit:
    key: str
    collection_id: str
    episode_id: str
    global_episode_id: str
    utterance_id: str
    speaker_id: int
    speaker_reference: str
    start_sec: float
    end_sec: float
    timestamp: str
    match_type: str
    score: float
    lexical_score: float
    semantic_score: float | None
    text: str
    context: str
    programme: str
    mean_word_confidence: float | None
    audio_available: bool
    targets: str


def resolve_stage_dir(bundle_root: Path, stage: str) -> Path:
    """Resolve a pipeline stage folder inside an exported bundle.

    Accepts both ``<bundle>/data/<stage>`` and ``<bundle>/<stage>`` and the
    stage-name aliases listed in :data:`STAGE_ALIASES`. When nothing exists the
    canonical path is returned so callers keep failing the usual ``is_dir()``
    check instead of raising.
    """

    for name in STAGE_ALIASES.get(stage, (stage,)):
        for candidate in (bundle_root / "data" / name, bundle_root / name):
            if candidate.is_dir():
                return candidate
    return bundle_root / "data" / stage


def _bundle_root_for_stage_dir(stage_dir: Path) -> Path:
    parent = stage_dir.parent
    return parent.parent if parent.name.casefold() == "data" else parent


def _canonical_registry(raw_dir: Path) -> Path | None:
    canonical = raw_dir / "registry.csv"
    if canonical.exists():
        return canonical
    candidates = sorted(raw_dir.glob("registry*.csv"), key=lambda p: (len(p.name), p.name))
    return candidates[0] if candidates else None


def _read_csv_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalized_filename(name: str) -> str:
    """Normalization- and case-insensitive key for a stored file name."""

    return unicodedata.normalize("NFC", str(name)).casefold()


def _ffmpeg_exe() -> str | None:
    """Path to an ffmpeg binary, preferring PATH then the bundled wheel.

    ``imageio-ffmpeg`` is a declared dependency and installs its binary
    outside PATH, so a PATH-only lookup reports "no ffmpeg" on a machine
    that has a perfectly usable one.
    """

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg

        candidate = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError):
        return None
    return candidate if candidate and Path(candidate).is_file() else None


def _ffprobe() -> str | None:
    probe = shutil.which("ffprobe")
    if probe:
        return probe
    # ffprobe normally sits beside ffmpeg. The imageio-ffmpeg wheel is the
    # exception: it ships ffmpeg alone, which _duration_via_ffmpeg handles.
    ffmpeg = _ffmpeg_exe()
    if ffmpeg:
        candidate = Path(ffmpeg).with_name("ffprobe" + Path(ffmpeg).suffix)
        if candidate.is_file():
            return str(candidate)
    return None


_FFMPEG_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2}(?:\.\d+)?)")


def _duration_via_ffmpeg(path: Path) -> float | None:
    """Read a media duration from ``ffmpeg -i`` when ffprobe is unavailable.

    ffmpeg writes the container header to stderr and exits non-zero because
    no output file was given; the banner is still the information we want.
    """

    ffmpeg = _ffmpeg_exe()
    if not ffmpeg:
        return None
    completed = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    match = _FFMPEG_DURATION_RE.search(completed.stderr or "")
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    total = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return total if total > 0 else None


_DURATION_CACHE: dict[tuple[str, int, int], float | None] = {}


def probe_duration(path: Path) -> float | None:
    """Media duration in seconds, or ``None`` when ffprobe cannot report one."""

    try:
        stat = path.stat()
    except OSError:
        return None
    key = (str(path), stat.st_size, int(stat.st_mtime))
    if key in _DURATION_CACHE:
        return _DURATION_CACHE[key]
    duration: float | None = None
    probe = _ffprobe()
    if probe:
        completed = subprocess.run(
            [
                probe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            value = _safe_float(completed.stdout.strip(), -1.0)
            duration = value if value > 0 else None
    if duration is None:
        duration = _duration_via_ffmpeg(path)
    _DURATION_CACHE[key] = duration
    return duration


def _audio_files_under(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in AUDIO_SUFFIXES
    )


def _match_by_duration(
    root: Path, duration_sec: float, *, exclude: set[Path] | None = None
) -> Path | None:
    """Find the recording whose real duration matches the registry entry.

    Members sometimes re-encode or renumber their audio before uploading, so
    the stored filename no longer exists. The registry duration identifies the
    recording without trusting a filename or a positional guess, which would
    silently attach the wrong audio to a transcript.
    """

    if duration_sec <= 0:
        return None
    best: tuple[float, Path] | None = None
    taken = exclude or set()
    for path in _audio_files_under(root):
        if path.resolve() in taken:
            continue
        probed = probe_duration(path)
        if probed is None:
            continue
        delta = abs(probed - duration_sec)
        if delta <= DURATION_TOLERANCE_SEC and (best is None or delta < best[0]):
            best = (delta, path)
    return best[1].resolve() if best else None


def _resolve_audio_path(
    local_path: str,
    *,
    bundle_root: Path,
    data_root: Path,
    episode_id: str,
    duration_sec: float = 0.0,
    claimed: set[Path] | None = None,
) -> Path | None:
    """Locate the recording an episode was transcribed from.

    ``claimed`` collects the files already handed to earlier episodes of the
    same bundle. Only the inexact strategies below consult it: two episodes of
    one programme can share a runtime to the centisecond, and without this a
    duration match would hand the second episode the first one's audio, so the
    demo would play one episode while displaying another's transcript.
    """

    source = Path(str(local_path or ""))
    basename = source.name
    contributor_root = _contributor_root(bundle_root, data_root)
    candidates: list[Path] = []
    if source.is_absolute():
        candidates.append(source)
    if local_path:
        candidates.extend(
            [
                bundle_root / source,
                data_root / source,
                data_root.parent / source,
                bundle_root / "audio" / basename,
                contributor_root / basename,
                contributor_root / "audio" / basename,
                data_root / basename,
            ]
        )
        # Older Colab registries stored paths such as ``datasets/01.mp3``.
        # In the roll-number layout that file now lives at
        # ``datasets/<roll>/01.mp3``. Resolve by basename without changing the
        # immutable registry artifact.
        if source.parts and source.parts[0].casefold() == "datasets":
            candidates.append(contributor_root.joinpath(*source.parts[1:]))
    def first_existing(paths: list[Path]) -> Path | None:
        seen: set[Path] = set()
        for candidate in paths:
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.is_file():
                return resolved
        return None

    exact = first_existing(candidates)
    if exact is not None:
        return exact

    # Bangla filenames survive a Drive/Windows round trip in a different
    # Unicode normal form, so an exact byte match can fail for a file that is
    # present. Compare normalized basenames before giving up.
    #
    # The roll's own folder is searched first so a roll always prefers its own
    # copy; the whole corpus is only searched afterwards, because members share
    # recordings and some registries point at a file another roll uploaded.
    taken = claimed if claimed is not None else set()
    wanted = _normalized_filename(basename) if basename else ""
    for root in (contributor_root, data_root):
        if not wanted:
            break
        for path in _audio_files_under(root):
            if _normalized_filename(path.name) == wanted and path.resolve() not in taken:
                return path.resolve()

    # The filename is gone (renamed or re-encoded on upload). Identify the
    # recording by its duration instead.
    for root in (contributor_root, data_root):
        matched = _match_by_duration(root, duration_sec, exclude=taken)
        if matched is not None:
            return matched

    # Last resort: the ``01.mp3`` numbering convention some registries use.
    # The numbering does not always follow the episode order, so a candidate
    # found this way is rejected when its real duration contradicts the
    # registry. It is accepted when no duration can be read on either side.
    numeric = re.search(r"(\d+)$", episode_id)
    if numeric:
        number = int(numeric.group(1))
        numbered: list[Path] = []
        for suffix in sorted(AUDIO_SUFFIXES):
            numbered.extend(
                [
                    bundle_root / "audio" / f"{number:02d}{suffix}",
                    contributor_root / f"{number:02d}{suffix}",
                    contributor_root / "audio" / f"{number:02d}{suffix}",
                    data_root / f"{number:02d}{suffix}",
                    data_root / f"{number}{suffix}",
                ]
            )
        guess = first_existing([path for path in numbered if path.resolve() not in taken])
        if guess is not None:
            probed = probe_duration(guess) if duration_sec > 0 else None
            if probed is None or abs(probed - duration_sec) <= DURATION_TOLERANCE_SEC:
                return guess
    return None


def _discover_bundle_roots(data_root: Path) -> list[Path]:
    roots: set[Path] = set()
    candidates = [data_root / "data" / "fused", data_root / "fused"]
    candidates.extend(data_root.rglob("fused"))
    for fused_dir in candidates:
        if _is_archived(fused_dir):
            continue
        if fused_dir.is_dir() and any(fused_dir.glob("*.json")):
            roots.add(_bundle_root_for_stage_dir(fused_dir).resolve())
    return sorted(roots)


def _collection_id(bundle_root: Path, data_root: Path) -> str:
    try:
        relative_path = bundle_root.relative_to(data_root.resolve())
        parts = relative_path.parts
        if len(parts) >= 2:
            owner = parts[0]
            # Preferred team layout:
            # datasets/<roll>/<roll>_results/data/fused/*.json
            if bundle_root.name.casefold() in {
                f"{bundle_root.parent.name}_results".casefold(),
                "results",
            } or re.fullmatch(r"\d{6,}", owner):
                return owner
        relative = relative_path.as_posix()
        return relative if relative != "." else bundle_root.name
    except ValueError:
        return bundle_root.name


def _contributor_root(bundle_root: Path, data_root: Path) -> Path:
    """Return the roll/member directory that owns a result bundle."""

    collection = _collection_id(bundle_root, data_root)
    candidate = data_root.resolve() / collection
    if candidate.is_dir():
        return candidate
    return bundle_root


def _source_audio_files(contributor_root: Path, bundle_roots: list[Path]) -> list[Path]:
    """Find source audio while excluding generated ``processed/`` WAVs."""

    files: list[Path] = []
    # Everything inside a result bundle is pipeline output, never source audio.
    generated_roots = [root.resolve() for root in bundle_roots]
    for path in contributor_root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in AUDIO_SUFFIXES:
            continue
        resolved = path.resolve()
        if any(resolved.is_relative_to(generated) for generated in generated_roots):
            continue
        files.append(resolved)
    return sorted(set(files))


def _contribution_state(audio_files: int, bundles: int, episodes: int) -> str:
    """Describe a roll folder's stored contribution without overstating it."""

    if episodes:
        return "Searchable" if bundles else "Searchable (no bundle)"
    if bundles:
        return "Bundle present, no fused transcripts"
    if audio_files:
        return "Source audio only, pipeline not run"
    return "Folder empty"


def _label_lookup(bundle_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Load optional labels without treating weak labels as human gold."""

    lookup: dict[str, list[dict[str, Any]]] = {}
    candidates = [
        resolve_stage_dir(bundle_root, "annotated"),
        resolve_stage_dir(bundle_root, "weak_audio"),
        resolve_stage_dir(bundle_root, "weak_labels"),
    ]
    # First source wins: human annotations outrank weak labels.
    for directory in candidates:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            try:
                obj = _load_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            for utterance in obj.get("utterances", []):
                uid = str(utterance.get("utterance_id", ""))
                if uid and uid not in lookup:
                    lookup[uid] = list(utterance.get("labels") or [])
    return lookup


class CorpusIndex:
    """In-memory view over one or more exported pipeline result bundles."""

    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root).expanduser().resolve()
        self.records: list[UtteranceRecord] = []
        self.episodes: dict[str, EpisodeSummary] = {}
        self.warnings: list[str] = []
        self.bundle_roots: list[Path] = []
        self.bundle_by_collection: dict[str, Path] = {}
        self.bundle_roots_by_collection: dict[str, list[Path]] = {}
        self.contributor_roots: dict[str, Path] = {}
        self._semantic: SemanticRetriever | None = None
        self._load()

    def _load(self) -> None:
        if not self.data_root.exists():
            raise FileNotFoundError(f"Corpus root does not exist: {self.data_root}")
        self.bundle_roots = _discover_bundle_roots(self.data_root)
        if not self.bundle_roots:
            raise FileNotFoundError(
                f"No data/fused/*.json outputs found recursively under {self.data_root}"
            )

        for bundle_root in self.bundle_roots:
            collection = _collection_id(bundle_root, self.data_root)
            if collection in self.bundle_by_collection:
                raise ValueError(
                    f"More than one result bundle resolves to roll/member {collection!r}. "
                    "Keep one <roll>_results bundle per roll or use distinct owner folders."
                )
            self.bundle_by_collection[collection] = bundle_root
            self.bundle_roots_by_collection.setdefault(collection, []).append(bundle_root)
            self.contributor_roots[collection] = _contributor_root(bundle_root, self.data_root)
            registry_rows = _read_csv_rows(
                _canonical_registry(resolve_stage_dir(bundle_root, "raw"))
            )
            registry = {row.get("episode_id", ""): row for row in registry_rows}
            labels = _label_lookup(bundle_root)
            fused_dir = resolve_stage_dir(bundle_root, "fused")
            claimed_audio: set[Path] = set()

            for fused_path in sorted(fused_dir.glob("*.json")):
                try:
                    fused = _load_json(fused_path)
                except (OSError, json.JSONDecodeError) as exc:
                    self.warnings.append(f"Skipped {fused_path}: {exc}")
                    continue
                episode_id = str(fused.get("episode_id") or fused_path.stem)
                global_episode = f"{collection}::{episode_id}"
                meta = registry.get(episode_id, {})
                audio = _resolve_audio_path(
                    str(meta.get("local_path", "")),
                    bundle_root=bundle_root,
                    data_root=self.data_root,
                    episode_id=episode_id,
                    duration_sec=_safe_float(meta.get("duration_sec")),
                    claimed=claimed_audio,
                )
                if audio:
                    claimed_audio.add(audio)
                else:
                    self.warnings.append(f"Audio not found for {global_episode}; transcript search still works.")

                utterances = list(fused.get("utterances", []))
                episode_records: list[UtteranceRecord] = []
                for index, utterance in enumerate(utterances):
                    text = str(
                        utterance.get("text_normalized")
                        or utterance.get("text_verbatim")
                        or ""
                    ).strip()
                    words = list(utterance.get("words") or [])
                    probabilities = [
                        _safe_float(word.get("prob"), float("nan"))
                        for word in words
                        if word.get("prob") is not None
                    ]
                    probabilities = [value for value in probabilities if math.isfinite(value)]
                    speaker_id = int(utterance.get("speaker_id", -1))
                    uid = str(utterance.get("utterance_id") or f"{episode_id}_utt{index:05d}")
                    key = f"{global_episode}::{uid}"
                    episode_records.append(
                        UtteranceRecord(
                            key=key,
                            collection_id=collection,
                            episode_id=episode_id,
                            global_episode_id=global_episode,
                            utterance_id=uid,
                            speaker_id=speaker_id,
                            start_sec=_safe_float(utterance.get("start_sec")),
                            end_sec=_safe_float(utterance.get("end_sec")),
                            text=text,
                            text_verbatim=str(utterance.get("text_verbatim") or text),
                            normalized_text=normalize_text(text),
                            programme=str(meta.get("programme", "")),
                            channel=str(meta.get("channel", "")),
                            audio_path=str(audio) if audio else "",
                            diarization_provider=str(fused.get("diarization_provider", "")),
                            asr_model=str(fused.get("asr_model", "")),
                            word_count=len(words) if words else len(text.split()),
                            mean_word_confidence=(
                                sum(probabilities) / len(probabilities) if probabilities else None
                            ),
                            targets=labels.get(uid, []),
                        )
                    )

                for index, record in enumerate(episode_records):
                    record.previous_text = episode_records[index - 1].text if index else ""
                    record.next_text = (
                        episode_records[index + 1].text
                        if index + 1 < len(episode_records)
                        else ""
                    )
                self.records.extend(episode_records)

                stats = fused.get("stats", {})
                word_count = int(stats.get("n_words") or sum(r.word_count for r in episode_records))
                unassigned = int(
                    stats.get("unassigned_words")
                    if stats.get("unassigned_words") is not None
                    else sum(r.word_count for r in episode_records if r.speaker_id < 0)
                )
                duration = _safe_float(meta.get("duration_sec"))
                if duration <= 0:
                    duration = max((r.end_sec for r in episode_records), default=0.0)
                speakers = {record.speaker_id for record in episode_records if record.speaker_id >= 0}
                assignment_rate = (1.0 - unassigned / word_count) if word_count else 0.0
                if word_count and assignment_rate < LOW_ASSIGNMENT_WARNING:
                    # Search hides unassigned words by default, so an episode
                    # whose diarization covered little speech would look almost
                    # empty without saying why.
                    self.warnings.append(
                        f"{global_episode}: diarization attached a speaker to only "
                        f"{assignment_rate * 100:.1f}% of ASR words. Tick "
                        f"'Include unassigned speaker (-1)' to search the remainder."
                    )
                self.episodes[global_episode] = EpisodeSummary(
                    collection_id=collection,
                    episode_id=episode_id,
                    global_episode_id=global_episode,
                    programme=str(meta.get("programme", "")),
                    channel=str(meta.get("channel", "")),
                    duration_sec=duration,
                    speaker_count=len(speakers),
                    utterance_count=len(episode_records),
                    word_count=word_count,
                    unassigned_words=unassigned,
                    assignment_rate=assignment_rate,
                    diarization_provider=str(fused.get("diarization_provider", "")),
                    asr_model=str(fused.get("asr_model", "")),
                    audio_path=str(audio) if audio else "",
                )

        self.records.sort(key=lambda r: (r.collection_id, r.episode_id, r.start_sec, r.end_sec))
        if not self.records:
            raise ValueError(f"Fused JSON files were found under {self.data_root}, but no utterances loaded.")

        # Include roll/member folders that hold source audio, or that are named
        # like a roll number but are still empty, even when no result bundle
        # exists yet. This keeps the group-work inventory honest while results
        # arrive incrementally.
        for child in sorted(self.data_root.iterdir()):
            if not child.is_dir() or child.name in self.contributor_roots:
                continue
            has_audio = any(
                path.is_file() and path.suffix.casefold() in AUDIO_SUFFIXES
                for path in child.rglob("*")
            )
            if has_audio or ROLL_PATTERN.fullmatch(child.name):
                self.contributor_roots[child.name] = child.resolve()

    def stage_dir(self, collection: str, stage: str) -> Path:
        """Stage folder for a roll, tolerant of both exported bundle layouts."""

        bundle_root = self.bundle_by_collection.get(collection)
        if bundle_root is None:
            return self.data_root / collection / "data" / stage
        return resolve_stage_dir(bundle_root, stage)

    @property
    def collections(self) -> list[str]:
        return sorted({record.collection_id for record in self.records})

    @property
    def global_episodes(self) -> list[str]:
        return sorted(self.episodes)

    @property
    def speaker_ids(self) -> list[int]:
        return sorted({record.speaker_id for record in self.records if record.speaker_id >= 0})

    def corpus_summary(self) -> dict[str, Any]:
        words = sum(episode.word_count for episode in self.episodes.values())
        unassigned = sum(episode.unassigned_words for episode in self.episodes.values())
        contributor_rows = self.contributor_rows()
        return {
            "collections": len(self.collections),
            "contributors": len(contributor_rows),
            "result_bundles": len(self.bundle_roots),
            "source_audio_files": sum(row["Source audio files"] for row in contributor_rows),
            "source_audio_bytes": sum(row["Source bytes"] for row in contributor_rows),
            "episodes": len(self.episodes),
            "hours": sum(episode.duration_sec for episode in self.episodes.values()) / 3600,
            "utterances": len(self.records),
            "words": words,
            "assignment_rate": (1.0 - unassigned / words) if words else 0.0,
            "audio_available": sum(bool(episode.audio_path) for episode in self.episodes.values()),
            "labelled_utterances": sum(bool(record.targets) for record in self.records),
        }

    def contributor_rows(self) -> list[dict[str, Any]]:
        """Summarize group totals and each roll/member's stored contribution."""

        rows: list[dict[str, Any]] = []
        for contributor, root in sorted(self.contributor_roots.items()):
            bundles = self.bundle_roots_by_collection.get(contributor, [])
            audio_files = _source_audio_files(root, bundles)
            episodes = [
                episode
                for episode in self.episodes.values()
                if episode.collection_id == contributor
            ]
            records = [
                record for record in self.records if record.collection_id == contributor
            ]
            rows.append(
                {
                    "Roll / member": contributor,
                    "Source audio files": len(audio_files),
                    "Source bytes": sum(path.stat().st_size for path in audio_files),
                    "Source size GB": round(
                        sum(path.stat().st_size for path in audio_files) / 1024**3, 3
                    ),
                    "Result bundles": len(bundles),
                    "Processed episodes": len(episodes),
                    "Processed hours": round(
                        sum(episode.duration_sec for episode in episodes) / 3600, 2
                    ),
                    "Utterances": len(records),
                    "ASR words": sum(episode.word_count for episode in episodes),
                    "Audio linked": sum(bool(episode.audio_path) for episode in episodes),
                    "State": _contribution_state(len(audio_files), len(bundles), len(episodes)),
                    "Folder": str(root.relative_to(self.data_root)),
                    "Bundle": ", ".join(
                        str(bundle.relative_to(self.data_root)) for bundle in bundles
                    ) or "Not available",
                }
            )
        return rows

    def episode_rows(self) -> list[dict[str, Any]]:
        rows = []
        for episode in sorted(self.episodes.values(), key=lambda value: value.global_episode_id):
            rows.append(
                {
                    "Roll": episode.collection_id,
                    "Episode": episode.episode_id,
                    "Programme": episode.programme,
                    "Duration": format_timestamp(episode.duration_sec),
                    "Speakers": episode.speaker_count,
                    "Utterances": episode.utterance_count,
                    "Words": episode.word_count,
                    "Assigned": f"{episode.assignment_rate * 100:.2f}%",
                    "Diarization": episode.diarization_provider or "unknown",
                    "ASR model": Path(episode.asr_model).name if episode.asr_model else "unknown",
                    "Audio": "Available" if episode.audio_path else "Missing",
                }
            )
        return rows

    def speaker_rows(self, global_episode_id: str) -> list[dict[str, Any]]:
        records = [r for r in self.records if r.global_episode_id == global_episode_id]
        rows = []
        for speaker in sorted({record.speaker_id for record in records}):
            selected = [record for record in records if record.speaker_id == speaker]
            seconds = sum(max(0.0, record.end_sec - record.start_sec) for record in selected)
            rows.append(
                {
                    "Speaker": _speaker_label(speaker),
                    "Speaker ID": speaker,
                    "Utterances": len(selected),
                    "Attributed time": format_timestamp(seconds),
                    "Words": sum(record.word_count for record in selected),
                }
            )
        return rows

    def transcript_rows(self, global_episode_id: str) -> list[dict[str, Any]]:
        return [
            {
                "Time": record.timestamp,
                "Speaker": _speaker_label(record.speaker_id),
                "Speaker ID": record.speaker_id,
                "Transcript": record.text,
                "ASR confidence": (
                    round(record.mean_word_confidence, 3)
                    if record.mean_word_confidence is not None
                    else None
                ),
            }
            for record in self.records
            if record.global_episode_id == global_episode_id
        ]

    def _filtered_records(
        self,
        *,
        collection: str = ALL_COLLECTIONS,
        episode: str = ALL_EPISODES,
        speaker: str | int = ALL_SPEAKERS,
        include_unassigned: bool = False,
    ) -> list[UtteranceRecord]:
        records = self.records
        if collection and collection != ALL_COLLECTIONS:
            records = [record for record in records if record.collection_id == collection]
        if episode and episode != ALL_EPISODES:
            records = [record for record in records if record.global_episode_id == episode]
        if speaker not in (None, "", ALL_SPEAKERS):
            speaker_id = int(speaker)
            records = [record for record in records if record.speaker_id == speaker_id]
        if not include_unassigned:
            records = [record for record in records if record.speaker_id >= 0]
        return records

    @staticmethod
    def _lexical_score(query: str, text: str, method: str, query_type: str) -> tuple[float, str]:
        q = normalize_text(query)
        t = normalize_text(text)
        if not q or not t:
            return 0.0, "none"
        q_tokens = q.split()
        t_tokens = t.split()
        token_set = set(t_tokens)
        exact_phrase = q in t
        exact_word = len(q_tokens) == 1 and q_tokens[0] in token_set
        all_words = all(token in token_set for token in q_tokens)
        any_ratio = sum(token in token_set for token in q_tokens) / len(q_tokens)
        fuzzy = SequenceMatcher(None, q, t).ratio()
        try:
            from rapidfuzz import fuzz

            fuzzy = max(fuzzy, fuzz.partial_ratio(q, t) / 100.0)
        except ImportError:
            # Sliding windows keep the stdlib fallback useful for short phrases.
            width = len(q_tokens)
            for size in range(max(1, width - 2), min(len(t_tokens), width + 3) + 1):
                for index in range(len(t_tokens) - size + 1):
                    fuzzy = max(
                        fuzzy,
                        SequenceMatcher(None, q, " ".join(t_tokens[index:index + size])).ratio(),
                    )

        method_key = method.casefold()
        if method_key == "exact":
            return (1.0, "exact") if (exact_word if query_type == "Word" else exact_phrase) else (0.0, "none")
        if method_key == "all words":
            return (0.96 if exact_phrase else 0.88, "all words") if all_words else (0.0, "none")
        if method_key == "any word":
            return (0.65 + 0.30 * any_ratio, "any word") if any_ratio else (0.0, "none")
        if method_key == "fuzzy":
            return fuzzy, "fuzzy"
        # Smart and Hybrid lexical component.
        if exact_word or exact_phrase:
            return 1.0, "exact"
        if all_words:
            return 0.90, "all words"
        if any_ratio:
            return max(0.55 + 0.30 * any_ratio, fuzzy * 0.92), "lexical"
        return fuzzy * 0.92, "fuzzy"

    def build_semantic_index(self, cache_dir: str | Path | None = None) -> str:
        if self._semantic is None:
            self._semantic = SemanticRetriever(self.records, cache_dir=cache_dir)
        self._semantic.ensure_ready()
        return self._semantic.status

    def search(
        self,
        query: str,
        *,
        query_type: str = "Auto",
        method: str = "Smart",
        collection: str = ALL_COLLECTIONS,
        episode: str = ALL_EPISODES,
        speaker: str | int = ALL_SPEAKERS,
        include_unassigned: bool = False,
        minimum_score: float = 0.55,
        top_k: int = 20,
        include_context: bool = True,
        semantic_cache_dir: str | Path | None = None,
    ) -> list[SearchHit]:
        if not normalize_text(query):
            return []
        inferred_type = query_type
        if query_type == "Auto":
            inferred_type = "Word" if len(normalize_text(query).split()) == 1 else "Sentence"
        candidates = self._filtered_records(
            collection=collection,
            episode=episode,
            speaker=speaker,
            include_unassigned=include_unassigned,
        )
        semantic_scores: dict[str, float] = {}
        if method in {"Semantic", "Hybrid"}:
            if self._semantic is None:
                self._semantic = SemanticRetriever(self.records, cache_dir=semantic_cache_dir)
            semantic_scores = self._semantic.query(query)

        hits: list[SearchHit] = []
        for record in candidates:
            lexical, match_type = self._lexical_score(query, record.text, method, inferred_type)
            semantic = semantic_scores.get(record.key)
            if method == "Semantic":
                score = semantic if semantic is not None else 0.0
                match_type = "semantic"
            elif method == "Hybrid":
                score = 0.45 * lexical + 0.55 * (semantic or 0.0)
                match_type = "hybrid"
            else:
                score = lexical
            if score < float(minimum_score):
                continue
            context_parts = []
            if include_context and record.previous_text:
                context_parts.append(f"Previous: {record.previous_text}")
            context_parts.append(f"Match: {record.text}")
            if include_context and record.next_text:
                context_parts.append(f"Next: {record.next_text}")
            targets = ", ".join(
                f"{label.get('target_type', '?')} · {label.get('polarity', '?')}"
                for label in record.targets
            )
            hits.append(
                SearchHit(
                    key=record.key,
                    collection_id=record.collection_id,
                    episode_id=record.episode_id,
                    global_episode_id=record.global_episode_id,
                    utterance_id=record.utterance_id,
                    speaker_id=record.speaker_id,
                    speaker_reference=record.speaker_reference,
                    start_sec=record.start_sec,
                    end_sec=record.end_sec,
                    timestamp=record.timestamp,
                    match_type=match_type,
                    score=float(score),
                    lexical_score=float(lexical),
                    semantic_score=float(semantic) if semantic is not None else None,
                    text=record.text,
                    context="\n\n".join(context_parts),
                    programme=record.programme,
                    mean_word_confidence=record.mean_word_confidence,
                    audio_available=bool(record.audio_path),
                    targets=targets,
                )
            )
        hits.sort(key=lambda hit: (-hit.score, hit.collection_id, hit.episode_id, hit.start_sec))
        return hits[: max(1, int(top_k))]

    def record_by_key(self, key: str) -> UtteranceRecord | None:
        return next((record for record in self.records if record.key == key), None)

    def create_audio_clip(
        self,
        key: str,
        *,
        padding_sec: float = 2.0,
        output_dir: str | Path | None = None,
    ) -> Path | None:
        record = self.record_by_key(key)
        if record is None or not record.audio_path:
            return None
        ffmpeg = _ffmpeg_exe()
        if not ffmpeg:
            return None
        target_dir = Path(output_dir or tempfile.gettempdir()) / "bangla_financial_search_clips"
        target_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        target = target_dir / f"{digest}.wav"
        start = max(0.0, record.start_sec - float(padding_sec))
        duration = max(0.5, record.end_sec - record.start_sec + 2 * float(padding_sec))
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            record.audio_path,
            "-t",
            f"{duration:.3f}",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(target),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        return target if completed.returncode == 0 and target.exists() else None

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(SEMANTIC_MODEL.encode("utf-8"))
        digest.update(SEMANTIC_MODEL_REVISION.encode("utf-8"))
        for record in self.records:
            digest.update(record.key.encode("utf-8"))
            digest.update(record.normalized_text.encode("utf-8"))
        return digest.hexdigest()[:20]


class SemanticRetriever:
    """Lazy multilingual-E5 index with a content-addressed NumPy cache."""

    def __init__(self, records: list[UtteranceRecord], cache_dir: str | Path | None = None):
        self.records = records
        self.cache_dir = Path(
            cache_dir
            or os.environ.get("BFT_SEARCH_CACHE", "")
            or (Path(tempfile.gettempdir()) / "bangla_financial_search")
        ).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.tokenizer = None
        self.vectors = None
        self.status = "Semantic index not loaded"

    def _fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(SEMANTIC_MODEL.encode("utf-8"))
        digest.update(SEMANTIC_MODEL_REVISION.encode("utf-8"))
        for record in self.records:
            digest.update(record.key.encode("utf-8"))
            digest.update(record.normalized_text.encode("utf-8"))
        return digest.hexdigest()[:20]

    @property
    def cache_path(self) -> Path:
        return self.cache_dir / f"multilingual_e5_{self._fingerprint()}.npz"

    def _load_model(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Semantic search requires torch and transformers. Install demo/requirements.txt."
            ) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(
            SEMANTIC_MODEL,
            revision=SEMANTIC_MODEL_REVISION,
        )
        self.model = AutoModel.from_pretrained(
            SEMANTIC_MODEL,
            revision=SEMANTIC_MODEL_REVISION,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def _encode(self, texts: list[str], *, prefix: str, batch_size: int = 32):
        import numpy as np
        import torch
        import torch.nn.functional as functional

        self._load_model()
        output_vectors = []
        for start in range(0, len(texts), batch_size):
            batch = [f"{prefix}: {text}" for text in texts[start:start + batch_size]]
            encoded = self.tokenizer(
                batch,
                max_length=256,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).bool()
                hidden = hidden.masked_fill(~mask, 0.0)
                pooled = hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = functional.normalize(pooled, p=2, dim=1)
            output_vectors.append(pooled.cpu().numpy().astype("float32"))
        return np.concatenate(output_vectors, axis=0)

    def ensure_ready(self) -> None:
        if self.vectors is not None:
            return
        import numpy as np

        if self.cache_path.exists():
            cached = np.load(self.cache_path, allow_pickle=False)
            keys = cached["keys"].tolist()
            expected = [record.key for record in self.records]
            if keys == expected:
                self.vectors = cached["vectors"]
                self.status = f"Semantic index loaded from cache ({len(self.records):,} utterances)"
                return
        self.status = f"Building semantic index for {len(self.records):,} utterances"
        self.vectors = self._encode([record.text for record in self.records], prefix="passage")
        np.savez_compressed(
            self.cache_path,
            keys=np.asarray([record.key for record in self.records]),
            vectors=self.vectors,
        )
        self.status = f"Semantic index ready ({len(self.records):,} utterances)"

    def query(self, query: str) -> dict[str, float]:
        import numpy as np

        self.ensure_ready()
        vector = self._encode([query], prefix="query", batch_size=1)[0]
        scores = np.asarray(self.vectors) @ vector
        # E5 scores are intended for ranking.  Convert the observed corpus range
        # into a stable 0..1 UI score without calling it a probability.
        low, high = float(scores.min()), float(scores.max())
        scaled = (scores - low) / (high - low) if high > low else np.ones_like(scores)
        return {record.key: float(score) for record, score in zip(self.records, scaled)}


def highlight_query(text: str, query: str) -> str:
    """HTML-safe, case-insensitive highlighting of normalized query tokens."""

    escaped = html.escape(text)
    tokens = [token for token in normalize_text(query).split() if len(token) > 1]
    for token in sorted(set(tokens), key=len, reverse=True):
        variants = {token}
        if token == "ব্যাংক":
            variants.add("ব্যাঙ্ক")
        for variant in variants:
            escaped = re.sub(
                re.escape(html.escape(variant)),
                lambda match: f"<mark>{match.group(0)}</mark>",
                escaped,
                flags=re.IGNORECASE,
            )
    return escaped


def search_hits_as_dicts(hits: Iterable[SearchHit]) -> list[dict[str, Any]]:
    return [asdict(hit) for hit in hits]
