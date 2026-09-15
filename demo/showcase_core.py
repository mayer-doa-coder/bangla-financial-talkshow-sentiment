"""Evidence-backed project showcase summaries for the Gradio demo."""

from __future__ import annotations

import csv
import json
import tempfile
import zipfile
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from search_core import AUDIO_SUFFIXES, CorpusIndex, format_timestamp


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def _best_json(directory: Path, stem: str) -> tuple[Path | None, Any]:
    candidates = list(directory.glob(f"{stem}*.json")) if directory.is_dir() else []
    if not candidates:
        return None, None
    # Colab/Drive creates '(1)' copies. Prefer the richest file, then canonical
    # filename when copies contain the same payload.
    path = max(candidates, key=lambda p: (p.stat().st_size, p.name == f"{stem}.json"))
    return path, _json(path)


class ProjectShowcase:
    def __init__(self, index: CorpusIndex):
        self.index = index

    @lru_cache(maxsize=1)
    def stage_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        stage = self.index.stage_dir
        for global_id, episode in sorted(self.index.episodes.items()):
            ep = episode.episode_id
            diar = _json(stage(episode.collection_id, "diarization") / f"{ep}.json") or {}
            asr = _json(stage(episode.collection_id, "asr") / f"{ep}.json") or {}
            fused = _json(stage(episode.collection_id, "fused") / f"{ep}.json") or {}
            draft = _json(stage(episode.collection_id, "annotation_drafts") / f"{ep}.json") or {}
            diar_segments = diar.get("segments") or []
            asr_segments = asr.get("segments") or []
            stats = fused.get("stats") or {}
            draft_utterances = draft.get("utterances") or []
            verified = sum(bool(item.get("human_verified")) for item in draft_utterances)
            labels = sum(len(item.get("labels") or []) for item in draft_utterances)
            chunks_done = int(asr.get("n_chunks_completed") or 0)
            chunks_total = int(asr.get("n_chunks_total") or 0)
            rows.append(
                {
                    "Roll": episode.collection_id,
                    "Episode": ep,
                    "Duration": format_timestamp(episode.duration_sec),
                    "Audio": "Ready" if episode.audio_path else "Missing",
                    "Diarization": "Ready" if diar_segments else "Missing",
                    "Diar turns": len(diar_segments),
                    "Speakers": len({s.get("speaker_id") for s in diar_segments if int(s.get("speaker_id", -1)) >= 0}),
                    "ASR": "Ready" if asr_segments else "Missing",
                    "ASR segments": len(asr_segments),
                    "Chunk counter": f"{chunks_done}/{chunks_total}" if chunks_total else "n/a",
                    "Fusion": "Ready" if fused.get("utterances") else "Missing",
                    "Words": int(stats.get("n_words") or episode.word_count),
                    "Utterances": int(stats.get("n_utterances") or episode.utterance_count),
                    "Unassigned": int(stats.get("unassigned_words") or 0),
                    "Assigned": f"{episode.assignment_rate * 100:.2f}%",
                    "Draft rows": len(draft_utterances),
                    "Human verified": verified,
                    "Target labels": labels,
                }
            )
        return rows

    @lru_cache(maxsize=1)
    def audio_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for collection in self.index.bundle_by_collection:
            raw_dir = self.index.stage_dir(collection, "raw")
            registry_path = raw_dir / "registry.csv"
            candidates = sorted(raw_dir.glob("registry*.csv"))
            registry = _csv(registry_path if registry_path.exists() else candidates[0]) if candidates else []
            for item in registry:
                rows.append(
                    {
                        "Roll": collection,
                        "Episode": item.get("episode_id", ""),
                        "Duration": format_timestamp(float(item.get("duration_sec") or 0)),
                        "Sample rate": item.get("sample_rate", ""),
                        "Channels": item.get("channels", ""),
                        "Codec": item.get("codec", ""),
                        "Bitrate kbps": item.get("bit_rate_kbps", ""),
                        "Size MB": round(float(item.get("file_size_bytes") or 0) / 1024**2, 1),
                        "Duration flag": item.get("duration_flag", ""),
                        "SHA-256": (item.get("sha256") or "")[:16] + "…",
                        "Licence attestation": item.get("license_attestation", ""),
                    }
                )
        return rows

    @lru_cache(maxsize=1)
    def diarization_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        stage = self.index.stage_dir
        for global_id, episode in sorted(self.index.episodes.items()):
            obj = _json(stage(episode.collection_id, "diarization") / f"{episode.episode_id}.json") or {}
            segments = obj.get("segments") or []
            params = obj.get("parameters") or {}
            rows.append(
                {
                    "Roll": episode.collection_id,
                    "Episode": episode.episode_id,
                    "Provider": obj.get("provider") or episode.diarization_provider or "unknown",
                    "Speakers": len({s.get("speaker_id") for s in segments if int(s.get("speaker_id", -1)) >= 0}),
                    "Turns": len(segments),
                    "Attributed duration": format_timestamp(sum(max(0.0, float(s.get("end_sec", 0)) - float(s.get("start_sec", 0))) for s in segments)),
                    "θ merge": params.get("theta_merge"),
                    "θ gap": params.get("theta_gap"),
                    "θ segmentation": params.get("theta_seg"),
                    "θ speaker": params.get("theta_spk"),
                    "RTTM": "Available" if (stage(episode.collection_id, "diarization") / f"{episode.episode_id}.rttm").exists() else "Missing",
                }
            )
        return rows

    @lru_cache(maxsize=1)
    def asr_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        stage = self.index.stage_dir
        for global_id, episode in sorted(self.index.episodes.items()):
            obj = _json(stage(episode.collection_id, "asr") / f"{episode.episode_id}.json") or {}
            segments = obj.get("segments") or []
            rows.append(
                {
                    "Roll": episode.collection_id,
                    "Episode": episode.episode_id,
                    "Status": "Durable output ready" if segments else "Missing",
                    "Segments": len(segments),
                    "Chunk metadata": f"{int(obj.get('n_chunks_completed') or 0)}/{int(obj.get('n_chunks_total') or 0)}",
                    "Language": obj.get("language", ""),
                    "Model": Path(str(obj.get("model") or episode.asr_model)).name,
                    "Device": obj.get("device", ""),
                    "Compute": obj.get("compute_type", ""),
                    "Chunk sec": obj.get("chunk_sec"),
                    "Beam": obj.get("beam_size"),
                    "Repetition penalty": obj.get("repetition_penalty"),
                    "Previous-text conditioning": obj.get("condition_on_previous_text"),
                    "Demucs": obj.get("demucs_mode", ""),
                    "Warnings/log": "Recorded" if obj.get("warnings") else "None",
                }
            )
        return rows

    @lru_cache(maxsize=1)
    def fusion_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        stage = self.index.stage_dir
        for global_id, episode in sorted(self.index.episodes.items()):
            obj = _json(stage(episode.collection_id, "fused") / f"{episode.episode_id}.json") or {}
            stats = obj.get("stats") or {}
            words = int(stats.get("n_words") or episode.word_count)
            unassigned = int(stats.get("unassigned_words") or 0)
            rows.append(
                {
                    "Roll": episode.collection_id,
                    "Episode": episode.episode_id,
                    "Words": words,
                    "Utterances": int(stats.get("n_utterances") or episode.utterance_count),
                    "Unassigned words": unassigned,
                    "Assignment coverage": f"{(1 - unassigned / words) * 100:.2f}%" if words else "0%",
                    "Coarse tokens": int(stats.get("coarse_tokens") or 0),
                    "Overlap policy": obj.get("overlap_policy", ""),
                    "Assignment algorithm": obj.get("assignment", ""),
                    "Diarization provider": obj.get("diarization_provider", ""),
                    "Schema": obj.get("schema_version", ""),
                }
            )
        return rows

    @lru_cache(maxsize=1)
    def annotation_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        stage = self.index.stage_dir
        for global_id, episode in sorted(self.index.episodes.items()):
            obj = _json(stage(episode.collection_id, "annotation_drafts") / f"{episode.episode_id}.json") or {}
            utterances = obj.get("utterances") or []
            statuses = Counter(str(u.get("annotation_status") or "missing") for u in utterances)
            rows.append(
                {
                    "Roll": episode.collection_id,
                    "Episode": episode.episode_id,
                    "Draft utterances": len(utterances),
                    "Review required": statuses.get("review_required", 0),
                    "Human verified": sum(bool(u.get("human_verified")) for u in utterances),
                    "Role resolved": sum(bool(u.get("role_tag")) for u in utterances),
                    "Target labels": sum(len(u.get("labels") or []) for u in utterances),
                    "Gold tier": bool(obj.get("is_gold_tier")),
                    "Transcript source": next((u.get("transcript_source") for u in utterances if u.get("transcript_source")), ""),
                    "Speaker source": next((u.get("speaker_source") for u in utterances if u.get("speaker_source")), ""),
                }
            )
        return rows

    @lru_cache(maxsize=1)
    def stage_summary_rows(self) -> list[dict[str, Any]]:
        episodes = len(self.index.episodes)
        stage = self.stage_rows()
        complete = lambda key, value="Ready": sum(row[key] == value for row in stage)
        drafts = sum(row["Draft rows"] for row in stage)
        verified = sum(row["Human verified"] for row in stage)
        labels = sum(row["Target labels"] for row in stage)
        profile_rows = self.profile_rows()
        gate_rows = self.gate_rows()
        error_files = 0
        error_episodes = 0
        for collection in self.index.bundle_by_collection:
            path, obj = _best_json(
                self.index.stage_dir(collection, "aggregated"), "factorial_error_propagation"
            )
            error_files += int(path is not None)
            if isinstance(obj, dict):
                error_episodes += len(obj.get("episodes") or [])
        return [
            {"Stage": "0 · Audio inventory", "State": f"{complete('Audio')}/{episodes} ready", "Evidence": "registry.csv + source audio", "Interpretation": "Completed inputs"},
            {"Stage": "1 · Speaker diarization", "State": f"{complete('Diarization')}/{episodes} ready", "Evidence": "JSON + RTTM speaker turns", "Interpretation": "Automatic episode-local speakers"},
            {"Stage": "2 · Long-form Bangla ASR", "State": f"{complete('ASR')}/{episodes} ready", "Evidence": "timestamped segments and words", "Interpretation": "Automatic transcripts"},
            {"Stage": "3 · Timestamp fusion", "State": f"{complete('Fusion')}/{episodes} ready", "Evidence": "speaker-attributed words/utterances", "Interpretation": "Completed searchable evidence"},
            {"Stage": "4 · Annotation drafts", "State": f"{drafts:,} rows", "Evidence": "review-ready draft JSON", "Interpretation": f"{verified:,} human verified"},
            {"Stage": "5 · Target/sentiment", "State": f"{labels:,} current labels", "Evidence": f"{len(profile_rows):,} exploratory profile rows", "Interpretation": "Not final without aligned verified labels"},
            {"Stage": "6 · Aggregation", "State": "Exploratory" if profile_rows else "Unavailable", "Evidence": "per-speaker/target profile snapshots", "Interpretation": "Weak-labelled snapshots kept separate"},
            {"Stage": "7 · Evaluation", "State": f"{len(gate_rows):,} gates recorded", "Evidence": "PASS/FAIL/SKIPPED gate table", "Interpretation": "Gold-dependent metrics remain unavailable"},
            {"Stage": "8 · Error propagation", "State": f"{error_episodes} evaluated episodes", "Evidence": f"{error_files} result file(s)", "Interpretation": "Requires gold transcripts, RTTM and labels"},
            {"Stage": "9 · Retrieval demo", "State": "Ready", "Evidence": "exact/fuzzy/semantic search + audio", "Interpretation": "Interactive project output"},
        ]

    @lru_cache(maxsize=1)
    def model_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for collection, root in self.index.bundle_by_collection.items():
            manifest = _json(root / "models" / "manifest.json") or []
            for model in manifest if isinstance(manifest, list) else []:
                rows.append(
                    {
                        "Roll": collection,
                        "Stage": "Diarization",
                        "Model": model.get("name", ""),
                        "Hub/source": model.get("hf_uri", ""),
                        "Revision": model.get("commit_hash") or "not recorded",
                        "Notes": model.get("notes", ""),
                    }
                )
        seen_asr: set[tuple[str, str]] = set()
        for episode in self.index.episodes.values():
            key = (episode.collection_id, episode.asr_model)
            if episode.asr_model and key not in seen_asr:
                seen_asr.add(key)
                rows.append(
                    {
                        "Roll": episode.collection_id,
                        "Stage": "ASR",
                        "Model": Path(episode.asr_model).name,
                        "Hub/source": "bengaliAI/tugstugi_bengaliai-asr_whisper-medium derivative",
                        "Revision": "runtime conversion",
                        "Notes": "CTranslate2 long-form inference; exact checkpoint path retained in fused JSON.",
                    }
                )
        rows.append(
            {
                "Roll": "group demo",
                "Stage": "Retrieval",
                "Model": "intfloat/multilingual-e5-small",
                "Hub/source": "Hugging Face Hub",
                "Revision": "c007d7ef6fd86656326059b28395a7a03a7c5846",
                "Notes": "Lazy semantic index; lexical search requires no model download.",
            }
        )
        return rows

    @lru_cache(maxsize=1)
    def gate_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for collection in self.index.bundle_by_collection:
            current_unassigned = sum(
                ep.unassigned_words
                for ep in self.index.episodes.values()
                if ep.collection_id == collection
            )
            path, report = _best_json(
                self.index.stage_dir(collection, "aggregated"), "submission_gates"
            )
            if not isinstance(report, dict):
                continue
            for gate in report.get("gates") or []:
                consistency = "Snapshot"
                if gate.get("gate") == "fusion_no_unassigned_words" and gate.get("value") != current_unassigned:
                    consistency = f"Stale vs current ({current_unassigned} unassigned)"
                rows.append(
                    {
                        "Roll": collection,
                        "Gate": gate.get("gate", ""),
                        "Status": gate.get("status", ""),
                        "Value": gate.get("value"),
                        "Threshold": gate.get("threshold", ""),
                        "Reason / next evidence": gate.get("reason", ""),
                        "Snapshot check": consistency,
                        "Source": path.name if path else "",
                    }
                )
        return rows

    @lru_cache(maxsize=1)
    def profile_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for collection in self.index.bundle_by_collection:
            path, report = _best_json(
                self.index.stage_dir(collection, "aggregated"), "discourse_profiles_v2"
            )
            if not isinstance(report, dict):
                continue
            for episode in report.get("episodes") or []:
                ep_id = str(episode.get("episode_id", ""))
                current = self.index.episodes.get(f"{collection}::{ep_id}")
                snapshot_n = int(episode.get("n_utterances") or 0)
                alignment = "Aligned"
                if current and snapshot_n != current.utterance_count:
                    alignment = f"Legacy/mismatched ({snapshot_n} vs {current.utterance_count} current)"
                for speaker in episode.get("speakers") or []:
                    rows.append(
                        {
                            "Roll": collection,
                            "Episode": ep_id,
                            "Speaker ID": speaker.get("speaker_id"),
                            "Role": speaker.get("role_tag", ""),
                            "Label tier": episode.get("label_tier", ""),
                            "Labelled targets": speaker.get("n_labelled_targets", 0),
                            "Positive": speaker.get("positive", 0),
                            "Negative": speaker.get("negative", 0),
                            "Positive rate": round(float(speaker.get("positive_rate") or 0), 3),
                            "Target counts": json.dumps(speaker.get("target_counts") or {}, ensure_ascii=False),
                            "Alignment": alignment,
                            "Source": path.name if path else "",
                        }
                    )
        return rows

    @lru_cache(maxsize=1)
    def target_policy_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for collection in self.index.bundle_by_collection:
            _, policy = _best_json(
                self.index.stage_dir(collection, "raw"), "target_annotation_policy"
            )
            if not isinstance(policy, dict):
                continue
            for target, description in (policy.get("target_types") or {}).items():
                rows.append({"Roll": collection, "Target type": target, "Definition": description})
        return rows

    @lru_cache(maxsize=1)
    def entity_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for collection in self.index.bundle_by_collection:
            candidates = sorted(
                self.index.stage_dir(collection, "raw").glob("financial_entities*.csv")
            )
            if not candidates:
                continue
            # Identical Drive copies are common; read one canonical/best candidate.
            path = next((p for p in candidates if p.name == "financial_entities.csv"), candidates[0])
            for item in _csv(path):
                rows.append({"Roll": collection, **item})
        return rows

    @lru_cache(maxsize=1)
    def artifact_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        categories = Counter()
        for collection, root in self.index.bundle_by_collection.items():
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(root).as_posix()
                category = relative.split("/", 1)[0]
                if relative.startswith("data/"):
                    parts = relative.split("/")
                    category = "/".join(parts[:2]) if len(parts) > 1 else "data"
                categories[(collection, category)] += 1
                rows.append(
                    {
                        "Roll": collection,
                        "Category": category,
                        "File": relative,
                        "Type": path.suffix.lower().lstrip(".") or "file",
                        "Size KB": round(path.stat().st_size / 1024, 1),
                    }
                )
        return rows

    @lru_cache(maxsize=1)
    def artifact_summary_rows(self) -> list[dict[str, Any]]:
        rows = self.artifact_rows()
        grouped: dict[tuple[str, str], dict[str, float]] = {}
        for row in rows:
            key = (row["Roll"], row["Category"])
            item = grouped.setdefault(key, {"files": 0, "kb": 0.0})
            item["files"] += 1
            item["kb"] += float(row["Size KB"])
        return [
            {
                "Roll": collection,
                "Category": category,
                "Files": int(value["files"]),
                "Size MB": round(value["kb"] / 1024, 2),
            }
            for (collection, category), value in sorted(grouped.items())
        ]

    def waveform_path(self, global_episode_id: str) -> str | None:
        episode = self.index.episodes.get(global_episode_id)
        if not episode:
            return None
        root = self.index.bundle_by_collection[episode.collection_id]
        path = root / "reports" / "figures" / f"waveform_{episode.episode_id}.png"
        return str(path) if path.is_file() else None

    def create_evidence_zip(self, output_dir: str | Path | None = None) -> str:
        directory = Path(output_dir or tempfile.gettempdir()) / "bangla_financial_demo_exports"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "bangla_financial_project_evidence.zip"
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for collection, root in self.index.bundle_by_collection.items():
                safe_collection = collection.replace("/", "__")
                for path in sorted(root.rglob("*")):
                    if not path.is_file():
                        continue
                    # Ship every durable artifact regardless of which bundle
                    # layout a member exported; skip regenerable audio so the
                    # download stays small.
                    if path.suffix.casefold() in AUDIO_SUFFIXES:
                        continue
                    archive.write(path, Path(safe_collection) / path.relative_to(root))
        return str(target)
