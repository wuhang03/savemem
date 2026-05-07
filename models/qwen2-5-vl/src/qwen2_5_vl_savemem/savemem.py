"""
SaveMem streaming memory for Qwen2.5-VL (short-term gate variant):
- short-term buffer keeps the latest frames;
- mid-term temporal overlap with Otsu + similarity drop to prune near-duplicate tokens;
- long-term per-frame clustering to keep anchors only;
- query-time recency gate: drops all mid/long-term tokens when query focuses on
  recent content (inspired by SimpleStream's recent-first finding).
"""

import json
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F

from .utils import scan_visual_indices, right_pad_and_stack


class SaveMem:
    _OFFSETS_3X3 = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 0), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )
    _OFFSETS_NEIGHBOR = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    def __init__(
        self,
        vision_start_token_id: int,
        vision_end_token_id: int,
        short_term_frames: int = 8,
        mid_term_frames: int = 256,
        direct_drop_sim_threshold: float = 0.8,
        retrieval_top_k: int | None = None,
        max_memory_tokens: int = 2048,
    ):
        self.short_term_frames = short_term_frames
        self.mid_term_frames = mid_term_frames
        self.temporal_patch_size = 2
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.direct_drop_sim_threshold = float(direct_drop_sim_threshold)
        self.retrieval_top_k = retrieval_top_k
        self.max_memory_tokens = max_memory_tokens
        self.scene_change_multiplier: float = 1.5  # frame is scene-change if mean_dist > multiplier * running_avg
        self.semantic_protect_weight: float = 0.15  # direct_drop threshold uplift for high-semantic tokens
        # Recency gate: default short-only, retrieve history only when
        # short-term ColBERT confidence drops below running average.
        # Set to None to disable gate (always use full retrieval).


        self.recency_gate_drop_ratio: float | None = None  # 0.0 = always short-only; 0.9 = normal gate; None = always full-memory
        self._short_score_ema: float = 0.0   # running average of short-term ColBERT scores
        self._ema_initialized: bool = False
        self._ema_alpha: float = 0.2  # EMA smoothing factor (higher = adapt faster)
        self.drop_vis_path: str | None = None
        # Visualization toggle: when True, retrieval-mode decisions
        # (kept/dropped frames, recency_bias) are printed.
        self.visualize: bool = False
        # Stats toggle: when True, per-frame trajectory + per-entry summary
        # (token counts per tier + GPU memory peak) are recorded.
        # When ``stats_path`` is also set, records are appended to that file as JSONL.
        # ``stats_method`` and ``stats_entry_id`` are tags written into each record
        # for cross-method / cross-entry plotting.
        self.stats: bool = False
        self.stats_path: str | None = None
        self.stats_method: str = "savemem"
        self.stats_entry_id: str | None = None
        # Semantic-score visualization toggle.
        # When True, each call to ``_query_guided_memory_selection`` records
        # per-frame semantic/final scores (Fig A: timeline) and a per-token
        # max-sim grid for one highlight frame (Fig B: token heatmap on the
        # frame). Records are appended to ``semantic_vis_path`` (JSONL).
        # ``semantic_vis_frame_id`` selects which frame's per-token grid to
        # dump; if None, picks the frame with the highest semantic score.
        # NOTE: requires the full-retrieval path (i.e. recency_gate_drop_ratio
        # is None or the gate decides FULL-MEMORY); the SHORT-ONLY gate branch
        # only scores short-term frames, so the dump is skipped there.
        self.semantic_vis: bool = False
        self.semantic_vis_path: str | None = None
        self.semantic_vis_frame_id: int | None = None
        # Stage-2 query temporal-intent probe.
        # False (default): late_interaction → top-k retrieval directly,
        # using raw semantic score (no recency_bias / temporal modulation).
        # True: enable recency_bias + temporal-modulated ranking.
        self.use_probe: bool = False
        # Ablation toggle for the entire Stage-2 query-guided retrieval block
        # (recency gate + ColBERT-style late interaction + top-k drop).
        # True (default): full retrieval pipeline runs.
        # False: skip the whole `_query_guided_memory_selection`; mid/long-term
        # tokens carried over from Stage 1 are preserved as-is.
        self.use_late_interaction: bool = True
        # Gate statistics (accumulated across samples)
        self._gate_stats: dict[str, int] = {
            "short_only": 0,
            "full_memory": 0,
        }
        # Top-k retrieval statistics (accumulated across samples).
        # "executed" = the entry actually ran the top-k drop (effective_top_k < n_cands).
        # The skip_* keys record why an entry bypassed top-k pruning.
        self._topk_stats: dict[str, int] = {
            "executed": 0,
            "skip_no_late_interaction": 0,
            "skip_no_query_tokens": 0,
            "skip_short_only": 0,
            "skip_no_candidates": 0,
            "skip_under_budget": 0,
            "skip_low_variance": 0,
        }
        # Pseudo-question semantic scoring (populated by set_tokenizer)
        self._pseudo_q_ids: torch.Tensor | None = None
        self._pseudo_q_norm: torch.Tensor | None = None
        # Debug: populated by _query_guided_memory_selection after each call
        self._last_retrieval: dict | None = None

    _PSEUDO_QUESTIONS = (
        "What objects are visible in the scene?",
        "How many items or people can be seen?",
        "What actions or events are happening?",
        "What has changed in the scene?",
        "Describe the spatial arrangement of objects.",
    )

    def set_tokenizer(self, tokenizer) -> None:
        """Pre-tokenize pseudo questions for semantic scoring during memory construction."""
        all_ids: list[int] = []
        for q in self._PSEUDO_QUESTIONS:
            all_ids.extend(tokenizer.encode(q, add_special_tokens=False))
        self._pseudo_q_ids = torch.tensor(all_ids, dtype=torch.long)
        self._pseudo_q_norm = None  # recomputed when embed_fn is available

    @staticmethod
    def _neighbor_offsets(device: torch.device, include_center: bool) -> torch.Tensor:
        offsets = SaveMem._OFFSETS_3X3 if include_center else SaveMem._OFFSETS_NEIGHBOR
        return torch.tensor(offsets, dtype=torch.long, device=device)

    @staticmethod
    def _compute_grid_hw(
        visual_indices: torch.Tensor,
        height_ids: torch.Tensor,
        width_ids: torch.Tensor,
    ) -> tuple[int, int] | None:
        if visual_indices.numel() == 0:
            return None
        max_h = int(height_ids[visual_indices].max().item())
        max_w = int(width_ids[visual_indices].max().item())
        return max_h + 1, max_w + 1

    @staticmethod
    def _localize_spatial_ids(
        visual_indices: torch.Tensor,
        height_ids: torch.Tensor,
        width_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if visual_indices.numel() == 0:
            return height_ids, width_ids
        height_ids = height_ids.clone()
        width_ids = width_ids.clone()
        height_ids = height_ids - height_ids[visual_indices].min()
        width_ids = width_ids - width_ids[visual_indices].min()
        return height_ids, width_ids

    @staticmethod
    def _pack_sample(
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        batch_index: int,
        kept_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "hidden": hidden_states[batch_index, kept_indices.to(hidden_states.device)],
            "pos_ids": position_ids[:, batch_index, kept_indices.to(position_ids.device)],
            "pos_emb1": position_embeddings[0][:, batch_index, kept_indices.to(position_embeddings[0].device)],
            "pos_emb2": position_embeddings[1][:, batch_index, kept_indices.to(position_embeddings[1].device)],
        }

    @staticmethod
    def _collect_drop_records(
        batch_index: int,
        frames: List[int],
        frame_to_indices: dict[int, torch.Tensor],
        height_ids: torch.Tensor,
        width_ids: torch.Tensor,
        keep_mask: torch.Tensor,
        grid_hw: tuple[int, int] | None,
    ) -> List[dict]:
        records = []
        for frame_idx, frame_id in enumerate(frames):
            frame_indices = frame_to_indices.get(frame_id)
            if frame_indices is None or frame_indices.numel() == 0:
                continue
            dropped_indices = frame_indices[~keep_mask[frame_indices]]
            if dropped_indices.numel() > 0:
                coords = torch.stack([height_ids[dropped_indices], width_ids[dropped_indices]], dim=1)
                drop_coords = coords.detach().cpu().tolist()
            else:
                drop_coords = []
            records.append(
                {
                    "batch_idx": int(batch_index),
                    "frame_id": int(frame_id),
                    "frame_idx": int(frame_idx),
                    "grid_hw": [int(grid_hw[0]), int(grid_hw[1])] if grid_hw is not None else None,
                    "final_drop": drop_coords,
                }
            )
        return records

    @staticmethod
    def _otsu_threshold(
        values: torch.Tensor,
        nbins: int = 128,
        fallback_to_median: bool = False,
    ) -> float | None:
        if values.numel() == 0:
            return None

        v = values.detach().float().clamp(0.0, 2.0)
        if fallback_to_median and float(v.var().item()) <= 1e-12:
            return float(torch.median(v).item())

        hist = torch.histc(v, bins=nbins, min=0.0, max=2.0)
        total = float(hist.sum().item())
        if total <= 0:
            return float(torch.median(v).item()) if fallback_to_median else None

        p = hist / total
        step = 2.0 / nbins
        centers = (torch.arange(nbins, dtype=torch.float32, device=v.device) + 0.5) * step
        omega = torch.cumsum(p, 0)
        mu_k = torch.cumsum(p * centers, 0)
        mu_total = mu_k[-1]
        denom = (omega * (1.0 - omega)).clamp_min(1e-12)
        sigma_b2 = (mu_total * omega - mu_k) ** 2 / denom
        sigma_b2[omega < 1e-6] = -1
        sigma_b2[(1.0 - omega) < 1e-6] = -1
        threshold_index = int(torch.argmax(sigma_b2).item())
        if float(sigma_b2[threshold_index].item()) <= 0.0:
            return float(torch.median(v).item()) if fallback_to_median else None
        return float((threshold_index + 0.5) * step)

    @staticmethod
    def _frame_grids(
        visual_indices: torch.Tensor,
        time_ids: torch.Tensor,
        height_ids: torch.Tensor,
        width_ids: torch.Tensor,
        grid_hw: tuple[int, int] | None,
    ) -> tuple[List[int], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        frame_ids = torch.unique(time_ids[visual_indices], sorted=True)
        frames = [int(frame_id) for frame_id in frame_ids.tolist()]
        frame_to_indices: dict[int, torch.Tensor] = {}
        frame_to_grid: dict[int, torch.Tensor] = {}
        device = visual_indices.device

        for frame_id in frames:
            frame_indices = visual_indices[time_ids[visual_indices] == frame_id]
            frame_to_indices[frame_id] = frame_indices
            if grid_hw is None:
                continue

            frame_heights = height_ids[frame_indices].to(torch.long)
            frame_widths = width_ids[frame_indices].to(torch.long)
            grid = torch.full(grid_hw, -1, dtype=torch.long, device=device)
            if frame_indices.numel() > 0:
                grid[frame_heights, frame_widths] = frame_indices

            padded_grid = torch.full((grid_hw[0] + 2, grid_hw[1] + 2), -1, dtype=torch.long, device=device)
            padded_grid[1:-1, 1:-1] = grid
            frame_to_grid[frame_id] = padded_grid

        return frames, frame_to_indices, frame_to_grid

    def _max_sim_against_frames(
        self,
        query_indices: torch.Tensor,
        neighbor_frames: List[int],
        height_ids: torch.Tensor,
        width_ids: torch.Tensor,
        frame_grids: dict[int, torch.Tensor],
        hidden_norm: torch.Tensor,
        offsets_3x3: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query_heights = height_ids[query_indices].to(torch.long)
        query_widths = width_ids[query_indices].to(torch.long)
        if query_heights.numel() == 0:
            raise RuntimeError("No neighbors found for window/pair matching")

        grid_heights = query_heights + 1
        grid_widths = query_widths + 1
        neighbor_heights = grid_heights.view(-1, 1) + offsets_3x3[:, 0].view(1, -1)
        neighbor_widths = grid_widths.view(-1, 1) + offsets_3x3[:, 1].view(1, -1)

        neighbor_mats: List[torch.Tensor] = []
        for neighbor_frame in neighbor_frames:
            neighbor_mats.append(frame_grids[neighbor_frame][neighbor_heights, neighbor_widths])
        neighbor_mat = neighbor_mats[0] if len(neighbor_mats) == 1 else torch.cat(neighbor_mats, dim=1)

        valid_neighbors = neighbor_mat >= 0
        has_neighbors = valid_neighbors.any(dim=1)
        if not bool(has_neighbors.any().item()):
            raise RuntimeError("No neighbors found for window/pair matching")

        query_with_neighbors = query_indices[has_neighbors]
        neighbor_mat = neighbor_mat[has_neighbors]
        valid_neighbors = valid_neighbors[has_neighbors]

        safe_indices = neighbor_mat.clamp_min(0)
        neighbor_features = hidden_norm[safe_indices]
        query_features = hidden_norm[query_with_neighbors].unsqueeze(1)
        similarities = (neighbor_features * query_features).sum(dim=2)
        similarities = similarities.masked_fill(~valid_neighbors, float("-inf"))
        max_similarities = similarities.max(dim=1).values
        distances = 1.0 - max_similarities
        return query_with_neighbors, max_similarities, distances, has_neighbors

    def _apply_adjacent_pruning(
        self,
        frame_id: int,
        buffer_frames: List[int],
        frame_to_indices: dict[int, torch.Tensor],
        frame_grids: dict[int, torch.Tensor],
        height_ids: torch.Tensor,
        width_ids: torch.Tensor,
        hidden_norm: torch.Tensor,
        pair_distance_cache: torch.Tensor,
        keep_mask: torch.Tensor,
        pair_distance_threshold: Optional[float],
        offsets_3x3: torch.Tensor,
        is_scene_change: bool = False,
        semantic_scores: Optional[torch.Tensor] = None,
    ) -> None:
        query_indices = frame_to_indices[frame_id]
        adjacent_frames = [buffer_frames[0]] if len(buffer_frames) >= 1 else []
        if len(adjacent_frames) == 0:
            raise RuntimeError("Empty neighbor frame for adjacent-threshold")

        query_with_next, max_sim_next, next_distances, has_next_neighbors = self._max_sim_against_frames(
            query_indices=query_indices,
            neighbor_frames=adjacent_frames,
            height_ids=height_ids,
            width_ids=width_ids,
            frame_grids=frame_grids,
            hidden_norm=hidden_norm,
            offsets_3x3=offsets_3x3,
        )

        if has_next_neighbors.numel() != query_indices.numel():
            raise RuntimeError("Mismatch between query tokens and neighbor mask")
        if not bool(has_next_neighbors.all().item()):
            missing_neighbors = query_indices[~has_next_neighbors]
            if missing_neighbors.numel() > 0:
                keep_mask[missing_neighbors] = True

        if pair_distance_threshold is not None:
            next_threshold = float(max(0.0, min(2.0, float(pair_distance_threshold))))
        else:
            next_threshold = self._otsu_threshold(next_distances, fallback_to_median=True)
            if next_threshold is None:
                raise RuntimeError("Empty set for Otsu thresholding")
        keep_by_next = next_distances >= next_threshold

        prev_distances = pair_distance_cache[query_indices]
        valid_prev = prev_distances >= 0
        prev_threshold = None
        if bool(valid_prev.any().item()):
            if pair_distance_threshold is not None:
                prev_threshold = float(max(0.0, min(2.0, float(pair_distance_threshold))))
            else:
                prev_threshold = self._otsu_threshold(prev_distances[valid_prev], fallback_to_median=True)
                if prev_threshold is None:
                    raise RuntimeError("Empty set for Otsu thresholding")

        if prev_threshold is not None:
            prev_distances_on_next = pair_distance_cache[query_with_next]
            keep_local = keep_by_next | (prev_distances_on_next >= prev_threshold)
        else:
            keep_local = keep_by_next

        # Direct-drop: skip for scene-change frames; semantic-adaptive for others
        if not is_scene_change and query_with_next.numel() > 0:
            if semantic_scores is not None:
                token_sem = semantic_scores[query_with_next]
                adaptive_thresh = self.direct_drop_sim_threshold + token_sem * self.semantic_protect_weight
                drop_high_similarity = max_sim_next > adaptive_thresh
            else:
                drop_high_similarity = max_sim_next > self.direct_drop_sim_threshold
            if bool(drop_high_similarity.any().item()):
                keep_local = keep_local & (~drop_high_similarity)

        kept_indices = query_with_next[keep_local]
        if kept_indices.numel() > 0:
            keep_mask[kept_indices] = True

    @staticmethod
    def _late_interaction_score(
        frame_features: torch.Tensor,
        query_feats: torch.Tensor,
        return_per_token: bool = False,
    ):
        """ColBERT-style: per visual token max-sim over all query tokens, then mean.

        When ``return_per_token`` is True, also returns the per-token max-sim
        tensor (shape (N,)) so callers can build a per-token heatmap.
        """
        sim_matrix = frame_features @ query_feats.T         # (N, Q)
        max_sim_per_token = sim_matrix.max(dim=1).values    # (N,)
        mean_score = float(max_sim_per_token.mean().item())
        if return_per_token:
            return mean_score, max_sim_per_token
        return mean_score

    # ------------------------------------------------------------------
    # Recency gate helper
    # ------------------------------------------------------------------

    def _print_topk_stats(self) -> None:
        s = self._topk_stats
        total = sum(s.values())
        executed = s["executed"]
        pct = (executed / total * 100.0) if total > 0 else 0.0
        print(f"[SaveMem TopK Stats] total={total} | executed={executed} ({pct:.2f}%) | "
              f"skip_no_late_interaction={s['skip_no_late_interaction']} | "
              f"skip_no_query_tokens={s['skip_no_query_tokens']} | "
              f"skip_short_only={s['skip_short_only']} | "
              f"skip_no_candidates={s['skip_no_candidates']} | "
              f"skip_under_budget={s['skip_under_budget']} | "
              f"skip_low_variance={s['skip_low_variance']}")

    @staticmethod
    def _drop_all_non_short(
        frames: List[int],
        short_term_frame_ids: set[int],
        frame_to_indices: dict[int, torch.Tensor],
        keep_mask: torch.Tensor,
    ) -> int:
        """Drop all mid/long-term visual tokens. Returns total tokens dropped."""
        total = 0
        for frame_id in frames:
            if frame_id in short_term_frame_ids:
                continue
            frame_indices = frame_to_indices[frame_id]
            alive = frame_indices[keep_mask[frame_indices]]
            if alive.numel() > 0:
                keep_mask[alive] = False
                total += int(alive.numel())
        return total

    def _record_semantic_vis(
        self,
        frame_semantic: dict[int, float],
        frame_to_indices: dict[int, torch.Tensor],
        keep_mask: torch.Tensor,
        short_term_frame_ids: set[int],
        mid_term_frame_ids: set[int],
        long_term_frame_ids: set[int],
        frame_time_map: dict[int, float],
        t_min: float,
        t_range: float,
        candidate_data: dict,
        kept_non_short_ids: set[int],
        hidden_norm: torch.Tensor,
        query_feats: torch.Tensor,
        height_ids: Optional[torch.Tensor],
        width_ids: Optional[torch.Tensor],
        grid_hw: Optional[tuple[int, int]],
    ) -> None:
        """Stash per-frame and per-token (highlight frame) score data on
        ``self._last_retrieval`` for downstream plotting."""
        if not self.semantic_vis:
            return
        if self._last_retrieval is None:
            self._last_retrieval = {}

        def _tier_of(fid: int) -> str:
            if fid in short_term_frame_ids:
                return "short"
            if fid in mid_term_frame_ids:
                return "mid"
            if fid in long_term_frame_ids:
                return "long"
            return "unknown"

        per_frame: list[dict] = []
        for frame_id in sorted(frame_semantic.keys()):
            sem = frame_semantic[frame_id]
            tier = _tier_of(frame_id)
            t_norm_default = (frame_time_map[frame_id] - t_min) / t_range if t_range > 0 else 0.0
            if frame_id in short_term_frame_ids:
                kept = True
                final, t_weight, t_norm = sem, 1.0, t_norm_default
            elif frame_id in candidate_data:
                final, t_norm, t_weight = candidate_data[frame_id]
                kept = frame_id in kept_non_short_ids
            else:
                final, t_weight, t_norm = sem, 1.0, t_norm_default
                kept = False
            per_frame.append({
                "frame_id": int(frame_id),
                "tier": tier,
                "t_norm": round(float(t_norm), 4),
                "semantic_score": round(float(sem), 6),
                "temporal_weight": round(float(t_weight), 6),
                "final_score": round(float(final), 6),
                "kept": bool(kept),
                "num_tokens": int(frame_to_indices[frame_id].numel()),
            })
        self._last_retrieval["per_frame_scores"] = per_frame

        if (height_ids is None or width_ids is None or grid_hw is None
                or hidden_norm is None or query_feats is None):
            return

        if self.semantic_vis_frame_id is not None and self.semantic_vis_frame_id in frame_semantic:
            highlight_id = int(self.semantic_vis_frame_id)
        else:
            highlight_id = int(max(frame_semantic, key=lambda fid: frame_semantic[fid]))

        frame_indices = frame_to_indices[highlight_id]
        kept_in_frame = frame_indices[keep_mask[frame_indices]]
        if kept_in_frame.numel() == 0:
            return
        _, per_token_max = self._late_interaction_score(
            hidden_norm[kept_in_frame], query_feats, return_per_token=True,
        )
        hs = height_ids[kept_in_frame].detach().cpu().tolist()
        ws = width_ids[kept_in_frame].detach().cpu().tolist()
        scores = per_token_max.detach().cpu().tolist()
        self._last_retrieval["per_token_heatmap"] = {
            "frame_id": highlight_id,
            "tier": _tier_of(highlight_id),
            "grid_hw": [int(grid_hw[0]), int(grid_hw[1])],
            "semantic_score": round(float(frame_semantic[highlight_id]), 6),
            "tokens": [
                {"h": int(h), "w": int(w), "max_sim": round(float(s), 6)}
                for h, w, s in zip(hs, ws, scores)
            ],
        }

    def _query_guided_memory_selection(
        self,
        batch_index: int,
        hidden_states: torch.Tensor,
        hidden_norm: torch.Tensor,
        input_ids: torch.Tensor,
        keep_mask: torch.Tensor,
        short_term_frame_ids: set[int],
        frame_to_indices: dict[int, torch.Tensor],
        frames: List[int],
        time_ids: torch.Tensor,
        retrieval_top_k: int,
        mid_term_frame_ids: Optional[set[int]] = None,
        long_term_frame_ids: Optional[set[int]] = None,
        height_ids: Optional[torch.Tensor] = None,
        width_ids: Optional[torch.Tensor] = None,
        grid_hw: Optional[tuple[int, int]] = None,
    ) -> None:
        """
        Temporal-aware, query-guided memory selection for mid/long-term frames.

        Three-stage process:
        1. **Semantic scoring** – ColBERT-style late-interaction between every
           frame's visual tokens and all query text tokens.
        2. **Adaptive temporal bias** – compare the query's average affinity to
           short-term frames vs. non-short-term frames.  A higher short-term
           affinity implies the query focuses on recent content, which produces
           a positive ``recency_bias`` that up-weights temporally newer frames
           (and vice-versa).
        3. **Temporal-modulated ranking** – multiply each candidate's semantic
           score by ``1 + recency_bias * (t_norm - 0.5)`` where ``t_norm``
           is the frame's normalised temporal position in [0, 1].

        Only the top-K scoring mid/long-term frames survive; short-term frames
        are always preserved in full.
        """
        device = hidden_norm.device

        # ---- 0. Ablation: skip Stage-2 entirely when late interaction is off ----
        # No late interaction means no query-aware scoring is available, so the
        # recency gate and top-k drop both become ill-defined. Carry over Stage 1
        # state as-is.
        if not self.use_late_interaction:
            self._last_retrieval = {
                "status": "skip_no_late_interaction",
                "use_late_interaction": False,
            }
            self._topk_stats["skip_no_late_interaction"] += 1
            print("[SaveMem Retrieval] Skip: use_late_interaction=False (Stage-2 ablated)")
            self._print_topk_stats()
            return

        # ---- 1. Collect query text token features ----
        visual_token_set = set()
        for indices in frame_to_indices.values():
            visual_token_set.update(indices.tolist())

        query_positions: List[int] = []
        sample_ids = input_ids[batch_index]
        for i in range(sample_ids.shape[0]):
            if i not in visual_token_set and keep_mask[i]:
                query_positions.append(i)
        if not query_positions:
            self._topk_stats["skip_no_query_tokens"] += 1
            print("[SaveMem Retrieval] No query text tokens found, skipping retrieval.")
            self._print_topk_stats()
            return

        query_indices_t = torch.tensor(query_positions, dtype=torch.long, device=device)
        query_feats = hidden_norm[query_indices_t]  # (Q, D)

        # ---- 2. Recency gate (EMA-based) ----
        # Default: short-only.  Only score short-term frames against query.
        # If score is at/above running average → short-term is sufficient.
        # If score drops → query may need historical context → full retrieval.
        # This avoids scoring 100+ mid/long-term frames in the common case.

        gate_drop_ratio = self.recency_gate_drop_ratio
        if gate_drop_ratio is not None:
            # Score ONLY short-term frames (cheap: typically 4 frames)
            short_term_scores: List[float] = []
            for frame_id in short_term_frame_ids:
                if frame_id not in frame_to_indices:
                    continue
                frame_indices = frame_to_indices[frame_id]
                kept_in_frame = frame_indices[keep_mask[frame_indices]]
                if kept_in_frame.numel() == 0:
                    continue
                score = self._late_interaction_score(hidden_norm[kept_in_frame], query_feats)
                short_term_scores.append(score)

            if short_term_scores:
                short_score = sum(short_term_scores) / len(short_term_scores)

                # Update EMA
                if not self._ema_initialized:
                    self._short_score_ema = short_score
                    self._ema_initialized = True
                else:
                    self._short_score_ema = (
                        self._ema_alpha * short_score
                        + (1.0 - self._ema_alpha) * self._short_score_ema
                    )

                ema = self._short_score_ema
                threshold = ema * gate_drop_ratio
                need_history = short_score < threshold

                s = self._gate_stats
                if not need_history:
                    # SHORT-ONLY: short-term confidence is normal
                    total_dropped = self._drop_all_non_short(
                        frames, short_term_frame_ids, frame_to_indices, keep_mask)
                    s["short_only"] += 1
                    self._last_retrieval = {
                        "status": "short_only",
                        "short_score": round(short_score, 4),
                        "ema": round(ema, 4),
                        "threshold": round(threshold, 4),
                        "short_term_frame_ids": sorted(short_term_frame_ids),
                        "total_dropped_tokens": total_dropped,
                    }
                    print(f"[SaveMem Gate] SHORT-ONLY: short_score={short_score:.4f} >= "
                          f"threshold={threshold:.4f} (ema={ema:.4f}×{gate_drop_ratio}), "
                          f"dropped {total_dropped} mid/long tokens")
                    print(f"[SaveMem Gate Stats] total={sum(s.values())} | "
                          f"SHORT-ONLY={s['short_only']} | FULL-MEMORY={s['full_memory']}")
                    self._topk_stats["skip_short_only"] += 1
                    self._print_topk_stats()
                    return
                else:
                    s["full_memory"] += 1
                    print(f"[SaveMem Gate] FULL-MEMORY: short_score={short_score:.4f} < "
                          f"threshold={threshold:.4f} (ema={ema:.4f}×{gate_drop_ratio}), "
                          f"retrieving history")
                    print(f"[SaveMem Gate Stats] total={sum(s.values())} | "
                          f"SHORT-ONLY={s['short_only']} | FULL-MEMORY={s['full_memory']}")
                    # fall through to full retrieval below

        # ---- 3. Per-frame temporal position [0, 1] ----
        frame_time_map: dict[int, float] = {}
        for frame_id in frames:
            first_idx = frame_to_indices[frame_id][0]
            frame_time_map[frame_id] = float(time_ids[first_idx].item())

        t_min = min(frame_time_map.values())
        t_max = max(frame_time_map.values())
        t_range = max(t_max - t_min, 1.0)

        # ---- 4. Semantic relevance for ALL frames (full retrieval path) ----
        frame_semantic: dict[int, float] = {}
        short_term_scores_full: List[float] = []
        non_short_scores: List[float] = []

        for frame_id in frames:
            frame_indices = frame_to_indices[frame_id]
            kept_in_frame = frame_indices[keep_mask[frame_indices]]
            if kept_in_frame.numel() == 0:
                continue
            score = self._late_interaction_score(hidden_norm[kept_in_frame], query_feats)
            frame_semantic[frame_id] = score
            if frame_id in short_term_frame_ids:
                short_term_scores_full.append(score)
            else:
                non_short_scores.append(score)

        if not non_short_scores:
            self._topk_stats["skip_no_candidates"] += 1
            self._print_topk_stats()
            return

        # ---- 5. Query temporal-intent probe (optional) ----
        # When ``self.use_probe`` is True, compute an adaptive ``recency_bias``
        # from the short vs non-short semantic gap and apply temporal-weighted
        # modulation to each candidate frame. Otherwise we go straight from
        # late interaction to top-k retrieval using the raw semantic score.
        if self.use_probe:
            short_mean = sum(short_term_scores_full) / len(short_term_scores_full) if short_term_scores_full else 0.0
            non_short_mean = sum(non_short_scores) / len(non_short_scores)
            denom = max((abs(short_mean) + abs(non_short_mean)) / 2.0, 1e-6)
            recency_bias = (short_mean - non_short_mean) / denom
        else:
            short_mean = 0.0
            non_short_mean = 0.0
            recency_bias = 0.0

        # ---- 6. Candidate ranking (temporal-modulated only when probe enabled) ----
        candidate_frames: List[tuple[int, float, float, float, float, torch.Tensor]] = []
        #                       frame_id, final, semantic, t_norm, t_weight, indices
        candidate_data: dict[int, tuple[float, float, float]] = {}
        #                       frame_id -> (final, t_norm, t_weight)  (used by semantic-vis dump)
        for frame_id in frames:
            if frame_id in short_term_frame_ids:
                continue
            if frame_id not in frame_semantic:
                continue
            frame_indices = frame_to_indices[frame_id]
            kept_in_frame = frame_indices[keep_mask[frame_indices]]
            if kept_in_frame.numel() == 0:
                continue

            t_norm = (frame_time_map[frame_id] - t_min) / t_range   # [0, 1]
            if self.use_probe:
                temporal_weight = max(1.0 + recency_bias * (t_norm - 0.5), 0.1)
            else:
                temporal_weight = 1.0
            final_score = frame_semantic[frame_id] * temporal_weight
            candidate_frames.append((
                frame_id, final_score, frame_semantic[frame_id],
                t_norm, temporal_weight, kept_in_frame,
            ))
            candidate_data[frame_id] = (final_score, t_norm, temporal_weight)

        mid_set = mid_term_frame_ids or set()
        long_set = long_term_frame_ids or set()

        # Print candidate frames summary (sorted by final score, show top/bottom 5)
        _sorted_cands = sorted(candidate_frames, key=lambda x: x[1], reverse=True)
        print(f"[SaveMem Retrieval] Candidate frames: {len(candidate_frames)} "
              f"(short-term: {len(short_term_frame_ids)}, total: {len(frames)})")
        _show = 5
        _top = _sorted_cands[:_show]
        _bot = _sorted_cands[-_show:] if len(_sorted_cands) > _show * 2 else _sorted_cands[_show:]
        for fid, final, sem, tn, tw, kept in _top:
            print(f"  f{fid}: semantic={sem:.4f}, t_norm={tn:.3f}, "
                  f"t_weight={tw:.3f}, final={final:.4f}, tokens={kept.numel()}")
        if len(_sorted_cands) > _show * 2:
            print(f"  ... ({len(_sorted_cands) - _show * 2} more) ...")
        for fid, final, sem, tn, tw, kept in _bot:
            print(f"  f{fid}: semantic={sem:.4f}, t_norm={tn:.3f}, "
                  f"t_weight={tw:.3f}, final={final:.4f}, tokens={kept.numel()}")

        if len(candidate_frames) <= retrieval_top_k:
            self._last_retrieval = {
                "status": "skip_under_budget",
                "num_candidates": len(candidate_frames),
                "retrieval_top_k": retrieval_top_k,
            }
            self._record_semantic_vis(
                frame_semantic=frame_semantic,
                frame_to_indices=frame_to_indices,
                keep_mask=keep_mask,
                short_term_frame_ids=short_term_frame_ids,
                mid_term_frame_ids=mid_set,
                long_term_frame_ids=long_set,
                frame_time_map=frame_time_map,
                t_min=t_min,
                t_range=t_range,
                candidate_data=candidate_data,
                kept_non_short_ids=set(candidate_data.keys()),
                hidden_norm=hidden_norm,
                query_feats=query_feats,
                height_ids=height_ids,
                width_ids=width_ids,
                grid_hw=grid_hw,
            )
            self._topk_stats["skip_under_budget"] += 1
            print(f"[SaveMem Retrieval] Skip: {len(candidate_frames)} candidates <= top_k={retrieval_top_k}")
            self._print_topk_stats()
            return  # already within budget

        # ---- 7. Adaptive retrieval: adjust effective top_k based on score variance ----
        # When semantic scores are tightly clustered (low variance), frames are
        # roughly equally relevant — aggressive pruning hurts temporal-coverage
        # tasks (e.g. counting).  When scores are spread out, some frames are
        # clearly irrelevant and safe to drop.
        all_final_scores = torch.tensor([c[1] for c in candidate_frames])
        score_std = float(all_final_scores.std().item()) if all_final_scores.numel() > 1 else 0.0
        score_mean = float(all_final_scores.mean().item())
        cv = score_std / max(abs(score_mean), 1e-8)  # coefficient of variation

        # Adaptive effective_top_k:
        #   cv < 0.05 → scores nearly uniform → keep all (skip retrieval)
        #   cv >= 0.3 → clear separation → use original top_k
        #   in between → interpolate, but never drop more than 10% of candidates
        n_cands = len(candidate_frames)
        cv_low, cv_high = 0.05, 0.3
        if cv <= cv_low:
            effective_top_k = n_cands  # keep all
        elif cv >= cv_high:
            effective_top_k = retrieval_top_k
        else:
            ratio = (cv - cv_low) / (cv_high - cv_low)  # 0→1
            effective_top_k = int(n_cands - ratio * (n_cands - retrieval_top_k))
        # Floor: always keep at least 90% of candidates
        min_keep = max(retrieval_top_k, int(n_cands * 0.9))
        effective_top_k = max(effective_top_k, min(min_keep, n_cands))

        print(f"[SaveMem Retrieval] score_cv={cv:.4f}, effective_top_k={effective_top_k} "
              f"(original={retrieval_top_k}, candidates={n_cands})")

        if effective_top_k >= n_cands:
            self._last_retrieval = {
                "status": "skip_low_variance",
                "score_cv": round(cv, 4),
                "num_candidates": n_cands,
                "retrieval_top_k": retrieval_top_k,
                "effective_top_k": effective_top_k,
            }
            self._record_semantic_vis(
                frame_semantic=frame_semantic,
                frame_to_indices=frame_to_indices,
                keep_mask=keep_mask,
                short_term_frame_ids=short_term_frame_ids,
                mid_term_frame_ids=mid_set,
                long_term_frame_ids=long_set,
                frame_time_map=frame_time_map,
                t_min=t_min,
                t_range=t_range,
                candidate_data=candidate_data,
                kept_non_short_ids=set(candidate_data.keys()),
                hidden_norm=hidden_norm,
                query_feats=query_feats,
                height_ids=height_ids,
                width_ids=width_ids,
                grid_hw=grid_hw,
            )
            self._topk_stats["skip_low_variance"] += 1
            print(f"[SaveMem Retrieval] Low score variance (cv={cv:.4f}), keeping all frames")
            self._print_topk_stats()
            return

        # ---- 8. Keep effective-top-K, drop the rest ----
        candidate_frames.sort(key=lambda x: x[1], reverse=True)
        kept_frames_info = []
        dropped_frames_info = []
        for rank, (fid, final, sem, tn, tw, indices) in enumerate(candidate_frames):
            entry = {
                "frame_id": int(fid),
                "semantic_score": round(sem, 4),
                "t_norm": round(tn, 4),
                "temporal_weight": round(tw, 4),
                "final_score": round(final, 4),
                "num_tokens": int(indices.numel()),
            }
            if rank < effective_top_k:
                kept_frames_info.append(entry)
            else:
                dropped_frames_info.append(entry)
                keep_mask[indices] = False

        # ---- 9. Record debug stats ----
        self._last_retrieval = {
            "short_term_frame_ids": sorted(short_term_frame_ids),
            "use_probe": bool(self.use_probe),
            "short_mean_semantic": round(short_mean, 4),
            "non_short_mean_semantic": round(non_short_mean, 4),
            "recency_bias": round(recency_bias, 4),
            "retrieval_top_k": retrieval_top_k,
            "effective_top_k": effective_top_k,
            "score_cv": round(cv, 4),
            "num_candidates": len(candidate_frames),
            "kept_frames": kept_frames_info,
            "dropped_frames": dropped_frames_info,
            "total_kept_tokens": sum(e["num_tokens"] for e in kept_frames_info),
            "total_dropped_tokens": sum(e["num_tokens"] for e in dropped_frames_info),
        }
        self._record_semantic_vis(
            frame_semantic=frame_semantic,
            frame_to_indices=frame_to_indices,
            keep_mask=keep_mask,
            short_term_frame_ids=short_term_frame_ids,
            mid_term_frame_ids=mid_set,
            long_term_frame_ids=long_set,
            frame_time_map=frame_time_map,
            t_min=t_min,
            t_range=t_range,
            candidate_data=candidate_data,
            kept_non_short_ids={e["frame_id"] for e in kept_frames_info},
            hidden_norm=hidden_norm,
            query_feats=query_feats,
            height_ids=height_ids,
            width_ids=width_ids,
            grid_hw=grid_hw,
        )
        self._topk_stats["executed"] += 1
        self._print_topk_stats()

    # main pipeline
    def process_memory_streaming(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        input_ids: torch.Tensor,
        video_grid_thw: Optional[torch.Tensor] = None,
        pair_distance_threshold: Optional[float] = None,
        embed_fn: Optional[torch.nn.Module] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Streaming memory (prefill only):
        - short-term: keep all tokens from the latest S frames;
        - mid-term: when the buffer overflows, prune the oldest frame via prev/next frame Otsu thresholds on 3x3 neighbors,
          with semantic-adaptive direct-drop;
        - long-term: when mid-term exceeds L, compress via spatial-semantic selection (or cluster merge as fallback).
        """
        if video_grid_thw is None:
            raise ValueError("video_grid_thw is None for streaming memory")

        batch_size, sequence_length, _ = hidden_states.shape
        device = hidden_states.device

        # Pre-compute pseudo-question norms for semantic scoring (once)
        if self._pseudo_q_ids is not None and embed_fn is not None:
            if self._pseudo_q_norm is None or self._pseudo_q_norm.device != device:
                with torch.no_grad():
                    q_embeds = embed_fn(self._pseudo_q_ids.to(device))
                self._pseudo_q_norm = F.normalize(q_embeds.float(), p=2, dim=-1)
        temporal_patch_size = int(getattr(self, "temporal_patch_size", 2))
        short_term_grids = int(self.short_term_frames / temporal_patch_size)
        mid_term_limit = int(self.mid_term_frames / temporal_patch_size)
        if short_term_grids < 1:
            raise ValueError(
                f"short_term_frames ({self.short_term_frames}) too small for temporal_patch_size "
                f"({temporal_patch_size}); got short_grids={short_term_grids}."
            )

        offsets_3x3 = self._neighbor_offsets(device, include_center=True)

        processed_samples = []
        kept_indices_list: List[torch.Tensor] = []

        for batch_index in range(batch_size):
            sample_input_ids = input_ids[batch_index]
            visual_indices = scan_visual_indices(
                sample_input_ids,
                self.vision_start_token_id,
                self.vision_end_token_id,
            )
            if visual_indices.numel() == 0:
                kept_indices = torch.arange(sequence_length, device=device)
                processed_samples.append(
                    self._pack_sample(hidden_states, position_ids, position_embeddings, batch_index, kept_indices)
                )
                kept_indices_list.append(kept_indices)
                continue

            drop_vis_path = self.drop_vis_path

            time_ids = position_ids[0, batch_index]
            height_ids = position_ids[1, batch_index].to(torch.long)
            width_ids = position_ids[2, batch_index].to(torch.long)
            height_ids, width_ids = self._localize_spatial_ids(visual_indices, height_ids, width_ids)
            grid_hw = self._compute_grid_hw(visual_indices, height_ids, width_ids)
            frames, frame_to_indices, frame_grids = self._frame_grids(
                visual_indices=visual_indices,
                time_ids=time_ids,
                height_ids=height_ids,
                width_ids=width_ids,
                grid_hw=grid_hw,
            )
            if len(frames) == 0:
                raise RuntimeError("No visual frames extracted")

            keep_mask = torch.zeros(sequence_length, dtype=torch.bool, device=device)
            non_visual_mask = torch.ones(sequence_length, dtype=torch.bool, device=device)
            non_visual_mask[visual_indices] = False
            keep_mask[non_visual_mask] = True

            hidden_norm = F.normalize(hidden_states[batch_index], p=2, dim=1, eps=1e-8)

            # Compute per-visual-token semantic importance (None if no pseudo-q)
            semantic_scores: Optional[torch.Tensor] = None
            if self._pseudo_q_norm is not None:
                vis_norm = hidden_norm[visual_indices].float()
                sem = (vis_norm @ self._pseudo_q_norm.T).max(dim=-1).values
                semantic_scores = torch.zeros(sequence_length, dtype=torch.float32, device=device)
                semantic_scores[visual_indices] = sem

            pair_distance_cache = torch.full((sequence_length,), -1.0, dtype=hidden_states.dtype, device=device)
            short_term_buffer: List[int] = []
            mid_term_frames: List[int] = []
            long_term_frame_ids: List[int] = []
            # Scene-change detection: running history of per-frame mean distances
            frame_mean_distances: List[float] = []
            frame_is_scene_change: dict[int, bool] = {}

            # ---- Stats: per-frame trajectory + per-entry summary ----
            original_vis_tokens = int(visual_indices.numel())
            if self.stats:
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                stats_records: List[dict] = []
                stats_max_total_kept = 0
                stats_max_peak_mb = 0.0
                stats_max_alloc_mb = 0.0
                stats_counters = {
                    "evict_to_mid": 0,
                    "evict_to_long": 0,
                    "budget_drop_events": 0,
                    "tokens_dropped_by_budget": 0,
                    "retrieval_dropped_tokens": 0,
                }

            for frame_offset, frame_id in enumerate(frames):
                frame_events: List[str] = []
                short_term_buffer.append(frame_id)

                if frame_offset > 0:
                    prev_frame_id = frames[frame_offset - 1]
                    current_indices = frame_to_indices[frame_id]
                    paired_indices, _, paired_distances, _ = self._max_sim_against_frames(
                        query_indices=current_indices,
                        neighbor_frames=[prev_frame_id],
                        height_ids=height_ids,
                        width_ids=width_ids,
                        frame_grids=frame_grids,
                        hidden_norm=hidden_norm,
                        offsets_3x3=offsets_3x3,
                    )
                    if paired_indices.numel() > 0:
                        pair_distance_cache[paired_indices] = paired_distances

                    # Scene-change detection using pair distances with previous frame
                    cur_mean_dist = float(paired_distances.mean().item()) if paired_distances.numel() > 0 else 0.0
                    if frame_mean_distances:
                        running_avg = sum(frame_mean_distances) / len(frame_mean_distances)
                        is_sc = cur_mean_dist > self.scene_change_multiplier * running_avg if running_avg > 1e-6 else False
                    else:
                        is_sc = False
                    frame_mean_distances.append(cur_mean_dist)
                    frame_is_scene_change[frame_id] = is_sc
                else:
                    frame_is_scene_change[frame_id] = False

                if len(short_term_buffer) > max(1, short_term_grids):
                    if self.stats:
                        frame_events.append("evict_to_mid")
                        stats_counters["evict_to_mid"] += 1
                    evicted_frame = short_term_buffer.pop(0)
                    self._apply_adjacent_pruning(
                        frame_id=evicted_frame,
                        buffer_frames=short_term_buffer,
                        frame_to_indices=frame_to_indices,
                        frame_grids=frame_grids,
                        height_ids=height_ids,
                        width_ids=width_ids,
                        hidden_norm=hidden_norm,
                        pair_distance_cache=pair_distance_cache,
                        keep_mask=keep_mask,
                        pair_distance_threshold=pair_distance_threshold,
                        offsets_3x3=offsets_3x3,
                        is_scene_change=frame_is_scene_change.get(evicted_frame, False),
                        semantic_scores=semantic_scores,
                    )

                    mid_term_frames.append(evicted_frame)
                    if len(mid_term_frames) > max(0, mid_term_limit):
                        if self.stats:
                            frame_events.append("evict_to_long")
                            stats_counters["evict_to_long"] += 1
                        oldest_mid_frame = mid_term_frames.pop(0)
                        # Choose long-term strategy: spatial-semantic for large frames,
                        # cluster merge for small frames (better spatial coverage)
                        use_semantic_lt = False
                        if semantic_scores is not None:
                            _lt_indices = frame_to_indices[oldest_mid_frame]
                            _lt_surviving = int(keep_mask[_lt_indices].sum().item())
                            use_semantic_lt = _lt_surviving >= 20

                        if use_semantic_lt:
                            self._long_term_spatial_semantic_selection(
                                long_frames={oldest_mid_frame},
                                visual_indices=visual_indices,
                                time_ids=time_ids,
                                height_ids=height_ids,
                                width_ids=width_ids,
                                keep_mask=keep_mask,
                                semantic_scores=semantic_scores,
                            )
                        else:
                            self._long_term_memory_merge_per_frames(
                                batch_index=batch_index,
                                long_frames={oldest_mid_frame},
                                visual_indices=visual_indices,
                                time_ids=time_ids,
                                height_ids=height_ids,
                                width_ids=width_ids,
                                hidden_states=hidden_states,
                                hidden_norm=hidden_norm,
                                keep_mask=keep_mask,
                                grid_hw=grid_hw,
                            )
                        long_term_frame_ids.append(oldest_mid_frame)

                        # Enforce max memory token budget by dropping lowest-semantic
                        # tokens across all long-term frames (query-agnostic: uses
                        # pseudo-question scores, not the actual user query).
                        total_vis = int(keep_mask[visual_indices].sum().item())
                        if total_vis > self.max_memory_tokens and long_term_frame_ids:
                            overflow = total_vis - self.max_memory_tokens
                            # Collect all surviving long-term token indices
                            lt_indices_list = []
                            for fid in long_term_frame_ids:
                                fi = frame_to_indices[fid]
                                alive = fi[keep_mask[fi]]
                                if alive.numel() > 0:
                                    lt_indices_list.append(alive)
                            if lt_indices_list:
                                all_lt_indices = torch.cat(lt_indices_list)
                                # Score: use semantic_scores if available, else hidden_norm L2 (pre-normalized, use raw norm)
                                if semantic_scores is not None:
                                    scores = semantic_scores[all_lt_indices]
                                else:
                                    scores = hidden_states[batch_index][all_lt_indices].float().norm(dim=-1)
                                # Drop the lowest-scored tokens
                                n_drop = min(overflow, all_lt_indices.numel())
                                _, drop_order = scores.topk(n_drop, largest=False)
                                drop_indices = all_lt_indices[drop_order]
                                keep_mask[drop_indices] = False
                                # Remove long-term frames that lost all tokens
                                long_term_frame_ids = [
                                    fid for fid in long_term_frame_ids
                                    if keep_mask[frame_to_indices[fid]].any()
                                ]
                                if self.stats:
                                    frame_events.append("budget_drop")
                                    stats_counters["budget_drop_events"] += 1
                                    stats_counters["tokens_dropped_by_budget"] += int(n_drop)
                                    print(f"[SaveMem Budget] Dropped {n_drop} lowest-semantic long-term tokens, "
                                          f"total: {total_vis} -> {total_vis - n_drop}")

                # ---- Stats: per-frame trajectory record ----
                if self.stats:
                    # Short-term frames are not yet marked kept in keep_mask
                    # (deferred to after the loop), so count them as all-alive.
                    short_tok = sum(
                        int(frame_to_indices[fid].numel()) for fid in short_term_buffer
                    )
                    mid_tok = sum(
                        int(keep_mask[frame_to_indices[fid]].sum().item()) for fid in mid_term_frames
                    )
                    long_tok = sum(
                        int(keep_mask[frame_to_indices[fid]].sum().item()) for fid in long_term_frame_ids
                    )
                    total_tok = short_tok + mid_tok + long_tok
                    if total_tok > stats_max_total_kept:
                        stats_max_total_kept = total_tok
                    if device.type == "cuda":
                        alloc_mb = int(torch.cuda.memory_allocated(device)) / (1024.0 ** 2)
                        peak_mb = int(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
                    else:
                        alloc_mb = 0.0
                        peak_mb = 0.0
                    if peak_mb > stats_max_peak_mb:
                        stats_max_peak_mb = peak_mb
                    if alloc_mb > stats_max_alloc_mb:
                        stats_max_alloc_mb = alloc_mb
                    stats_records.append({
                        "type": "trajectory",
                        "method": self.stats_method,
                        "entry_id": self.stats_entry_id,
                        "batch_idx": int(batch_index),
                        "frame_offset": int(frame_offset),
                        "frame_id": int(frame_id),
                        "stage": "post_frame",
                        "events": list(frame_events),
                        "tokens": {
                            "short": int(short_tok),
                            "mid": int(mid_tok),
                            "long": int(long_tok),
                            "total_kept": int(total_tok),
                            "original_video": int(original_vis_tokens),
                        },
                        "gpu_mem_mb": {
                            "allocated_mb": round(alloc_mb, 2),
                            "peak_mb": round(peak_mb, 2),
                        },
                    })

            for frame_id in short_term_buffer:
                keep_mask[frame_to_indices[frame_id]] = True

            # Stage 1 snapshot: keep_mask state *after* memory generation
            # (short → mid → long pruning) and *before* the retrieval / recency gate.
            # Persisted as records with ``stage="stage1"``; consumed by visualizers.
            if drop_vis_path:
                with open(drop_vis_path, "a") as f_out:
                    for rec in self._collect_drop_records(
                        batch_index=batch_index,
                        frames=frames,
                        frame_to_indices=frame_to_indices,
                        height_ids=height_ids,
                        width_ids=width_ids,
                        keep_mask=keep_mask,
                        grid_hw=grid_hw,
                    ):
                        rec["stage"] = "stage1"
                        f_out.write(json.dumps(rec) + "\n")

            # --- Query-guided retrieval: prune irrelevant mid/long-term frames ---
            if self.retrieval_top_k is not None and self.retrieval_top_k > 0:
                keep_mask_before_retrieval = keep_mask.clone() if drop_vis_path else None
                tokens_before = int(keep_mask[visual_indices].sum().item())
                self._query_guided_memory_selection(
                    batch_index=batch_index,
                    hidden_states=hidden_states,
                    hidden_norm=hidden_norm,
                    input_ids=input_ids,
                    keep_mask=keep_mask,
                    short_term_frame_ids=set(short_term_buffer),
                    frame_to_indices=frame_to_indices,
                    frames=frames,
                    time_ids=time_ids,
                    retrieval_top_k=self.retrieval_top_k,
                    mid_term_frame_ids=set(mid_term_frames),
                    long_term_frame_ids=set(long_term_frame_ids),
                    height_ids=height_ids,
                    width_ids=width_ids,
                    grid_hw=grid_hw,
                )
                tokens_after = int(keep_mask[visual_indices].sum().item())
                r = self._last_retrieval or {}

                # ---- Stats: post-retrieval trajectory record ----
                if self.stats:
                    stats_counters["retrieval_dropped_tokens"] += max(0, tokens_before - tokens_after)
                    short_tok = sum(
                        int(keep_mask[frame_to_indices[fid]].sum().item()) for fid in short_term_buffer
                    )
                    mid_tok = sum(
                        int(keep_mask[frame_to_indices[fid]].sum().item()) for fid in mid_term_frames
                    )
                    long_tok = sum(
                        int(keep_mask[frame_to_indices[fid]].sum().item()) for fid in long_term_frame_ids
                    )
                    total_tok = short_tok + mid_tok + long_tok
                    if device.type == "cuda":
                        alloc_mb = int(torch.cuda.memory_allocated(device)) / (1024.0 ** 2)
                        peak_mb = int(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
                    else:
                        alloc_mb = 0.0
                        peak_mb = 0.0
                    if peak_mb > stats_max_peak_mb:
                        stats_max_peak_mb = peak_mb
                    if alloc_mb > stats_max_alloc_mb:
                        stats_max_alloc_mb = alloc_mb
                    stats_records.append({
                        "type": "trajectory",
                        "method": self.stats_method,
                        "entry_id": self.stats_entry_id,
                        "batch_idx": int(batch_index),
                        "frame_offset": int(len(frames)),
                        "frame_id": -1,
                        "stage": "post_retrieval",
                        "events": ["retrieval"],
                        "tokens": {
                            "short": int(short_tok),
                            "mid": int(mid_tok),
                            "long": int(long_tok),
                            "total_kept": int(total_tok),
                            "original_video": int(original_vis_tokens),
                        },
                        "gpu_mem_mb": {
                            "allocated_mb": round(alloc_mb, 2),
                            "peak_mb": round(peak_mb, 2),
                        },
                    })

                # Summary: which mode was used?
                gate_status = r.get("status", "unknown")
                is_short_only = gate_status.startswith("short_only")
                if self.visualize:
                    if is_short_only:
                        short_vis_tokens = sum(
                            int(keep_mask[frame_to_indices[fid]].sum().item())
                            for fid in short_term_buffer
                        )
                        print(f"[SaveMem] batch={batch_index} | MODE=SHORT-ONLY | "
                              f"visual tokens: {tokens_before} -> {tokens_after} "
                              f"(short-term only: {short_vis_tokens} tokens from "
                              f"{len(short_term_buffer)} frames) | "
                              f"{r.get('gate_reason', '')}")
                    else:
                        print(f"[SaveMem] batch={batch_index} | MODE=FULL-MEMORY | "
                              f"visual tokens: {tokens_before} -> {tokens_after} "
                              f"(dropped {tokens_before - tokens_after}) | "
                              f"status={gate_status}, "
                              f"top_k={r.get('retrieval_top_k', '?')}, "
                              f"candidates={r.get('num_candidates', '?')}")
                    if r.get("kept_frames"):
                        kept_ids = [f"f{e['frame_id']}({e['final_score']:.3f})" for e in r["kept_frames"]]
                        print(f"  kept   frames: {', '.join(kept_ids)}")
                    if r.get("dropped_frames"):
                        drop_ids = [f"f{e['frame_id']}({e['final_score']:.3f})" for e in r["dropped_frames"]]
                        print(f"  dropped frames: {', '.join(drop_ids)}")
                    if "recency_bias" in r:
                        print(f"  recency_bias={r['recency_bias']:.4f}, "
                              f"short_mean={r.get('short_mean_semantic', 0):.4f}, "
                              f"non_short_mean={r.get('non_short_mean_semantic', 0):.4f}")
                if drop_vis_path and keep_mask_before_retrieval is not None:
                    retrieval_dropped = keep_mask_before_retrieval & (~keep_mask)
                    frame_id_to_idx = {fid: idx for idx, fid in enumerate(frames)}
                    with open(drop_vis_path, "a") as f_out:
                        for frame_id in frames:
                            frame_indices = frame_to_indices[frame_id]
                            dropped_in_frame = frame_indices[retrieval_dropped[frame_indices]]
                            if dropped_in_frame.numel() == 0:
                                continue
                            coords = torch.stack(
                                [height_ids[dropped_in_frame], width_ids[dropped_in_frame]], dim=1
                            ).detach().cpu().tolist()
                            kept_in_frame = frame_indices[keep_mask[frame_indices]]
                            kept_coords = torch.stack(
                                [height_ids[kept_in_frame], width_ids[kept_in_frame]], dim=1
                            ).detach().cpu().tolist() if kept_in_frame.numel() > 0 else []
                            f_out.write(json.dumps({
                                "stage": "retrieval",
                                "batch_idx": int(batch_index),
                                "frame_id": int(frame_id),
                                "frame_idx": frame_id_to_idx[frame_id],
                                "grid_hw": [int(grid_hw[0]), int(grid_hw[1])] if grid_hw is not None else None,
                                "retrieval_dropped": coords,
                                "retrieval_kept": kept_coords,
                                "num_dropped": len(coords),
                                "num_kept": len(kept_coords),
                                **(self._last_retrieval or {}),
                            }) + "\n")

                # ---- Semantic-score visualization dump (Fig A + Fig B data) ----
                if self.semantic_vis and self.semantic_vis_path:
                    r = self._last_retrieval or {}
                    per_frame = r.get("per_frame_scores")
                    per_token = r.get("per_token_heatmap")
                    if per_frame is not None or per_token is not None:
                        record = {
                            "type": "semantic_vis",
                            "method": self.stats_method,
                            "entry_id": self.stats_entry_id,
                            "batch_idx": int(batch_index),
                            "num_frames": int(len(frames)),
                            "retrieval_status": r.get("status", "executed"),
                            "use_probe": bool(self.use_probe),
                            "recency_bias": r.get("recency_bias", 0.0),
                            "per_frame_scores": per_frame or [],
                            "per_token_heatmap": per_token,
                        }
                        with open(self.semantic_vis_path, "a") as f_out:
                            f_out.write(json.dumps(record) + "\n")

            if drop_vis_path:
                with open(drop_vis_path, "a") as f_out:
                    for rec in self._collect_drop_records(
                        batch_index=batch_index,
                        frames=frames,
                        frame_to_indices=frame_to_indices,
                        height_ids=height_ids,
                        width_ids=width_ids,
                        keep_mask=keep_mask,
                        grid_hw=grid_hw,
                    ):
                        f_out.write(json.dumps(rec) + "\n")

            if self.stats:
                final_short = sum(
                    int(keep_mask[frame_to_indices[fid]].sum().item()) for fid in short_term_buffer
                )
                final_mid = sum(
                    int(keep_mask[frame_to_indices[fid]].sum().item()) for fid in mid_term_frames
                )
                final_long = sum(
                    int(keep_mask[frame_to_indices[fid]].sum().item()) for fid in long_term_frame_ids
                )
                final_total = int(keep_mask[visual_indices].sum().item())
                # Final values count toward max trackers (final state may exceed
                # in-loop peak when retrieval is disabled).
                if final_total > stats_max_total_kept:
                    stats_max_total_kept = final_total
                if device.type == "cuda":
                    final_peak_mb = int(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
                    final_alloc_mb = int(torch.cuda.memory_allocated(device)) / (1024.0 ** 2)
                else:
                    final_peak_mb = 0.0
                    final_alloc_mb = 0.0
                if final_peak_mb > stats_max_peak_mb:
                    stats_max_peak_mb = final_peak_mb
                if final_alloc_mb > stats_max_alloc_mb:
                    stats_max_alloc_mb = final_alloc_mb

                summary = {
                    "type": "summary",
                    "method": self.stats_method,
                    "entry_id": self.stats_entry_id,
                    "batch_idx": int(batch_index),
                    "input": {
                        "num_frames": int(len(frames)),
                        "original_video_tokens": int(original_vis_tokens),
                        "grid_hw": [int(grid_hw[0]), int(grid_hw[1])] if grid_hw is not None else None,
                    },
                    "max": {
                        "total_kept_tokens": int(stats_max_total_kept),
                        "gpu_mem_peak_mb": round(stats_max_peak_mb, 2),
                        "gpu_mem_alloc_mb": round(stats_max_alloc_mb, 2),
                    },
                    "final": {
                        "total_kept": int(final_total),
                        "short": int(final_short),
                        "mid": int(final_mid),
                        "long": int(final_long),
                        "compression_ratio": round(
                            final_total / max(1, original_vis_tokens), 4
                        ),
                        "gpu_mem_alloc_mb": round(final_alloc_mb, 2),
                        "gpu_mem_peak_mb": round(final_peak_mb, 2),
                    },
                    "counters": dict(stats_counters),
                }
                stats_records.append(summary)

                if self.stats_path:
                    with open(self.stats_path, "a") as f_out:
                        for rec in stats_records:
                            f_out.write(json.dumps(rec) + "\n")

                print(
                    f"[SaveMem Stats] batch={batch_index} entry={self.stats_entry_id} | "
                    f"frames={len(frames)} original={original_vis_tokens} | "
                    f"final: short={final_short} mid={final_mid} long={final_long} "
                    f"total={final_total} (ratio={final_total / max(1, original_vis_tokens):.3f}) | "
                    f"max_kept={stats_max_total_kept} max_peak={stats_max_peak_mb:.1f}MB"
                )

            kept_indices = keep_mask.nonzero(as_tuple=True)[0]
            processed_samples.append(
                self._pack_sample(hidden_states, position_ids, position_embeddings, batch_index, kept_indices)
            )
            kept_indices_list.append(kept_indices)

        hidden_states_out, position_embeddings_out, position_ids_out, attention_mask_out = right_pad_and_stack(
            processed_samples,
            pos_key1="pos_emb1",
            pos_key2="pos_emb2",
        )
        return hidden_states_out, position_embeddings_out, position_ids_out, attention_mask_out, kept_indices_list

    def _long_term_spatial_semantic_selection(
        self,
        long_frames: set[int],
        visual_indices: torch.Tensor,
        time_ids: torch.Tensor,
        height_ids: torch.Tensor,
        width_ids: torch.Tensor,
        keep_mask: torch.Tensor,
        semantic_scores: torch.Tensor,
    ) -> None:
        """
        Long-term compression via spatial-semantic selection:
        - divide frame tokens into 2x2 spatial quadrants;
        - within each quadrant, keep top tokens by semantic importance;
        - adaptive keep ratio based on frame semantic density (15%-40%, avg ~25%);
        - does NOT modify token embeddings (preserves original representations).
        """
        time_ids_visual = time_ids[visual_indices]
        device = visual_indices.device

        for frame_id in sorted(long_frames):
            frame_visual_indices = visual_indices[time_ids_visual == frame_id]
            if frame_visual_indices.numel() == 0:
                continue

            kept_mask = keep_mask[frame_visual_indices]
            kept_indices = frame_visual_indices[kept_mask]
            num_kept = int(kept_indices.numel())
            if num_kept <= 4:
                continue  # already small enough

            # Adaptive keep ratio based on frame mean semantic score
            scores = semantic_scores[kept_indices]
            frame_mean_sem = float(scores.mean().item())
            # Map semantic score [0, 1] to keep ratio [0.15, 0.40]
            keep_ratio = 0.15 + 0.25 * min(1.0, max(0.0, frame_mean_sem))
            total_keep = max(4, int(num_kept * keep_ratio))
            if total_keep >= num_kept:
                continue

            # 2x2 spatial quadrants
            h = height_ids[kept_indices].float()
            w = width_ids[kept_indices].float()
            h_mid = (h.min() + h.max()) / 2.0
            w_mid = (w.min() + w.max()) / 2.0
            quadrant = ((h > h_mid).long() << 1) | (w > w_mid).long()

            per_region_k = max(1, total_keep // 4)

            keep_local = torch.zeros(num_kept, dtype=torch.bool, device=device)
            for q in range(4):
                region_mask = quadrant == q
                if not bool(region_mask.any().item()):
                    continue
                region_indices = region_mask.nonzero(as_tuple=True)[0]
                k = min(per_region_k, region_indices.numel())
                _, topk_local = scores[region_mask].topk(k)
                keep_local[region_indices[topk_local]] = True

            # Fill remaining budget from unselected tokens by global score
            current_kept = int(keep_local.sum().item())
            remaining = total_keep - current_kept
            if remaining > 0:
                unselected_scores = scores.clone()
                unselected_scores[keep_local] = -1.0
                n_unselected = int((~keep_local).sum().item())
                if n_unselected > 0:
                    _, topk_global = unselected_scores.topk(min(remaining, n_unselected))
                    keep_local[topk_global] = True

            drop_indices = kept_indices[~keep_local]
            if drop_indices.numel() > 0:
                keep_mask[drop_indices] = False

    def _long_term_memory_merge_per_frames(
        self,
        batch_index: int,
        long_frames: set[int],
        visual_indices: torch.Tensor,
        time_ids: torch.Tensor,
        height_ids: torch.Tensor,
        width_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        keep_mask: torch.Tensor,
        hidden_norm: Optional[torch.Tensor] = None,
        grid_hw: tuple[int, int] | None = None,
    ) -> None:
        """
        In-place merge for long-term frames:
        - build intra-frame 3x3 neighbor graph with cosine distances;
        - Otsu threshold on d=1-sim, connect edges with d <= tau;
        - pick one anchor per cluster (closest to mean), replace it by the mean, drop others.
        """
        device = hidden_states.device
        neighbor_offsets = self._neighbor_offsets(device, include_center=False)
        time_ids_visual = time_ids[visual_indices]

        for frame_id in sorted(long_frames):
            frame_visual_indices = visual_indices[time_ids_visual == frame_id]
            if frame_visual_indices.numel() == 0:
                continue

            kept_visual_mask = keep_mask[frame_visual_indices]
            if kept_visual_mask.numel() == 0:
                continue

            kept_frame_indices = frame_visual_indices[kept_visual_mask]
            if kept_frame_indices.numel() == 0:
                continue

            frame_heights = height_ids[kept_frame_indices].to(torch.long)
            frame_widths = width_ids[kept_frame_indices].to(torch.long)
            num_nodes = int(kept_frame_indices.numel())
            if num_nodes <= 1:
                continue

            if grid_hw is None:
                local_grid_hw = self._compute_grid_hw(kept_frame_indices, frame_heights, frame_widths)
                if local_grid_hw is None:
                    continue
            else:
                local_grid_hw = grid_hw

            padded_grid = torch.full(
                (local_grid_hw[0] + 2, local_grid_hw[1] + 2),
                -1,
                dtype=torch.long,
                device=device,
            )
            node_indices = torch.arange(num_nodes, dtype=torch.long, device=device)
            grid_heights = frame_heights + 1
            grid_widths = frame_widths + 1
            padded_grid[grid_heights, grid_widths] = node_indices

            neighbor_heights = grid_heights.view(-1, 1) + neighbor_offsets[:, 0].view(1, -1)
            neighbor_widths = grid_widths.view(-1, 1) + neighbor_offsets[:, 1].view(1, -1)
            neighbor_mat = padded_grid[neighbor_heights, neighbor_widths]
            source_mat = node_indices.unsqueeze(1).expand_as(neighbor_mat)
            upper_triangle = (neighbor_mat >= 0) & (neighbor_mat > source_mat)
            if not bool(upper_triangle.any().item()):
                continue

            edge_i = source_mat[upper_triangle]
            edge_j = neighbor_mat[upper_triangle]

            if hidden_norm is not None and hidden_norm.dtype == torch.float32:
                features_norm = hidden_norm[kept_frame_indices]
            else:
                features_norm = F.normalize(hidden_states[batch_index, kept_frame_indices].float(), p=2, dim=1, eps=1e-8)

            edge_distances = 1.0 - (features_norm[edge_i] * features_norm[edge_j]).sum(dim=1)
            edge_distances = edge_distances.clamp(0.0, 2.0)
            threshold = self._otsu_threshold(edge_distances)
            if threshold is None:
                continue

            keep_edges = edge_distances <= threshold
            if not bool(keep_edges.any().item()):
                continue

            edge_i = edge_i[keep_edges]
            edge_j = edge_j[keep_edges]

            labels = torch.arange(num_nodes, device=device, dtype=torch.long)
            for _ in range(max(1, num_nodes)):
                min_labels = torch.minimum(labels[edge_i], labels[edge_j])
                new_labels = labels.clone()
                new_labels.scatter_reduce_(0, edge_i, min_labels, reduce="amin", include_self=True)
                new_labels.scatter_reduce_(0, edge_j, min_labels, reduce="amin", include_self=True)
                new_labels = new_labels[new_labels]
                if torch.equal(new_labels, labels):
                    break
                labels = new_labels

            _, inverse = torch.unique(labels, return_inverse=True)
            num_clusters = int(inverse.max().item()) + 1 if inverse.numel() > 0 else 0
            if num_clusters <= 0:
                continue

            features = hidden_states[batch_index, kept_frame_indices].float()
            cluster_sum = torch.zeros((num_clusters, features.size(1)), dtype=features.dtype, device=device)
            cluster_sum.index_add_(0, inverse, features)
            counts = torch.zeros(num_clusters, dtype=features.dtype, device=device)
            counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=features.dtype))
            counts = counts.clamp_min(1.0)
            cluster_mean = cluster_sum / counts.unsqueeze(1)

            squared_distances = (features - cluster_mean[inverse]).pow(2).sum(dim=1)
            min_squared_distances = torch.full((num_clusters,), float("inf"), dtype=squared_distances.dtype, device=device)
            min_squared_distances.scatter_reduce_(0, inverse, squared_distances, reduce="amin", include_self=True)
            is_cluster_min = squared_distances == min_squared_distances[inverse]
            candidate_indices = torch.nonzero(is_cluster_min, as_tuple=True)[0]
            candidate_clusters = inverse[candidate_indices]
            anchor_local = torch.full((num_clusters,), num_nodes, dtype=torch.long, device=device)
            anchor_local.scatter_reduce_(0, candidate_clusters, candidate_indices, reduce="amin", include_self=True)

            merge_clusters = counts > 1.0
            if not bool(merge_clusters.any().item()):
                continue

            anchor_local = anchor_local[merge_clusters]
            anchor_global = kept_frame_indices[anchor_local]
            hidden_states[batch_index, anchor_global] = cluster_mean[merge_clusters].to(hidden_states.dtype)

            local_keep = torch.ones(num_nodes, dtype=torch.bool, device=device)
            local_keep[merge_clusters[inverse]] = False
            local_keep[anchor_local] = True
            dropped_indices = kept_frame_indices[~local_keep]
            keep_mask[dropped_indices] = False