import logging
import random

import polars as pl

from src.config.schema import Config

logger = logging.getLogger(__name__)

'''
Select frames — rank all frames by score, pick the top N (label_budget), but with diversity
constraints (cap how many frames come from similar scenes) so you don't just grab 2000
nearly-identical frames.
'''


def select(config: Config):

	label_budget = config.mining.label_budget
	max_per_scene = config.mining.diversity.max_per_scene

	# load scores
	signals_dir = config.data.selections_dir.parent / "signals"
	scores = pl.read_parquet(signals_dir / "scores.parquet")

	# sort by combined score descending
	scores_sorted = scores.sort("combined", descending=True)

	# select top frames with per-video diversity cap
	selected = []
	video_counts = {}

	for row in scores_sorted.iter_rows(named=True):
		video_id = row["frame_id"].split("-")[0]
		if video_counts.get(video_id, 0) >= max_per_scene:
			continue
		selected.append(row["frame_id"])
		video_counts[video_id] = video_counts.get(video_id, 0) + 1
		if len(selected) >= label_budget:
			break

	logger.info("Selected %d mined frames (budget: %d)", len(selected), label_budget)

	# generate random baseline selection with the same per-video cap
	all_frame_ids = scores["frame_id"].to_list()
	random.seed(42)
	random.shuffle(all_frame_ids)

	random_selected = []
	random_video_counts = {}

	for frame_id in all_frame_ids:
		video_id = frame_id.split("-")[0]
		if random_video_counts.get(video_id, 0) >= max_per_scene:
			continue
		random_selected.append(frame_id)
		random_video_counts[video_id] = random_video_counts.get(video_id, 0) + 1
		if len(random_selected) >= label_budget:
			break

	logger.info("Selected %d random baseline frames", len(random_selected))

	# write selection manifests
	selections_dir = config.data.selections_dir
	selections_dir.mkdir(parents=True, exist_ok=True)

	pl.DataFrame({"frame_id": selected}).write_parquet(selections_dir / "mined.parquet")
	pl.DataFrame({"frame_id": random_selected}).write_parquet(selections_dir / "random.parquet")

	logger.info("Wrote selection manifests to %s", selections_dir)


