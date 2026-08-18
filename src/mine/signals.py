import logging

import polars as pl

from src.config.schema import Config

logger = logging.getLogger(__name__)

'''

1. Compute scores — for each frame, calculate two signals:
	- Uncertainty: how confused is the model? Frames with lots of detections in the 0.3-0.7 confidence range are frames where the model isn't
	sure. Those are valuable to label.
	- Rare-class presence: does the model think it sees a motorcycle, bicycle, or train? Those classes are underrepresented, so any frame
	containing them is worth labeling.
2. Combine scores — normalize both signals to [0,1], take a weighted sum. Each frame gets a single "value" score.


'''




def compute_scores(config: Config):

	# load all frame predictions into a single dataframe
	predictions_dir = config.data.predictions_dir
	df = pl.read_parquet(predictions_dir / "shard_*.parquet")

	# count how often each class appears                                                                                                           
	class_counts = df.group_by("class_id").len()

	# join back and compute inverse frequency                                                                                                    
	df = df.join(class_counts, on="class_id")

	# uncertainty: count detections in the ambiguous confidence band per frame
	uncertainty = (
		df.filter(
			(pl.col("confidence") >= 0.3) & (pl.col("confidence") <= 0.7)
		)
		.group_by("frame_id")
		.len()
		.rename({"len": "uncertainty"})
	)

	# rare-class presence: sum of inverse class frequency per frame
	rare_class = (
		df.filter(pl.col("class_id") >= 0)  # exclude sentinel rows
		.with_columns((1.0 / pl.col("len")).alias("inv_freq"))
		.group_by("frame_id")
		.agg(pl.col("inv_freq").sum().alias("rare_class"))
	)

	# get all unique frame IDs and join both signals
	all_frames = df.select("frame_id").unique()
	scores = (
		all_frames
		.join(uncertainty, on="frame_id", how="left")
		.join(rare_class, on="frame_id", how="left")
		.fill_null(0)
	)

	# normalize both signals to [0, 1] via percentile rank
	scores = scores.with_columns(
		(pl.col("uncertainty").rank() / pl.col("uncertainty").len()).alias("uncertainty_norm"),
		(pl.col("rare_class").rank() / pl.col("rare_class").len()).alias("rare_class_norm"),
	)

	# combined score: equal weight
	scores = scores.with_columns(
		((pl.col("uncertainty_norm") + pl.col("rare_class_norm")) / 2).alias("combined"),
	)

	# write signals
	signals_dir = config.data.selections_dir.parent / "signals"
	signals_dir.mkdir(parents=True, exist_ok=True)
	scores.write_parquet(signals_dir / "scores.parquet")

	logger.info("Computed scores for %d frames", len(scores))
	logger.info("Top combined score: %.4f, Median: %.4f",
		scores["combined"].max(), scores["combined"].median())

	return scores


