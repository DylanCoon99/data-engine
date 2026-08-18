import click
from src.config.schema import Config




'''
- Each stage is its own CLI command (python -m src.mine, python -m src.infer, etc.)                                                                          
- The Makefile is your orchestrator — make pipeline just calls each stage in order


├── mine/                     # Stage 3: signal computation + frame selection
│   ├── __init__.py
│   ├── signals.py            # Uncertainty score, rare-class score
│   ├── select.py             # Rank, diversity constraints, output selection manifest
│   └── cli.py


1. Compute scores — for each frame, calculate two signals:
	- Uncertainty: how confused is the model? Frames with lots of detections in the 0.3-0.7 confidence range are frames where the model isn't
	sure. Those are valuable to label.
	- Rare-class presence: does the model think it sees a motorcycle, bicycle, or train? Those classes are underrepresented, so any frame
	containing them is worth labeling.
2. Combine scores — normalize both signals to [0,1], take a weighted sum. Each frame gets a single "value" score.
3. Select frames — rank all frames by score, pick the top N (your label_budget of 2000), but with diversity constraints (cap how many frames 
come from similar scenes) so you don't just grab 2000 nearly-identical frames.

Output: A selection manifest — a Parquet file listing the 2000 frame IDs chosen, with their scores. This is the key artifact. A separate random
selection of 2000 frames is also generated for the baseline comparison.

Both manifests feed into the train stage, where you "reveal" the ground truth for those frames and train a model on each set.

'''

@click.command()
@click.option('--config', default='configs/base.yaml', help='Path to config file', type=click.Path(exists=True))
def main(config):

	# test the yaml parsing
	config = Config(config)

	

	return



if __name__ == '__main__':
    main()