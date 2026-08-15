import click
from src.config.schema import Config




'''
- Each stage is its own CLI command (python -m src.train, python -m src.infer, etc.)                                                                          
- The Makefile is your orchestrator — make pipeline just calls each stage in order

├── train/                    # Stage 4: train both arms (mined + random)
│   ├── __init__.py
│   ├── prepare.py            # Build YOLO dataset dirs from selection manifest + seed pool
│   ├── run.py                # Kick off YOLOv8 training, log to MLflow
│   └── cli.py
'''

@click.command()
@click.option('--config', default='configs/base.yaml', help='Path to config file', type=click.Path(exists=True))
def main(config):

	# test the yaml parsing
	config = Config(config)


	print(config)

	return



if __name__ == '__main__':
    main()