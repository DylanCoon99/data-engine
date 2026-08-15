import click
from src.config.schema import Config




'''
- Each stage is its own CLI command (python -m src.infer, python -m src.infer, etc.)                                                                          
- The Makefile is your orchestrator — make pipeline just calls each stage in order

├── infer/                    # Stage 2: batch inference over unlabeled pool
│   ├── __init__.py
│   ├── batch.py              # Chunked, resumable inference → Parquet shards
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