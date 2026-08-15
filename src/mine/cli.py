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