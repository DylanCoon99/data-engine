import click
from src.config.schema import Config




'''
- Each stage is its own CLI command (python -m src.evaluate, python -m src.infer, etc.)                                                                          
- The Makefile is your orchestrator — make pipeline just calls each stage in order


└── evaluate/                 # Stage 5: evaluate both arms, generate report
    ├── __init__.py
    ├── metrics.py            # mAP comparison, per-class AP
    ├── report.py             # Generate plots and summary
    └── cli.py
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