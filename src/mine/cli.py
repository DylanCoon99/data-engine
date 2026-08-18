import logging

import click
from src.config.schema import Config
from src.mine.signals import compute_scores
from src.mine.select import select


@click.command()
@click.option('--config', default='configs/base.yaml', help='Path to config file', type=click.Path(exists=True))
def main(config):
	logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

	config = Config(config)

	compute_scores(config)
	select(config)


if __name__ == '__main__':
    main()
