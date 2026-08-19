import logging

import click
from src.config.schema import Config
from src.train.prepare import prepare
from src.train.run import run


@click.command()
@click.option('--config', default='configs/base.yaml', help='Path to config file', type=click.Path(exists=True))
def main(config):
	logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

	config = Config(config)

	prepare(config)
	run(config)


if __name__ == '__main__':
    main()
