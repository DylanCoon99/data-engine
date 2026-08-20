import logging

import click
from src.config.schema import Config
from src.evaluate.metrics import evaluate
from src.evaluate.report import generate_report


@click.command()
@click.option('--config', default='configs/base.yaml', help='Path to config file', type=click.Path(exists=True))
def main(config):
	logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

	config = Config(config)

	mined_metrics, random_metrics = evaluate(config)

	if mined_metrics and random_metrics:
		output_dir = config.data.yolo_dir / "reports"
		generate_report(mined_metrics, random_metrics, output_dir)


if __name__ == '__main__':
    main()
