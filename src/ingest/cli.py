import click




'''
- Each stage is its own CLI command (python -m src.ingest, python -m src.infer, etc.)                                                                          
- The Makefile is your orchestrator — make pipeline just calls each stage in order

'''

@click.command()
@click.option('--count', default=1, help='Number of times to greet.')
@click.option('--name', prompt='Your name', help='The person to greet.')
@click.option('--shout', is_flag=True, help='Print the message in uppercase.')
@click.argument('filename', type=click.Path())
def ingest():


	return



if __name__ == '__main__':
    ingest()