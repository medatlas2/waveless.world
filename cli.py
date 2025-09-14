#!/usr/bin/env python3
import click, json, time
from pathlib import Path

@click.group()
def cli():
    """Waveless World CLI (prototype)
    - scan: mock spectrum scan
    - annotate: tag events
    """

@cli.command()
@click.option('--seconds', default=5, help='Scan duration')
def scan(seconds):
    """Mock scanner (placeholder for RF/ultrasonic capture)."""
    start=time.time()
    events=[]
    while time.time()-start<seconds:
        time.sleep(0.2)
    click.echo(json.dumps({'events': events, 'duration': seconds}))

@cli.command()
@click.option('--file', type=click.Path(), required=True)
@click.option('--label', required=True)
def annotate(file, label):
    """Attach a label to a capture file (placeholder)."""
    p=Path(file)
    meta=p.with_suffix(p.suffix+'.meta.json')
    meta.write_text(json.dumps({'label': label, 'source': p.name}))
    click.echo(str(meta))

if __name__ == '__main__':
    cli()
