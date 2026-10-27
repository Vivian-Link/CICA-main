"""Run independently trained one-factor sensitivity conditions."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--sweep', default='configs/sensitivity.json')
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--factor', required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    spec = json.loads(Path(a.sweep).read_text(encoding='utf-8'))
    factors = dict(spec['one_factor_at_a_time'], **spec.get('additional_training_conditions', {}))
    if a.factor not in factors:
        raise ValueError('Unknown sensitivity factor')
    for value in factors[a.factor]:
        out = Path(a.output) / (a.factor + '_' + str(value))
        subprocess.run([sys.executable, '-m', 'cica', 'train', '--config', a.config,
            '--data', a.data, '--output', str(out), '--device', a.device,
            '--seeds', *map(str, spec['seeds']), '--set', a.factor + '=' + json.dumps(value)], check=True)


if __name__ == '__main__':
    main()
