import argparse
import json
import torch
from .engine import train, predict


def main():
    parser = argparse.ArgumentParser(prog='cica', description='CICA training and prediction')
    commands = parser.add_subparsers(dest='command', required=True)
    t = commands.add_parser('train')
    t.add_argument('--config', required=True)
    t.add_argument('--data', required=True)
    t.add_argument('--output', required=True)
    t.add_argument('--variant', choices=['cica', 'uniform'], default='cica')
    t.add_argument('--seeds', type=int, nargs='+', default=[11, 23, 37, 51, 71])
    t.add_argument('--set', action='append', default=[], metavar='SECTION.KEY=JSON')
    p = commands.add_parser('predict')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--split', choices=['validation', 'test'], default='test')
    p.add_argument('--drop', choices=['L', 'V', 'A'], nargs='*', default=[])
    p.add_argument('--missing-rate', type=float, default=0.0)
    for cmd in [t, p]:
        cmd.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
        cmd.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    if args.command == 'train':
        with open(args.config, encoding='utf-8') as f:
            config = json.load(f)
        for override in args.set:
            key, value = override.split('=', 1)
            section, field = key.split('.', 1)
            if section not in config or field not in config[section]:
                parser.error('Unknown setting: ' + key)
            config[section][field] = json.loads(value)
        train(config, args.data, args.output, args.seeds, args.variant, args.device)
    else:
        predict(args.checkpoint, args.data, args.output, args.split, args.device,
                args.drop, args.missing_rate)
