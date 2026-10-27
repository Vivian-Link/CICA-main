import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from cica.cli import main


class CliTests(unittest.TestCase):
    def run_command(self, extra):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'settings.json'
            path.write_text(json.dumps({'dataset': 'SMOKE', 'seeds': [7, 17]}), encoding='utf-8')
            argv = ['cica', 'train', '--config', str(path), '--data', 'data', '--output', 'runs', *extra]
            with patch('sys.argv', argv), patch('cica.cli.train') as train:
                main()
                return train.call_args.args

    def test_reads_seeds_from_configuration(self):
        args = self.run_command([])
        self.assertEqual(args[3], [7, 17])
        self.assertEqual(args[0]['seeds'], args[3])

    def test_explicit_seeds_override_configuration(self):
        args = self.run_command(['--seeds', '3', '9'])
        self.assertEqual(args[3], [3, 9])
        self.assertEqual(args[0]['seeds'], args[3])


if __name__ == '__main__':
    unittest.main()
