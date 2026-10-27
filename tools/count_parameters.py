"""Export registered parameter counts from a saved checkpoint architecture."""
import argparse
import json
from pathlib import Path
import torch
from cica.config import ModelConfig
from cica.data import file_sha256
from cica.model import CICA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Parameter report already exists')
    checkpoint = torch.load(args.checkpoint, map_location='meta', weights_only=True)
    config = ModelConfig(**checkpoint['model_config'])
    with torch.device('meta'):
        model = CICA(config, load_pretrained=False)
    # Strict shape/key validation ties the count to this checkpoint, rather than a separate model.
    model.load_state_dict(checkpoint['model'], strict=True)
    report = dict(checkpoint_sha256=file_sha256(args.checkpoint),
                  hidden_dim=config.hidden_dim, text_mode=config.text_mode,
                  variant=config.variant, **model.parameter_inventory())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
