"""Run an analysis example and leave only its result PNG in the output directory."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from analysis.collect import DEFAULT_QUESTION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tool', choices=['activation', 'attention_mass', 'layer_similarity'])
    parser.add_argument('--model', default='Qwen/Qwen3-VL-4B-Thinking')
    parser.add_argument('--image', default='assets/vent_example.png')
    parser.add_argument('--question', default=DEFAULT_QUESTION)
    parser.add_argument('--layer', type=int, default=17, help='Zero-based attention layer')
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, nargs='+', help='Activation panels: zero-based generated steps')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    output = args.output_dir or Path('outputs/analysis') / args.tool
    modules = {
        'activation': 'analysis.activation.visualize',
        'attention_mass': 'analysis.attention_mass.correlate',
        'layer_similarity': 'analysis.layer_similarity.similarity',
    }
    plot = [sys.executable, '-m', modules[args.tool], '--output-dir', str(output)]
    if args.steps is not None and args.tool != 'activation':
        parser.error('--steps is only used by activation.')
    # Attention traces are temporary intermediate data, not additional user outputs.
    with tempfile.TemporaryDirectory(prefix='visionpulse-analysis-') as temporary:
        collect = [sys.executable, '-m', 'analysis.collect', '--model', args.model,
                   '--image', args.image, '--question', args.question, '--layer', str(args.layer),
                   '--max-new-tokens', str(args.max_new_tokens), '--seed', str(args.seed),
                   '--output-dir', temporary]
        if args.tool == 'layer_similarity':
            collect.append('--all-layers')
        subprocess.run(collect, check=True)
        sample = Path(temporary) / 'sample_0000'
        metadata = json.loads((sample / 'metadata.json').read_text())
        if not metadata['stopped_on_eos']:
            raise SystemExit('Generation reached its token limit. Increase --max-new-tokens to plot a complete response.')
        trace = sample / 'trace.npz'
        if args.tool == 'activation' and args.steps is not None:
            plot += ['--steps', *map(str, args.steps)]
        subprocess.run(plot + ['--data', str(trace)], check=True)


if __name__ == '__main__':
    main()
