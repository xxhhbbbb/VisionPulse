"""Collect dense Qwen3-VL visual attention for the three observational analyses."""
import argparse
import json
from pathlib import Path

import numpy as np

from analysis.common import cosine_matrix, write_json

DEFAULT_QUESTION = (
    'Question: Where is the brown square vent relative to the door?\n'
    'Options:\nA. There is no brown square vent.\n'
    'B. The brown square vent is to the left of the door.\n'
    'C. The brown square vent is to the right of the door.\n'
    'Please select the correct answer from the options above.'
)


class AttentionCollector:
    """Keep only the last query's head-mean visual vector, never full attention maps."""
    def __init__(self, layers, anchor_layer, visual_indices):
        self.layers = list(layers)
        self.anchor_layer = anchor_layer
        self.visual_indices = visual_indices
        self.pending = {}
        self.scores = []
        self.similarities = []
        self.handles = []

    def hook(self, layer_index):
        def capture(module, inputs, outputs):
            if not isinstance(outputs, tuple) or len(outputs) < 2 or outputs[1] is None:
                raise RuntimeError('Attention weights are unavailable; use eager language-model attention.')
            weights = outputs[1]
            if weights.ndim != 4 or weights.shape[0] != 1:
                raise ValueError('Analysis supports batch size 1 and [batch, heads, queries, keys] attention.')
            if layer_index in self.pending:
                raise RuntimeError('Duplicate layer hook before the current generation step completed.')
            indices = self.visual_indices.to(weights.device)
            visual = weights[0, :, -1, :].index_select(-1, indices).float().mean(0)
            self.pending[layer_index] = visual.detach().cpu().numpy()
            if layer_index == self.layers[-1]:
                if set(self.pending) != set(self.layers):
                    raise RuntimeError('Missing attention layers in a generation step.')
                self.scores.append(self.pending[self.anchor_layer])
                if len(self.layers) > 1:
                    self.similarities.append(cosine_matrix(np.stack([self.pending[i] for i in self.layers])))
                self.pending.clear()
        return capture

    def register(self, model):
        layers = model.model.language_model.layers
        try:
            for i in self.layers:
                self.handles.append(layers[i].self_attn.register_forward_hook(self.hook(i)))
        except Exception:
            self.remove()
            raise

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def samples_from_args(args):
    if args.samples:
        path = Path(args.samples).resolve()
        samples = json.loads(path.read_text())
        if not isinstance(samples, list) or not samples:
            raise ValueError('Sample manifest must be a nonempty JSON list.')
        for i, item in enumerate(samples):
            image = Path(item['image']).expanduser()
            if not image.is_absolute():
                image = path.parent / image
            yield {'id': str(item.get('id', i)), 'image': str(image.resolve()), 'question': item['question']}
    elif args.dataset:
        from vlmeval.dataset import build_dataset
        dataset = build_dataset(args.dataset)
        if dataset is None:
            raise ValueError(f'Cannot load dataset {args.dataset}')
        rows = dataset.data.sample(n=min(args.sample_size, len(dataset.data)), random_state=args.seed)
        for _, row in rows.iterrows():
            message = dataset.build_prompt(row)
            images = [x['value'] for x in message if x['type'] == 'image']
            if len(images) != 1:
                raise ValueError('Analysis requires exactly one image per sample.')
            yield {'id': str(row['index']), 'image': images[0],
                   'question': '\n'.join(x['value'] for x in message if x['type'] == 'text')}
    else:
        yield {'id': 'example', 'image': str(Path(args.image).expanduser().resolve()), 'question': args.question}


def collect_sample(model, processor, sample, args, output_dir):
    import torch
    from PIL import Image
    from transformers import set_seed

    set_seed(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(sample['image']) as source:
        image = source.convert('RGB')
    messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': sample['question']}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors='pt').to(model.device)
    indices = torch.where(inputs.input_ids[0] == model.config.image_token_id)[0]
    grid = inputs.image_grid_thw[0].tolist()
    merge = processor.image_processor.merge_size
    grid_hw = np.array([grid[1] // merge, grid[2] // merge], dtype=np.int64)
    if grid[0] != 1 or int(np.prod(grid_hw)) != len(indices) or len(indices) == 0:
        raise ValueError('Only a single still image with a matching merged visual-token grid is supported.')
    layers = list(range(model.config.text_config.num_hidden_layers)) if args.all_layers else [args.layer]
    hook = AttentionCollector(layers, args.layer, indices)
    hook.register(model)
    try:
        generation = dict(max_new_tokens=args.max_new_tokens, do_sample=args.do_sample, use_cache=True)
        if args.do_sample:
            generation.update(temperature=0.7, top_p=0.8, top_k=20)
        with torch.inference_mode():
            outputs = model.generate(**inputs, **generation)
    finally:
        hook.remove()
    ids = outputs[0, inputs.input_ids.shape[1]:].detach().cpu().numpy()
    if not len(ids) or len(hook.scores) != len(ids) or hook.pending:
        raise RuntimeError(f'Generation/attention alignment failed: {len(ids)} tokens, {len(hook.scores)} rows.')
    token_texts = [processor.tokenizer.decode([int(i)], skip_special_tokens=False) for i in ids]
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else (eos or [])
    metadata = {
        'format_version': 1, 'sample_id': sample['id'], 'question': sample['question'],
        'model': args.model, 'layer': args.layer, 'layers': layers, 'seed': args.seed,
        'attention': 'dense eager, head-mean probabilities, temperature=1',
        'generation': generation, 'min_pixels': args.min_pixels, 'max_pixels': args.max_pixels,
        'generated_tokens': len(ids), 'stopped_on_eos': int(ids[-1]) in eos,
        'step_alignment': 'Row 0 is the last prompt query predicting token_ids[0]; row t predicts token_ids[t].',
    }
    fields = dict(visual_scores=np.stack(hook.scores), token_ids=ids,
                  token_texts=np.asarray(token_texts), grid_hw=grid_hw,
                  layer_indices=np.asarray(layers), metadata=np.asarray(json.dumps(metadata)))
    if hook.similarities:
        fields['similarity_by_step'] = np.stack(hook.similarities)
    np.savez_compressed(output_dir / 'trace.npz', **fields)
    image.save(output_dir / 'image.png')
    (output_dir / 'response.txt').write_text(processor.tokenizer.decode(ids, skip_special_tokens=True))
    write_json(output_dir / 'metadata.json', metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--image', default='assets/vent_example.png')
    source.add_argument('--samples', help='JSON list of {id, image, question}; image paths relative to the JSON file')
    source.add_argument('--dataset', help='Bundled VLMEvalKit single-image dataset name')
    parser.add_argument('--question', default=DEFAULT_QUESTION)
    parser.add_argument('--sample-size', type=int, default=50)
    parser.add_argument('--model', default='Qwen/Qwen3-VL-4B-Thinking')
    parser.add_argument('--layer', type=int, default=17, help='Zero-based layer for activation and mass analysis')
    parser.add_argument('--all-layers', action='store_true', help='Also collect per-step inter-layer cosine matrices')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--min-pixels', type=int, default=1280 * 32 * 32)
    parser.add_argument('--max-pixels', type=int, default=4096 * 32 * 32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--do-sample', action='store_true', help='Use sampling instead of greedy decoding')
    parser.add_argument('--vision-attention', choices=['eager', 'sdpa', 'flash_attention_2'], default='sdpa')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/analysis/traces'))
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.sample_size < 1 or not 0 < args.min_pixels <= args.max_pixels:
        parser.error('Token/sample counts must be positive and 0 < min-pixels <= max-pixels.')
    samples = list(samples_from_args(args))
    for sample in samples:
        if not Path(sample['image']).is_file() or not isinstance(sample['question'], str) or not sample['question'].strip():
            parser.error(f'Invalid image/question for sample {sample["id"]}')
    if not samples:
        parser.error('No samples selected.')
    # Keep collection runs separate so failed or shorter reruns cannot leave stale traces.
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Output directory is not empty; choose a fresh directory for this collection.')
    import torch
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
    if not torch.cuda.is_available():
        parser.error('Collection requires a CUDA GPU; plotting saved traces is CPU-only.')
    config = AutoConfig.from_pretrained(args.model)
    if config.model_type != 'qwen3_vl' or not 0 <= args.layer < config.text_config.num_hidden_layers:
        parser.error('Use a dense Qwen3-VL checkpoint and an in-range language-model layer index.')
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype='auto', device_map='auto',
        attn_implementation={'text_config': 'eager', 'vision_config': args.vision_attention},
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {'samples': [], 'num_valid_samples': 0, 'num_failed_samples': 0}
    for i, sample in enumerate(samples):
        directory = args.output_dir / f'sample_{i:04d}'
        try:
            meta = collect_sample(model, processor, sample, args, directory)
            summary['samples'].append({'id': sample['id'], 'directory': directory.name, 'status': 'ok',
                                       'generated_tokens': meta['generated_tokens']})
            summary['num_valid_samples'] += 1
            print(f'[{i + 1}/{len(samples)}] {sample["id"]}: {meta["generated_tokens"]} steps', flush=True)
        except Exception as exc:
            (directory / 'trace.npz').unlink(missing_ok=True)
            summary['samples'].append({'id': sample['id'], 'status': 'failed', 'error': str(exc)})
            summary['num_failed_samples'] += 1
            print(f'[{i + 1}/{len(samples)}] FAILED {sample["id"]}: {exc}', flush=True)
        write_json(args.output_dir / 'summary.json', summary)
    if summary['num_failed_samples']:
        raise SystemExit('Some samples failed; inspect summary.json. Successful traces have been retained.')


if __name__ == '__main__':
    main()
