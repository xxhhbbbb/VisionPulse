import glob
import json
import logging
import os
import time
import warnings

import pandas as pd
import portalocker
import torch
import torch.distributed as dist

from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *

FAIL_MSG = 'Failed to obtain answer via API.'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


def _get_video_job_name(result_file_name):
    return osp.splitext(result_file_name)[0]


def _build_video_prompt(model, dataset, sample_map, task_idx, effective_dataset_name):
    sample = sample_map[task_idx]
    if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(effective_dataset_name):
        if dataset.nframe == 0:
            raise ValueError(f'nframe must be set for custom prompt, fps is not suitable for {effective_dataset_name}')
        struct = model.build_prompt(
            dataset.data.iloc[sample],
            dataset=dataset,
            video_llm=getattr(model, 'VIDEO_LLM', False),
        )
    else:
        struct = dataset.build_prompt(sample, video_llm=getattr(model, 'VIDEO_LLM', False))
    return struct


def _sync_video_model_sampling(model, model_name, dataset):
    dataset_name = dataset.dataset_name

    if 'megabench' in dataset_name.lower() and 'llava_onevision' in model_name.lower():
        print(
            'LLaVA-OneVision does not support Megabench dataset as video dataset, '
            'will set its VIDEO_LLM to False to enable multi-image input for video.'
        )
        setattr(model, 'VIDEO_LLM', False)

    if getattr(model, 'nframe', None) is not None and getattr(model, 'nframe', 0) > 0:
        if dataset.nframe > 0:
            if getattr(model, 'nframe', 0) != dataset.nframe:
                print(f'{model_name} is a video-llm model, nframe is set to {dataset.nframe}, not using default')
                setattr(model, 'nframe', dataset.nframe)
        elif getattr(model, 'fps', 0) == 0:
            raise ValueError(f'fps is not suitable for {model_name}')
        else:
            setattr(model, 'nframe', None)

    if getattr(model, 'fps', None) is not None and getattr(model, 'fps', 0) > 0:
        if dataset.fps > 0:
            if getattr(model, 'fps', 0) != dataset.fps:
                print(f'{model_name} is a video-llm model, fps is set to {dataset.fps}, not using default')
                setattr(model, 'fps', dataset.fps)
        elif getattr(model, 'nframe', 0) == 0:
            raise ValueError(f'nframe is not suitable for {model_name}')
        else:
            setattr(model, 'fps', None)

    if (
        'Qwen2-VL' in model_name
        or 'Qwen2.5-VL' in model_name
        or 'Qwen2.5-Omni' in model_name
    ):
        if getattr(model, 'nframe', None) is None and dataset.nframe > 0:
            print(f'using {model_name} default setting for video, dataset.nframe is ommitted')
        if getattr(model, 'fps', None) is None and dataset.fps > 0:
            print(f'using {model_name} default setting for video, dataset.fps is ommitted')


def _split_prediction_and_thinking(prediction, model=None):
    def split_thinking(s):
        if '</think>' in s:
            splits = s.split('</think>')
            pred = splits[-1].strip()
            if len(splits) == 2 and '<think>' in splits[0]:
                thinking = splits[0].split('<think>')[1].strip()
            else:
                thinking = '</think>'.join(splits[:-1])
                thinking += '</think>'
                warnings.warn('Failed to parse thinking, multiple </think> tags or missing <think> tag.')
        else:
            thinking = ''
            pred = s
        return pred, thinking

    def extract_boxed_content(ans):
        idx = ans.rfind(r'\boxed{')
        if idx == -1:
            return ans

        idx += len(r'\boxed{')
        brace_level = 1
        content_start = idx
        i = idx

        while i < len(ans):
            if ans[i] == '{':
                brace_level += 1
            elif ans[i] == '}':
                brace_level -= 1
                if brace_level == 0:
                    break
            i += 1

        if brace_level != 0:
            return ans

        return ans[content_start:i]

    split_func = model.split_thinking if model is not None and hasattr(model, 'split_thinking') else split_thinking
    pairs = [split_func(x) for x in prediction]
    pairs = [(extract_boxed_content(x[0]), x[1]) for x in pairs]
    return pairs


def _extract_prev_results_from_result_file(result_file):
    data = load(result_file)
    if isinstance(data, pd.DataFrame):
        if os.getenv('SPLIT_THINK', False) and 'thinking' in data.columns:
            results = {}
            for idx, pred, think in zip(data['index'], data['prediction'], data['thinking']):
                pred_str = str(pred)
                think_str = str(think).strip()
                if think_str:
                    results[idx] = f"<think>{think_str}</think> {pred_str}"
                else:
                    results[idx] = pred_str
        else:
            results = {k: v for k, v in zip(data['index'], data['prediction'])}
    else:
        results = data
    results = {k: v for k, v in results.items() if FAIL_MSG not in str(v)}
    return results


# This entry point accepts API models only.
def infer_data_api(model, work_dir, model_name, dataset, samples_dict={}, api_nproc=4):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    model = supported_VLM[model_name]() if isinstance(model, str) else model
    assert getattr(model, 'is_api', False)

    indices = list(samples_dict.keys())
    if getattr(model, 'backend', None) == 'genai':
        if dataset.nframe > 0:
            print(
                'Gemini model (with genai backend) does not support nframe, '
                'will set its VIDEO_LLM to False to enable multi-image input for video.'
            )
            setattr(model, 'VIDEO_LLM', False)
        else:
            print(
                'Gemini model (with genai backend) is a video-llm, '
                'will reset fps setting in model to match the dataset.'
            )
            setattr(model, 'fps', dataset.fps)
            print(f'The fps is set to {dataset.fps} for the model {model_name}.')
    elif getattr(model, 'backend', None) == 'vertex':
        print(
            'Gemini model (with vertex backend) does not support video input, '
            'will set its VIDEO_LLM to False to enable multi-image input for video.'
        )
        setattr(model, 'VIDEO_LLM', False)

    packstr = 'pack' if getattr(dataset, 'pack', False) else 'nopack'
    build_prompt_input = [(samples_dict[idx], getattr(model, 'VIDEO_LLM', False)) for idx in indices]
    if dataset.nframe > 0:
        struct_tmp_file = f'{work_dir}/{model_name}_{dataset.dataset_name}_{dataset.nframe}frame_{packstr}_structs.pkl'
    else:
        struct_tmp_file = f'{work_dir}/{model_name}_{dataset.dataset_name}_{dataset.fps}fps_{packstr}_structs.pkl'
    structs = track_progress_rich(
        dataset.build_prompt,
        tasks=build_prompt_input,
        nproc=api_nproc,
        save=struct_tmp_file,
        keys=indices,
    )

    if dataset.nframe > 0:
        out_file = f'{work_dir}/{model_name}_{dataset.dataset_name}_{dataset.nframe}frame_{packstr}_supp.pkl'
    else:
        out_file = f'{work_dir}/{model_name}_{dataset.dataset_name}_{dataset.fps}fps_{packstr}_supp.pkl'
    res = load(out_file) if osp.exists(out_file) else {}

    structs = [s for i, s in zip(indices, structs) if i not in res or res[i] == FAIL_MSG]
    structs = [struct for struct in structs if struct is not None]
    indices = [i for i in indices if i not in res or res[i] == FAIL_MSG]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset.dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    return res


def infer_data_new_video(
    model,
    model_name,
    work_dir,
    dataset,
    out_file,
    result_file_name,
    verbose=False,
    api_nproc=4,
    use_vllm=False,
    budget=1,
):
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name
    job_name = _get_video_job_name(result_file_name)
    log_adapter = logging.LoggerAdapter(logging.getLogger(), {'rank': rank})

    sample_indices = list(dataset.videos) if getattr(dataset, 'pack', False) else list(dataset.data['index'])
    samples = list(dataset.videos) if getattr(dataset, 'pack', False) else list(range(len(dataset.data)))
    sample_map = {i: s for i, s in zip(sample_indices, samples)}

    task_pool_file = osp.join(work_dir, f'{job_name}_task_pool.json')
    prev_file = osp.join(work_dir, f'{job_name}_PREV.pkl')

    # Collect previous results before creating the shared task pool.
    all_prev_results = {}
    if rank == 0:
        log_adapter.info(f'Rank 0: Aggregating previous video results for {job_name} ...')
        if osp.exists(prev_file):
            all_prev_results.update(load(prev_file))

        pattern = osp.join(work_dir, f'*_{job_name}.pkl')
        for rank_out_file in glob.glob(pattern):
            if rank_out_file.endswith('_PREV.pkl'):
                continue
            try:
                rank_data = load(rank_out_file)
            except Exception as err:
                log_adapter.warning(f'Failed to load historical rank file {rank_out_file}: {err}')
                continue
            all_prev_results.update(rank_data)

        if not osp.exists(task_pool_file):
            remaining_indices = [idx for idx in sample_indices if idx not in all_prev_results]
            task_pool = {
                'indices': remaining_indices,
                'next_task': 0,
                'finished_tasks': 0,
                'total_tasks': len(remaining_indices),
            }
            dump(task_pool, task_pool_file)
            log_adapter.info(f'Rank 0: Task pool created with {len(remaining_indices)} video tasks.')

    if world_size > 1:
        # Broadcast completed indices instead of full prediction records.
        prev_keys = list(all_prev_results.keys())
        object_list = [prev_keys]
        dist.broadcast_object_list(object_list, src=0)
        prev_keys = set(object_list[0])
        dist.barrier()
    else:
        prev_keys = set(all_prev_results.keys())

    res = load(out_file) if osp.exists(out_file) else {}
    log_adapter.info(f'Loaded {len(res)} local video results. Global finished: {len(prev_keys)}')

    kwargs = {}
    if model_name is not None and (
        'Llama-4' in model_name
        or 'Qwen2-VL' in model_name
        or 'Qwen2.5-VL' in model_name
        or 'Qwen2.5-Omni' in model_name
    ):
        kwargs = {'use_vllm': use_vllm}

    # Avoid automatic tensor parallelism when loading one model per worker.
    ws_bak = os.environ.pop('WORLD_SIZE', None)
    model = supported_VLM[model_name](**kwargs) if isinstance(model, str) else model
    if ws_bak:
        os.environ['WORLD_SIZE'] = ws_bak

    is_api = getattr(model, 'is_api', False)
    if is_api:
        pending = [idx for idx in sample_indices if idx not in prev_keys and idx not in res]
        supp = infer_data_api(
            model=model,
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            samples_dict={k: sample_map[k] for k in pending},
            api_nproc=api_nproc,
        )
        for idx, value in supp.items():
            res[idx] = {'prediction': value}
        dump(res, out_file)
        return model

    assert not getattr(dataset, 'pack', False), 'Current model not supported pack mode!'
    if hasattr(model, 'set_dump_image') and hasattr(dataset, 'dump_image'):
        model.set_dump_image(dataset.dump_image)
    _sync_video_model_sampling(model, model_name, dataset)

    loop_start_time = time.time()
    processed_count = 0

    while True:
        # Claim one task while holding the shared task-pool lock.
        task_idx = None
        try:
            with portalocker.Lock(task_pool_file, 'r+', timeout=60) as f:
                task_pool = json.load(f)
                current_task_num = task_pool['next_task']
                if current_task_num < len(task_pool['indices']):
                    task_idx = task_pool['indices'][current_task_num]
                    task_pool['next_task'] += 1
                    f.seek(0)
                    f.truncate()
                    json.dump(task_pool, f)
                else:
                    break
        except (portalocker.exceptions.LockException, FileNotFoundError) as err:
            log_adapter.error(f'Rank {rank} failed to access task pool: {err}. Retrying...')
            time.sleep(rank * 0.1 + 0.1)
            continue

        if task_idx is None:
            break
        if task_idx in res or task_idx in prev_keys:
            continue

        sample = sample_map[task_idx]
        effective_dataset_name = dataset_name
        if (
            not getattr(dataset, 'pack', False)
            and isinstance(sample, int)
            and 'SUB_DATASET' in dataset.data.iloc[sample]
        ):
            effective_dataset_name = dataset.data.iloc[sample]['SUB_DATASET']

        struct = _build_video_prompt(model, dataset, sample_map, task_idx, effective_dataset_name)
        if struct is None:
            continue


        if os.environ.get('SKIP_ERR', '0') == '1':
            try:
                model.budget = budget
                response = model.generate(message=struct, dataset=effective_dataset_name)
            except RuntimeError as err:
                torch.cuda.synchronize()
                warnings.warn(f'{type(err)} {str(err)}')
                response = f'{FAIL_MSG}: {type(err)} {str(err)}'
        else:
            model.budget = budget
            response = model.generate(message=struct, dataset=effective_dataset_name)
        torch.cuda.empty_cache()

        res[task_idx] = {'prediction': response}
        processed_count += 1

        try:
            with portalocker.Lock(task_pool_file, 'r+', timeout=60) as f:
                task_pool = json.load(f)
                task_pool['finished_tasks'] += 1
                f.seek(0)
                f.truncate()
                json.dump(task_pool, f)
        except Exception as err:
            log_adapter.warning(f'Rank {rank} failed to update video progress: {err}')

        if verbose:
            verbose_str = (
                f'================================================\n'
                f'Rank: {rank}, Task: {task_idx}\n'
                f'------------------------------------------------\n'
                f'PROMPT: {struct}\n'
                f'------------------------------------------------\n'
                f'RESPONSE: {response}\n'
                f'================================================'
            )
            print(verbose_str, flush=True)

        if processed_count > 0 and processed_count % 5 == 0:
            dump(res, out_file)

    dump(res, out_file)
    loop_end_time = time.time()
    log_adapter.info(
        f'Rank {rank}: Finished {processed_count} video tasks for {job_name}. '
        f'Time: {loop_end_time - loop_start_time:.2f}s.'
    )

    if world_size > 1:
        dist.barrier()
    return model


# Prepare resume state, run video inference, and merge worker results.
def infer_data_job_video(
    model,
    work_dir,
    model_name,
    dataset,
    result_file_name,
    verbose=False,
    api_nproc=4,
    use_vllm=False,
    budget=1,
):
    dataset_name = dataset.dataset_name
    rank, world_size = get_rank_and_world_size()
    result_file = osp.join(work_dir, result_file_name)
    job_name = _get_video_job_name(result_file_name)

    if rank == 0 and not getattr(logging.getLogger(), '_vlmeval_configured', False):
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        log_file = osp.join(work_dir, f'inference_log_{time.strftime("%Y%m%d-%H%M%S")}.log')
        formatter = logging.Formatter(
            '%(asctime)s - RANK %(rank)s - %(levelname)s - %(message)s',
            defaults={'rank': 'N/A'},
        )
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        setattr(root_logger, '_vlmeval_configured', True)

    if world_size > 1:
        dist.barrier()

    prev_file = osp.join(work_dir, f'{job_name}_PREV.pkl')
    if osp.exists(result_file):
        if rank == 0:
            try:
                dump(_extract_prev_results_from_result_file(result_file), prev_file)
            except Exception as err:
                print(f'Failed to prepare prev results from {result_file}: {err}')
        if world_size > 1:
            dist.barrier()

    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{job_name}.pkl')
    out_file = tmpl.format(rank)

    model = infer_data_new_video(
        model=model,
        model_name=model_name,
        work_dir=work_dir,
        dataset=dataset,
        out_file=out_file,
        result_file_name=result_file_name,
        verbose=verbose,
        api_nproc=api_nproc,
        use_vllm=use_vllm,
        budget=budget,
    )

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data_all = {}

        for i in range(world_size):
            rank_file = tmpl.format(i)
            if not osp.exists(rank_file):
                continue
            rank_data = load(rank_file)
            for idx, item in rank_data.items():
                if isinstance(item, dict) and 'prediction' in item:
                    data_all[idx] = item['prediction']
                else:
                    data_all[idx] = item
            os.remove(rank_file)

        # Recover worker files left by interrupted runs.
        # Current predictions take precedence over historical results.
        pattern_hist = osp.join(work_dir, f'*_{job_name}.pkl')
        for rank_out_file in glob.glob(pattern_hist):
            if rank_out_file.endswith('_PREV.pkl'):
                continue
            if not osp.exists(rank_out_file):
                continue
            try:
                rank_data = load(rank_out_file)
            except Exception as err:
                print(f'Failed to load historical rank file {rank_out_file}: {err}')
                continue

            for idx, item in rank_data.items():
                if idx in data_all:
                    continue
                if isinstance(item, dict) and 'prediction' in item:
                    data_all[idx] = item['prediction']
                else:
                    data_all[idx] = item
            os.remove(rank_out_file)

        if osp.exists(prev_file):
            try:
                prev_results = load(prev_file)
                for idx, val in prev_results.items():
                    if idx not in data_all:
                        data_all[idx] = val
            except Exception as err:
                print(f'Failed to load prev results from {prev_file}: {err}')

        task_pool_file = osp.join(work_dir, f'{job_name}_task_pool.json')
        if osp.exists(task_pool_file):
            os.remove(task_pool_file)

        # Preserve dataset row order and optionally separate reasoning traces.
        meta = dataset.data
        if dataset_name == 'MMBench-Video' and getattr(dataset, 'pack', False):
            meta, vstats = dataset.load_pack_answers(data_all)
            print(f'Statitics of Pack Video Inference: {vstats}')
        else:
            prediction = [str(data_all.get(x, FAIL_MSG)) for x in meta['index']]
            if os.getenv('SPLIT_THINK', False):
                pairs = _split_prediction_and_thinking(prediction, model=model)
                meta['prediction'] = [x[0] for x in pairs]
                meta['thinking'] = [x[1] for x in pairs]
            else:
                meta['prediction'] = prediction
            if 'image' in meta:
                meta.pop('image')

        dump(meta, result_file)

        try:
            total = len(meta)
            fail_count = 0
            if 'prediction' in meta:
                for p in meta['prediction']:
                    if FAIL_MSG in str(p):
                        fail_count += 1
            log_adapter = logging.LoggerAdapter(logging.getLogger(), {'rank': rank})
            log_adapter.info(
                f"{model_name} x {dataset_name}: {fail_count}/{total} samples FAILED (contain '{FAIL_MSG}')"
            )
            print(f'[SUMMARY] {model_name} x {dataset_name}: {fail_count}/{total} samples FAILED.')
        except Exception:
            pass

    if world_size > 1:
        dist.barrier()
    return model
