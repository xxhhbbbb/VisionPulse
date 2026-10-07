import torch
import torch.distributed as dist
from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *
import logging
import time
import portalocker
import json
import os
import glob
import pandas as pd
import warnings
FAIL_MSG = 'Failed to obtain answer'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


# This entry point accepts API models only.
def infer_data_api(model, work_dir, model_name, dataset, index_set=None, api_nproc=4, ignore_failed=False):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    data = dataset.data
    if index_set is not None:
        data = data[data['index'].isin(index_set)]

    model = supported_VLM[model_name]() if isinstance(model, str) else model
    assert getattr(model, 'is_api', False)
    if hasattr(model, 'set_dump_image'):
        model.set_dump_image(dataset.dump_image)

    lt, indices = len(data), list(data['index'])

    structs = []
    for i in range(lt):
        item = data.iloc[i]
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            assert hasattr(model, 'build_prompt')
            struct = model.build_prompt(item, dataset=dataset_name)
        else:
            struct = dataset.build_prompt(item)
        structs.append(struct)

    out_file = f'{work_dir}/{model_name}_{dataset_name}_supp.pkl'

    # Reuse compatible predictions from MMBench_V11.
    if dataset_name in ['MMBench', 'MMBench_CN']:
        pred_format = get_pred_file_format()
        v11_pred = f'{work_dir}/{model_name}_{dataset_name}_V11.{pred_format}'
        if osp.exists(v11_pred):
            try:
                reuse_inds = load('http://opencompass.openxlab.space/utils/mmb_reuse.pkl')
                data = load(v11_pred)
                ans_map = {x: y for x, y in zip(data['index'], data['prediction']) if x in reuse_inds}
                dump(ans_map, out_file)
            except Exception as err:
                print(type(err), err)

    res = {}
    if osp.exists(out_file):
        res = load(out_file)
        if ignore_failed:
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

    structs = [s for i, s in zip(indices, structs) if i not in res]
    indices = [i for i in indices if i not in res]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    if index_set is not None:
        res = {k: v for k, v in res.items() if k in index_set}
    os.remove(out_file)
    return res


def infer_data(model, model_name, work_dir, dataset, out_file, verbose=False, api_nproc=4, use_vllm=False):
    dataset_name = dataset.dataset_name
    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    res = load(prev_file) if osp.exists(prev_file) else {}
    if osp.exists(out_file):
        res.update(load(out_file))

    rank, world_size = get_rank_and_world_size()

    # Include the worker rank in log records.
    log_adapter = logging.LoggerAdapter(logging.getLogger(), {'rank': rank})

    sheet_indices = list(range(rank, len(dataset), world_size))
    lt = len(sheet_indices)
    data = dataset.data.iloc[sheet_indices]
    data_indices = [i for i in data['index']]

    # Return without loading the model when all assigned samples are complete.
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model

    # Select samples without saved predictions.
    data = data[~data['index'].isin(res)]
    lt = len(data)
    log_adapter.info(f"Starting inference for {lt} items on dataset {dataset_name}.")

    kwargs = {}
    if model_name is not None and (
        'Llama-4' in model_name
        or 'Qwen2-VL' in model_name
        or 'Qwen2.5-VL' in model_name
    ):
        kwargs = {'use_vllm': use_vllm}

    # (25.06.05) In newer version of transformers (after 4.50), with device_map='auto' and torchrun launcher,
    # Transformers automatically adopt TP parallelism, which leads to compatibility problems with VLMEvalKit
    # (In VLMEvalKit, we use torchrun to launch multiple model instances on a single node).
    # To bypass this problem, we unset `WORLD_SIZE` before building the model to not use TP parallel.
    ws_bak = os.environ.pop('WORLD_SIZE', None)
    model = supported_VLM[model_name](**kwargs) if isinstance(model, str) else model
    if ws_bak:
        os.environ['WORLD_SIZE'] = ws_bak

    is_api = getattr(model, 'is_api', False)
    if is_api:
        lt, indices = len(data), list(data['index'])
        supp = infer_data_api(
            model=model,
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            index_set=set(indices),
            api_nproc=api_nproc)
        for idx in indices:
            assert idx in supp
        res.update(supp)
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model
    else:
        model.set_dump_image(dataset.dump_image)

    loop_start_time = time.time()

    for i in tqdm(range(lt), desc=f'Infer {model_name}/{dataset_name}, Rank {rank}/{world_size}'):

        idx = data.iloc[i]['index']
        if idx in res:
            continue

        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            struct = model.build_prompt(data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(data.iloc[i])
        # Record RuntimeError failures and continue when SKIP_ERR is enabled.
        if os.environ.get('SKIP_ERR', '0') == '1':
            try:
                response = model.generate(message=struct, dataset=dataset_name)
            except RuntimeError as err:
                torch.cuda.synchronize()
                warnings.warn(f'{type(err)} {str(err)}')
                # Use the shared failure marker for downstream filtering.
                response = f'{FAIL_MSG}: {type(err)} {str(err)}'
        else:
            response = model.generate(message=struct, dataset=dataset_name)
        torch.cuda.empty_cache()

        if verbose:
            print(response, flush=True)

        res[idx] = response
        if (i + 1) % 5 == 0:
            dump(res, out_file)

    loop_end_time = time.time()
    log_adapter.info(f"Finished inference for {lt} items. Total time: {loop_end_time - loop_start_time:.2f} seconds.")

    res = {k: res[k] for k in data_indices}
    dump(res, out_file)
    return model


def infer_data_new(model, model_name, work_dir, dataset, out_file, verbose=False, api_nproc=4, use_vllm=False, budget=1):
    dataset_name = dataset.dataset_name
    rank, world_size = get_rank_and_world_size()

    # Use the root logger handlers and attach the worker rank.
    log_adapter = logging.LoggerAdapter(logging.getLogger(), {'rank': rank})

    log_adapter.info(f"Starting inference on dataset {dataset_name}.")
    task_pool_file = osp.join(work_dir, f"{model_name}_{dataset_name}_task_pool.json")

    # Collect previous results to resume interrupted runs.
    all_prev_results = {}
    if rank == 0:
        log_adapter.info("Rank 0: Aggregating previous results and creating task pool...")
        # Load predictions recovered from the previous result file.
        prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
        if osp.exists(prev_file):
            all_prev_results.update(load(prev_file))

        pattern = osp.join(work_dir, f"*_{dataset_name}.pkl")
        for rank_out_file in glob.glob(pattern):
            # Skip the separately handled previous-result file.
            if rank_out_file.endswith("_PREV.pkl"):
                continue

            rank_data = load(rank_out_file)
            all_prev_results.update(rank_data)

        # Create the shared task pool if it does not already exist.
        if not osp.exists(task_pool_file):
            all_indices = list(dataset.data['index'])
            # Treat indices found in previous results as completed.
            remaining_indices = [idx for idx in all_indices if idx not in all_prev_results]
            task_pool = {
                'indices': remaining_indices,
                'next_task': 0,
                'finished_tasks': 0,
                'total_tasks': len(remaining_indices)
            }
            dump(task_pool, task_pool_file)
            log_adapter.info(f"Rank 0: Task pool created with {len(remaining_indices)} tasks.")

    if world_size > 1:
        # Broadcast completed indices so every rank can skip them.
        # Send only keys to reduce communication overhead.
        prev_keys = list(all_prev_results.keys())
        object_list = [prev_keys]
        dist.broadcast_object_list(object_list, src=0)
        prev_keys = set(object_list[0])
        dist.barrier()
    else:
        prev_keys = set(all_prev_results.keys())

    # Keep this rank's results in a local cache.
    # Restore its existing output file when resuming.
    res = {}
    if osp.exists(out_file):
        res = load(out_file)

    log_adapter.info(f"Loaded {len(res)} local results. Global finished: {len(prev_keys)}")

    # Initialize the model for this worker.
    kwargs = {}
    if model_name is not None and (
        'Llama-4' in model_name
        or 'Qwen2-VL' in model_name
        or 'Qwen2.5-VL' in model_name
    ):
        kwargs = {'use_vllm': use_vllm}

    ws_bak = os.environ.pop('WORLD_SIZE', None)
    model = supported_VLM[model_name](**kwargs) if isinstance(model, str) else model
    if ws_bak:
        os.environ['WORLD_SIZE'] = ws_bak

    is_api = getattr(model, 'is_api', False)
    if is_api:
        return None
    else:
        model.set_dump_image(dataset.dump_image)

    loop_start_time = time.time()
    processed_count = 0

    while True:
        task_idx = None
        # Claim one task while holding the shared task-pool lock.
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
        except (portalocker.exceptions.LockException, FileNotFoundError) as e:
            log_adapter.error(f"Rank {rank} failed to access task pool: {e}. Retrying...")
            time.sleep(rank * 0.1 + 0.1)
            continue

        if task_idx is None:
            break

        # Skip tasks already present in local or previous results.
        if task_idx in res or task_idx in prev_keys:
            continue

        data_item = dataset.data[dataset.data['index'] == task_idx].iloc[0]

        # Use the model-specific prompt when available.
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            struct = model.build_prompt(data_item, dataset=dataset_name)
        else:
            struct = dataset.build_prompt(data_item)


        # Generate a response with the requested visual budget.
        if os.environ.get('SKIP_ERR', '0') == '1':
            try:
                model.budget = budget
                response = model.generate(message=struct, dataset=dataset_name)
            except RuntimeError as err:
                torch.cuda.synchronize()
                warnings.warn(f'{type(err)} {str(err)}')
                # Use the same failure marker as the job summary.
                response = f'{FAIL_MSG}: {type(err)} {str(err)}'
        else:
            model.budget = budget
            response = model.generate(message=struct, dataset=dataset_name)
        torch.cuda.empty_cache()

        res[task_idx] = {'prediction': response}
        processed_count += 1
        # Update the shared completion counter under the lock.
        try:
            with portalocker.Lock(task_pool_file, 'r+', timeout=60) as f:
                task_pool = json.load(f)
                task_pool['finished_tasks'] += 1
                f.seek(0)
                f.truncate()
                json.dump(task_pool, f)
        except Exception as e:
            log_adapter.warning(f"Rank {rank} failed to update progress: {e}")

        if verbose:
            verbose_str = (
                f"================================================\n"
                f"Rank: {rank}, Task: {task_idx}\n"
                f"------------------------------------------------\n"
                f"PROMPT: {struct}\n"
                f"------------------------------------------------\n"
                f"RESPONSE: {response}\n"
                f"================================================"
            )
            print(verbose_str, flush=True)

        if processed_count > 0 and processed_count % 5 == 0:
            dump(res, out_file)

    dump(res, out_file)
    loop_end_time = time.time()
    log_adapter.info(f"Rank {rank}: Finished {processed_count} tasks. Time: {loop_end_time - loop_start_time:.2f}s.")

    if world_size > 1:
        dist.barrier()
    return model


def infer_data_job(
    model, work_dir, model_name, dataset, verbose=False, api_nproc=4, ignore_failed=False, use_vllm=False, budget=1
):
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name

    # Configure file logging once on rank 0.
    if rank == 0 and not getattr(logging.getLogger(), '_vlmeval_configured', False):
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        log_file = osp.join(work_dir, f"inference_log_{time.strftime('%Y%m%d-%H%M%S')}.log")
        formatter = logging.Formatter('%(asctime)s - RANK %(rank)s - %(levelname)s - %(message)s', defaults={'rank': 'N/A'})
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        setattr(root_logger, '_vlmeval_configured', True)

    if world_size > 1:
        dist.barrier()

    # Resolve output paths for predictions and resume state.
    result_file = get_pred_file_path(work_dir, model_name, dataset_name, use_env_format=True)
    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'

    # Normalize previous results into an index-to-prediction mapping.
    if osp.exists(result_file):
        if rank == 0:
            data = load(result_file)
            # Saved results may be a DataFrame or a dictionary.
            # Use sample indices to identify completed predictions.
            if isinstance(data, pd.DataFrame):
                # Reconstruct full responses when thinking was saved separately.
                if os.getenv('SPLIT_THINK', False) and 'thinking' in data.columns:
                    results = {}
                    for idx, pred, think in zip(
                        data['index'],
                        data['prediction'],
                        data['thinking']
                    ):
                        pred_str = str(pred)
                        think_str = str(think).strip()
                        if think_str:
                            full = f"<think>{think_str}</think> {pred_str}"
                        else:
                            full = pred_str
                        results[idx] = full
                else:
                    # Otherwise, reuse the prediction column directly.
                    results = {k: v for k, v in zip(data['index'], data['prediction'])}
            else:
                # Reuse dictionary results directly.
                results = data

            if not ignore_failed:
                results = {k: v for k, v in results.items() if FAIL_MSG not in str(v)}
            dump(results, prev_file)
        if world_size > 1:
            dist.barrier()

    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{dataset_name}.pkl')
    out_file = tmpl.format(rank)

    # Run inference using the shared task pool.
    model = infer_data_new(
        model=model, work_dir=work_dir, model_name=model_name, dataset=dataset,
        out_file=out_file, verbose=verbose, api_nproc=api_nproc, use_vllm=use_vllm, budget=budget)

    if world_size > 1:
        dist.barrier()

    # Merge worker outputs on rank 0.
    if rank == 0:
        data_all = {}  # Map sample indices to prediction values.

        for i in range(world_size):
            rank_file = tmpl.format(i)
            if osp.exists(rank_file):
                rank_data = load(rank_file)
                # Rank files may contain prediction dictionaries or legacy strings.
                for idx, item in rank_data.items():
                    # Unwrap prediction records while accepting legacy strings.
                    if isinstance(item, dict) and 'prediction' in item:
                        data_all[idx] = item['prediction']
                    else:
                        # Preserve legacy prediction values.
                        data_all[idx] = item

                # Remove the merged worker file.
                os.remove(rank_file)


        # Recover remaining worker files from interrupted runs.
        pattern_hist = osp.join(work_dir, f"*_{dataset_name}.pkl")
        for rank_out_file in glob.glob(pattern_hist):
            # Skip the separately handled previous-result file.
            if rank_out_file.endswith("_PREV.pkl"):
                continue
            # Current worker files have been removed; remaining files are from earlier runs.
            if not osp.exists(rank_out_file):
                continue

            try:
                rank_data = load(rank_out_file)
            except Exception as e:
                print(f"Failed to load historical rank file {rank_out_file}: {e}")
                continue

            for idx, item in rank_data.items():
                # Prefer current predictions; use historical results only for missing indices.
                if idx in data_all:
                    continue
                if isinstance(item, dict) and 'prediction' in item:
                    data_all[idx] = item['prediction']
                else:
                    data_all[idx] = item

            # Remove each historical file after merging it.
            os.remove(rank_out_file)


        prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
        if osp.exists(prev_file):
            try:
                prev_results = load(prev_file)
                for idx, val in prev_results.items():
                    # Keep current predictions when filling gaps from previous results.
                    if idx not in data_all:
                        data_all[idx] = val
            except Exception as e:
                print(f"Failed to load prev results from {prev_file}: {e}")

        # Remove the task pool after all workers finish.
        task_pool_file = osp.join(work_dir, f"{model_name}_{dataset_name}_task_pool.json")
        if osp.exists(task_pool_file):
            os.remove(task_pool_file)

        # Write predictions in the standard dataset result format.
        data = dataset.data

        # Optionally separate reasoning traces from final predictions.
        if os.getenv('SPLIT_THINK', False):
            prediction = [str(data_all.get(x, FAIL_MSG)) for x in data['index']]
            def split_thinking(s):
                if '</think>' in s:
                    splits = s.split('</think>')
                    prediction = splits[-1].strip()
                    if len(splits) == 2 and '<think>' in splits[0]:
                        thinking = splits[0].split('<think>')[1].strip()
                    else:
                        thinking = '</think>'.join(splits[:-1])
                        thinking += '</think>'
                        warnings.warn('Failed to parse thinking, multiple </think> tags or missing <think> tag.')
                else:
                    thinking = ''
                    prediction = s
                return (prediction, thinking)


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
                    # Unbalanced braces
                    return ans

                content = ans[content_start:i]
                return content

            split_func = model.split_thinking if hasattr(model, 'split_thinking') else split_thinking
            tups = [split_func(x) for x in prediction]
            tups = [(extract_boxed_content(x[0]), x[1]) for x in tups]
            data['prediction'] = [x[0] for x in tups]
            data['thinking'] = [x[1] for x in tups]
        else:
            data['prediction'] = [str(data_all.get(x, FAIL_MSG)) for x in data['index']]

        if 'image' in data:
            data.pop('image')

        dump(data, result_file)

        try:
            total = len(data)
            fail_count = 0
            if 'prediction' in data:
                for p in data['prediction']:
                    if FAIL_MSG in str(p):
                        fail_count += 1
            log_adapter = logging.LoggerAdapter(logging.getLogger(), {'rank': rank})
            log_adapter.info(
                f"{model_name} x {dataset_name}: {fail_count}/{total} samples FAILED (contain '{FAIL_MSG}')"
            )
            print(f"[SUMMARY] {model_name} x {dataset_name}: {fail_count}/{total} samples FAILED.")
        except Exception:
            pass

    if world_size > 1:
        dist.barrier()
    return model
