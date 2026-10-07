from __future__ import annotations

import logging
import os
import warnings

import torch

from ..base import BaseModel
from .prompt import Qwen3VLPromptMixin
from ...smp import get_gpu_memory


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


class Qwen3VLChat(Qwen3VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_new_tokens: int = 32768,
        top_p: float = 0.8,
        top_k: int = 20,
        temperature: float = 0.01,
        repetition_penalty: float = 1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,
        verbose: bool = False,
        use_visionpulse: bool = True,
        vision_attn_implementation: str = 'eager',
        visionpulse_anchor_layer: int | None = None,
        visionpulse_visual_mass_tau: float | None = None,
        visionpulse_budget_min: float | None = None,
        visionpulse_budget_max: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = 1.5
        self.temperature = temperature
        if self.total_pixels and self.total_pixels > 24576 * 32 * 32:
            print('The total number of video tokens might too large, resulting in an overly long input sequence.')
        self.generate_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process
        self.fps = kwargs.pop('fps', 2)
        self.nframe = kwargs.pop('nframe', 128)
        self.FRAME_FACTOR = 2

        self.budget = 1.0

        if not use_visionpulse or kwargs.pop('use_vllm', False):
            raise ValueError("This release supports VisionPulse with Transformers eager attention.")
        if kwargs:
            raise TypeError(f"Unexpected Qwen3VLChat options: {', '.join(sorted(kwargs))}")
        self.visionpulse_anchor_layer = visionpulse_anchor_layer
        self.visionpulse_visual_mass_tau = visionpulse_visual_mass_tau
        self.visionpulse_budget_min = visionpulse_budget_min
        self.visionpulse_budget_max = visionpulse_budget_max
        assert model_path is not None
        self.model_path = model_path
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_path)
        if hasattr(self.processor, 'tokenizer') and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = 'left'
        self.input_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_device_map = 'auto'

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0

        self.use_vllm = False
        from visionpulse.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype='auto', device_map=self.model_device_map,
            attn_implementation={'text_config': 'eager', 'vision_config': vision_attn_implementation},
        )
        self._apply_visionpulse_config()
        self.model.eval()

        torch.cuda.empty_cache()

    def _apply_visionpulse_config(self):
        if not hasattr(self, 'model') or self.model is None:
            return
        if self.visionpulse_anchor_layer is None and self.visionpulse_visual_mass_tau is None and self.visionpulse_budget_min is None and self.visionpulse_budget_max is None:
            return

        cfg_targets = []
        text_cfg = getattr(getattr(self.model, 'config', None), 'text_config', None)
        if text_cfg is not None:
            cfg_targets.append(text_cfg)
        for module in self.model.modules():
            if 'Qwen3VLTextAttention' in module.__class__.__name__ and hasattr(module, 'config'):
                cfg_targets.append(module.config)

        seen = set()
        unique_targets = []
        for cfg in cfg_targets:
            ident = id(cfg)
            if ident not in seen:
                seen.add(ident)
                unique_targets.append(cfg)

        for cfg in unique_targets:
            if self.visionpulse_anchor_layer is not None:
                cfg.visionpulse_anchor_layer = int(self.visionpulse_anchor_layer)
            if self.visionpulse_visual_mass_tau is not None:
                cfg.visionpulse_visual_mass_tau = float(self.visionpulse_visual_mass_tau)
            if self.visionpulse_budget_min is not None:
                cfg.visionpulse_budget_min = float(self.visionpulse_budget_min)
            if self.visionpulse_budget_max is not None:
                cfg.visionpulse_budget_max = float(self.visionpulse_budget_max)

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 32 * 32
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                for key in ['min_pixels', 'max_pixels', 'total_pixels', 'resized_height', 'resized_width']:
                    if key in s and s[key] is not None:
                        item[key] = s[key]
            elif s['type'] == 'video':
                value = s['value']
                if isinstance(value, list):
                    item = {
                        'type': 'video',
                        'video': [ensure_image_url(v) for v in value],
                    }
                else:
                    item = {'type': 'video', 'video': ensure_video_url(value)}
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                for key in ['resized_height', 'resized_width', 'fps', 'nframes', 'sample_fps']:
                    if key in s and s[key] is not None:
                        item[key] = s[key]
                if not isinstance(value, list):
                    if self.fps is not None and 'fps' not in item:
                        item['fps'] = self.fps
                    elif self.nframe is not None and 'nframes' not in item:
                        import cv2
                        video = cv2.VideoCapture(s['value'])
                        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        video.release()
                        if frame_count < self.nframe:
                            new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                            print(f"use {new_frame_count} for {s['value']}")
                            item['nframes'] = new_frame_count
                        else:
                            item['nframes'] = self.nframe
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def generate_inner_transformers(self, message, dataset=None):
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")
            raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})


        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        video_metadatas = None
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)

        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            do_resize=False,
            return_tensors='pt',
            **(video_kwargs or {}),
        )
        inputs = inputs.to(self.input_device)

        if self.budget != 1.0:
            self.model.budget = self.budget
            print("Using parameter: ", self.budget)
        generated_ids = self.model.generate(
            **inputs,
            **self.generate_kwargs,
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = out[0]

        if self.post_process:
            resp = response.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]

        return response

    def generate_inner(self, message, dataset=None):
        return self.generate_inner_transformers(message, dataset=dataset)
