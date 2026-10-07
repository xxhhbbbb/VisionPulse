"""VisionPulse registrations for the bundled VLMEvalKit evaluation runner."""
from functools import partial
import os
from vlmeval.vlm.qwen3_vl import Qwen3VLChat
from vlmeval.vlm.internvl import InternVLChat

VISIONPULSE_QWEN3_VL_4B = os.getenv("VISIONPULSE_QWEN3_VL_4B", "Qwen/Qwen3-VL-4B-Thinking")
VISIONPULSE_INTERNVL35_4B = os.getenv("VISIONPULSE_INTERNVL35_4B", "OpenGVLab/InternVL3_5-4B")
VISIONPULSE_QWEN3_VL_8B = os.getenv("VISIONPULSE_QWEN3_VL_8B", "Qwen/Qwen3-VL-8B-Thinking")
VISIONPULSE_INTERNVL35_8B = os.getenv("VISIONPULSE_INTERNVL35_8B", "OpenGVLab/InternVL3_5-8B")

# Judge APIs remain available through vlmeval.api and the dataset evaluators.
api_models = {}

supported_VLM = {
    "Qwen3-VL-4B-Thinking-VisionPulse-5pct": partial(
        Qwen3VLChat,
        model_path=VISIONPULSE_QWEN3_VL_4B,
        use_custom_prompt=False,
        temperature=0.7,
        max_new_tokens=16384,
        use_visionpulse=True,
        visionpulse_anchor_layer=17,
        visionpulse_visual_mass_tau=0.1,
        visionpulse_budget_min=0.01,
        visionpulse_budget_max=0.05,
        min_pixels=1280 * 32 * 32,
        max_pixels=4096 * 32 * 32,
    ),
    "Qwen3-VL-4B-Thinking-VisionPulse-10pct": partial(
        Qwen3VLChat,
        model_path=VISIONPULSE_QWEN3_VL_4B,
        use_custom_prompt=False,
        temperature=0.7,
        max_new_tokens=16384,
        use_visionpulse=True,
        visionpulse_anchor_layer=17,
        visionpulse_visual_mass_tau=0.4,
        visionpulse_budget_min=0.05,
        visionpulse_budget_max=0.10,
        min_pixels=1280 * 32 * 32,
        max_pixels=4096 * 32 * 32,
    ),
    "Qwen3-VL-8B-Thinking-VisionPulse-5pct": partial(
        Qwen3VLChat,
        model_path=VISIONPULSE_QWEN3_VL_8B,
        use_custom_prompt=False,
        temperature=0.7,
        max_new_tokens=16384,
        use_visionpulse=True,
        visionpulse_anchor_layer=17,
        visionpulse_visual_mass_tau=0.1,
        visionpulse_budget_min=0.01,
        visionpulse_budget_max=0.05,
        min_pixels=1280 * 32 * 32,
        max_pixels=4096 * 32 * 32,
    ),
    "Qwen3-VL-8B-Thinking-VisionPulse-10pct": partial(
        Qwen3VLChat,
        model_path=VISIONPULSE_QWEN3_VL_8B,
        use_custom_prompt=False,
        temperature=0.7,
        max_new_tokens=16384,
        use_visionpulse=True,
        visionpulse_anchor_layer=17,
        visionpulse_visual_mass_tau=0.4,
        visionpulse_budget_min=0.05,
        visionpulse_budget_max=0.10,
        min_pixels=1280 * 32 * 32,
        max_pixels=4096 * 32 * 32,
    ),
    "InternVL3_5-4B-Thinking-VisionPulse-5pct": partial(
        InternVLChat, model_path=VISIONPULSE_INTERNVL35_4B, use_lmdeploy=False,
        use_visionpulse=True,
        visionpulse_anchor_layer=17,
        visionpulse_visual_mass_tau=0.1,
        visionpulse_budget_min=0.01,
        visionpulse_budget_max=0.05,
        max_new_tokens=16384, cot_prompt_version="r1", do_sample=True, version="V2.0"
    ),
    "InternVL3_5-8B-Thinking-VisionPulse-5pct": partial(
        InternVLChat, model_path=VISIONPULSE_INTERNVL35_8B, use_lmdeploy=False,
        use_visionpulse=True,
        visionpulse_anchor_layer=17,
        visionpulse_visual_mass_tau=0.1,
        visionpulse_budget_min=0.01,
        visionpulse_budget_max=0.05,
        max_new_tokens=16384, cot_prompt_version="r1", do_sample=True, version="V2.0"
    ),
}
