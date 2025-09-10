## VLLM implementation for any model supported by VLLM
import os
from typing import List, Union

import torch
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

VLLM_MODELS = {
    "vllm-perception-lm-1b": {
        "path": "facebook/Perception-LM-1B",
        "image_size": 448,
        "max_video_frames": 32,
        "max_num_tiles": 36,
    },
    "vllm-perception-lm-3b": {
        "path": "facebook/Perception-LM-3B",
        "image_size": 448,
        "max_video_frames": 32,
        "max_num_tiles": 36,
    },
    "vllm-perception-lm-8b": {
        "path": "facebook/Perception-LM-8B",
        "image_size": 448,
        "max_video_frames": 32,
        "max_num_tiles": 36,
    },
}


class VLLMModel:
    video_mode = "direct"
    allows_image = True

    def __init__(
        self, model_name="vllm-perception-lm-8b", device="cuda", cache_dir=None
    ):
        assert model_name in VLLM_MODELS, f"Model {model_name} not found in VLLM_MODELS"
        self.model_name = VLLM_MODELS[model_name]["path"]
        self.device = device
        self.cache_dir = cache_dir
        self.model_info = VLLM_MODELS[model_name]
        self.load_model()

        self.tmp_files = []

    def load_model(self):
        self.model = LLM(model=self.model_name, gpu_memory_utilization=0.5)
        self.processor = AutoProcessor.from_pretrained(self.model_name, use_fast=True)

    def load_images(
        self, paths: List[str], num_tiles: int = 4
    ) -> List[Union[torch.Tensor, List[torch.Tensor]]]:
        processed_data = []
        for path in paths:
            if path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):  # Video file
                raise NotImplementedError("video loading not implemented yet")
            else:  # Regular image file
                image = Image.open(path).convert("RGB")
                processed_data.append(image)
        return processed_data

    def clear_temp_files(self):
        for file_name in self.tmp_files:
            try:
                os.remove(file_name)
            except OSError:
                pass
        self.tmp_files.clear()

    def forward(
        self,
        paths: List[str],
        texts: List[str],
        num_frames: int = 32,
        num_tiles: int = 36,
        question_template: str = 'Does this image show "{}"? Answer the question with Yes or No',
        answer_template: str = "Yes",
    ) -> torch.Tensor:
        assert len(paths) == len(texts), "Number of paths and texts must match"

        questions = [question_template.format(text) for text in texts]
        answers = [answer_template]

        yes_token_id = self.processor.tokenizer.encode("Yes")[-1]

        processed_data = self.load_images(paths, num_tiles)

        lm_probs = []
        for data, question, answer, path in zip(
            processed_data, questions, answers, paths
        ):
            media_type = "image"
            # Create prompt
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": path,
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(
                [conversation],
                add_generation_prompt=True,
            )[0]

            sampling_params = SamplingParams(
                temperature=0.2,
                max_tokens=1,
                stop_token_ids=[self.processor.tokenizer.eos_token_id],
                allowed_token_ids=[yes_token_id],
                logprobs=1,
            )

            outputs = self.model.generate(
                [
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"image": data},
                    }
                ],
                sampling_params=sampling_params,
            )

            lm_prob = torch.exp(
                torch.tensor(outputs[0].outputs[0].logprobs[0][yes_token_id].logprob)
            ).item()
            lm_probs.append(lm_prob)

        self.clear_temp_files()
        return torch.tensor(lm_probs)

    def generate(
        self,
        images: List[str],
        texts: List[str],
        num_frames: int = 32,
        num_tiles: int = 36,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        max_new_tokens: int = 256,
    ) -> List[str]:
        assert len(images) == len(texts), "Number of paths and texts must match"

        processed_data = self.load_images(images, num_tiles)
        generated_texts = []

        for data, prompt, path in zip(processed_data, texts, images):
            media_type = "image"
            # Create prompt
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": media_type,
                            "image": path,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            inp_prompt = self.processor.apply_chat_template(
                [conversation],
                add_generation_prompt=True,
            )[0]

            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=256,
                stop_token_ids=[self.processor.tokenizer.eos_token_id],
            )

            outputs = self.model.generate(
                [
                    {
                        "prompt": inp_prompt,
                        "multi_modal_data": {"image": data},
                    }
                ],
                sampling_params=sampling_params,
            )
            generated_texts.append(outputs[0].outputs[0].text)

        return generated_texts

    def batch_generate(
        self,
        media_paths: List[str],
        questions: List[str],
        media_types: List[str],
        num_frames: int = 32,
        num_tiles: int = 36,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        max_new_tokens: int = 256,
    ) -> List[str]:
        """
        Batch generation for multiple media files and questions
        """
        assert len(media_paths) == len(questions) == len(media_types), (
            "Inputs must have the same length"
        )

        processed_data = self.load_images(media_paths, num_tiles)

        prompts = []
        for data, path, question, media_type in zip(
            processed_data, media_paths, questions, media_types
        ):
            assert media_type == "image"
            # Create prompt
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": path,
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ]
            inp_prompt = self.processor.apply_chat_template(
                [conversation],
                add_generation_prompt=True,
            )[0]
            prompts.append(
                {
                    "prompt": inp_prompt,
                    "multi_modal_data": {"image": data},
                }
            )

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=256,
            stop_token_ids=[self.processor.tokenizer.eos_token_id],
        )
        outputs = self.model.generate(prompts, sampling_params=sampling_params)
        generation = []
        for o in outputs:
            generated_text = o.outputs[0].text
            generation.append(generated_text)

        return generation
