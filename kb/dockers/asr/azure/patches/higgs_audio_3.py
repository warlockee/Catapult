# SPDX-License-Identifier: Apache-2.0
# ruff: noqa
"""Inference-only audio model compatible with HuggingFace weights."""
from functools import lru_cache
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoFeatureExtractor
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer
from vllm.multimodal import MULTIMODAL_REGISTRY

from .higgs_audio_config import HiggsAudio3Config
from .higgs_audio_tokenizer import AudioTokenizer
from .higgs_audio import (
    HiggsAudioMultiModalProcessor,
    HiggsAudioDummyInputsBuilder,
    HiggsAudioInputs,
    HiggsAudioMultiModalProcessor,
    HiggsAudioProcessingInfo,
    HiggsAudioDummyInputsBuilder,
    HFHiggsAudioProcessor,
    HiggsAudioForConditionalGeneration,
    get_processor,
    _validate_and_reshape_mm_tensor,
)


logger = init_logger(__name__)

AutoFeatureExtractor.register(HiggsAudio3Config, AudioTokenizer)


def _ceil_to_nearest(n, round_to):
    if round_to <= 1: return n  # avoid division by zero
    return (n + round_to - 1) // round_to * round_to

def _fix_and_round_time_dim(
    audio_features: torch.Tensor,                 # (B, C, T_feat)
    attention_mask: torch.Tensor | None,         # (B, T_mask) or None
    round_to: int = 4,
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Make feature & mask time dims match and pad to a multiple of `round_to`.
    Also resolves Whisper's 'longest' oddity where T_mask = T_feat +/- 1.

    Returns: (features_fixed, mask_fixed_or_none)
    """
    assert audio_features.dim() == 3, "audio_features must be (B, C, T)"
    B, C, T_feat = audio_features.shape
    dev = audio_features.device
    dtype_feat = audio_features.dtype
    if attention_mask is not None:
        assert attention_mask.dim() == 2, "attention_mask must be (B, T)"
        assert attention_mask.size(0) == B, "batch size mismatch"
        T_mask = attention_mask.size(1)
    else:
        T_mask = T_feat
    # 1) First, equalize T between features and mask (crop/pad conservatively).
    T = max(T_feat, T_mask)
    if attention_mask is None:
        with torch.no_grad():
            col_nonpad = (audio_features != pad_value).any(dim=1)  # (B, T_feat)
            attention_mask = col_nonpad.to(dtype=torch.int32)
    # Now reconcile off-by-one cases first (common Whisper bug).
    if T_mask == T_feat + 1:
        if (attention_mask[:, -1] == 0).all():
            attention_mask = attention_mask[:, :-1]
            T_mask -= 1
        else:
            audio_features = F.pad(audio_features, (0, 1), value=pad_value)
            T_feat += 1
    elif T_feat == T_mask + 1:
        if (audio_features[:, :, -1] == pad_value).all():
            audio_features = audio_features[:, :, :-1]
            T_feat -= 1
        else:
            attention_mask = F.pad(attention_mask, (0, 1), value=0)
            T_mask += 1
    T = max(T_feat, T_mask)
    if T_feat < T:
        audio_features = F.pad(audio_features, (0, T - T_feat), value=pad_value)
        T_feat = T
    if T_mask < T:
        attention_mask = F.pad(attention_mask, (0, T - T_mask), value=0)
        T_mask = T
    # 2) Round up to multiple of `round_to`
    if round_to is not None and round_to > 1:
        T_rounded = _ceil_to_nearest(T, round_to)
        if T_rounded != T:
            audio_features = F.pad(audio_features, (0, T_rounded - T), value=pad_value)
            attention_mask = F.pad(attention_mask, (0, T_rounded - T), value=0)
            T = T_rounded
    audio_features = audio_features.to(dtype=dtype_feat, device=dev)
    attention_mask = attention_mask.to(dtype=torch.int32, device=dev)
    assert audio_features.size(-1) == attention_mask.size(-1), "time dims should now match"
    return audio_features, attention_mask


class HFHiggsAudio3Processor(HFHiggsAudioProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feat_whisper_padding = "longest"

    @property
    def default_chat_template(self):
        # fmt: off
        return (
            "{% set loop_messages = messages %}"
            "{% for message in loop_messages %}"
                "{% set content = '<|im_start|>' + message['role'] + "
                "'\n\n' + message['content'] | trim + '<|im_end|>' %}"
            "{{ content }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
                "{{ '<|im_start|>assistant\n\n' }}"
            "{% endif %}"
        )
        # fmt: on


cached_get_processor = lru_cache(get_processor(HFHiggsAudio3Processor))

class HiggsAudio3ProcessingInfo(HiggsAudioProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(HiggsAudio3Config)

    def get_hf_processor(
        self,
        *,
        # Ignored in initialization
        sampling_rate: Optional[int] = None,
        **kwargs: object,
    ) -> HFHiggsAudio3Processor:
        hf_config = self.get_hf_config()
        supports_audio_out = bool(getattr(hf_config, "skip_audio_tower", False))
        return cached_get_processor(
            self.ctx.tokenizer,
            audio_stream_bos_id=hf_config.audio_stream_bos_id,
            audio_stream_eos_id=hf_config.audio_stream_eos_id,
            # Prefer audio tokenizer for input on TTS generation models
            prefer_audio_tokenizer_for_input=supports_audio_out,
            is_audio_out_model=supports_audio_out,
        )


@MULTIMODAL_REGISTRY.register_processor(
    HiggsAudioMultiModalProcessor,
    info=HiggsAudio3ProcessingInfo,
    dummy_inputs=HiggsAudioDummyInputsBuilder)
@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is typically 1D (seq_len,) but may be 2D when using M-RoPE,
        # so mark the last dimension as dynamic to cover both layouts.
        "positions": -1,
        "intermediate_tensors": 0,
        # inputs_embeds has shape (seq_len, hidden_size); make seq_len dynamic.
        "inputs_embeds": 0,
    },
    # Disable Dynamo/Inductor for this model to avoid shape-specialization
    # pitfalls during cudagraph capture on short warmup batches.
    # We still benefit from CUDA Graph capture at the runner level.
    enable_if=lambda _cfg: False,
)
class HiggsAudio3ForConditionalGeneration(HiggsAudioForConditionalGeneration, SupportsMultiModal):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix, llm_backbone=Qwen3DecoderLayer)

    def _parse_and_validate_audio_input(
            self, **kwargs: object) -> Optional[HiggsAudioInputs]:
        audio_features = kwargs.pop("audio_features", None)
        audio_feature_attention_mask = kwargs.pop(
            "audio_feature_attention_mask", None)
        audio_out_ids = kwargs.pop("audio_out_ids", None)
        if audio_features is None and audio_out_ids is None:
            return None
        if audio_features is not None:
            # FIXME: revisit the padding change
            """
            When you have multiple audios in a list,
            it pads them ALL to the max size in that batch.
            So the SAME audio gets different padding depending
            on what OTHER audios are batched with it.
            Then in _parse_and_validate_audio_input:
                - audio_features: uses pad_with=None for Whisper → NO padding
                - audio_feature_attention_mask: uses pad_with=0 → ALWAYS pads to batch max
            This creates a mismatch -
            The same audio arrives with different mask lengths depending on batch composition.
            """
            audio_features = _validate_and_reshape_mm_tensor(
                audio_features,
                "audio_features",
                pad_with=0) # FIXME: if not self.use_whisper_tokenizer else None)
            audio_feature_attention_mask = _validate_and_reshape_mm_tensor(
                audio_feature_attention_mask,
                "audio_feature_attention_mask",
                pad_with=0,
            )
            if not isinstance(audio_features, (torch.Tensor, list)):
                raise ValueError("Incorrect type of audio input features. "
                                 f"Got type: {type(audio_features)}")
        if audio_out_ids is not None:
            audio_out_ids = _validate_and_reshape_mm_tensor(
                audio_out_ids, "audio_out_ids")
            # audio_out_ids_length = _validate_and_reshape_mm_tensor(
            #     audio_out_ids_length, "audio_out_ids_length")
        return HiggsAudioInputs(
            audio_features=audio_features,
            audio_feature_attention_mask=audio_feature_attention_mask,
            audio_out_ids=audio_out_ids,
        )

    def _process_whisper_audio_input(
        self, audio_features: torch.Tensor,
        audio_feature_attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        expected_seq_length = (self.audio_tower.config.max_source_positions *
                               self.audio_tower.conv1.stride[0] *
                               self.audio_tower.conv2.stride[0])

        current_seq_length = audio_features.shape[-1]
        if current_seq_length != expected_seq_length:
            if current_seq_length > expected_seq_length:
                audio_features = audio_features[..., :expected_seq_length]
                audio_feature_attention_mask = audio_feature_attention_mask[
                    ..., :expected_seq_length]
            # TODO: double check if we need to pad or not
            # else:
            #     pad_amount = expected_seq_length - current_seq_length
            #     audio_features = nn.functional.pad(audio_features,
            #                                        (0, pad_amount))
            #     audio_feature_attention_mask = nn.functional.pad(
            #         audio_feature_attention_mask, (0, pad_amount))

        # Guard: ensure the attention mask matches the (possibly adjusted)
        # feature length even if the features already had expected length.
        # We have observed cases where mask length == expected_len + 1.
        feat_len = audio_features.shape[-1]
        mask_len = audio_feature_attention_mask.shape[-1]
        if mask_len != feat_len:
            if mask_len > feat_len:
                audio_feature_attention_mask = (
                    audio_feature_attention_mask[..., :feat_len]
                )
            else:
                pad_amount = feat_len - mask_len
                audio_feature_attention_mask = nn.functional.pad(
                    audio_feature_attention_mask, (0, pad_amount)
                )

        (
            audio_feat_lengths,
            audio_feat_out_lengths,
        ) = self.audio_tower._get_feat_extract_output_lengths(
            audio_feature_attention_mask.sum(-1))

        try:
            logger.debug(
                "HiggsAudio _process_whisper_audio_input: mel_len=%s, out_len=%s, "
                "features_shape=%s, mask_shape=%s",
                audio_feature_attention_mask.sum(-1).tolist(),
                audio_feat_out_lengths.view(-1).tolist(),
                tuple(audio_features.shape),
                tuple(audio_feature_attention_mask.shape),
            )
        except Exception:
            pass

        batch_size, _, max_mel_seq_len = audio_features.shape
        max_seq_len = (max_mel_seq_len - 2) // 2 + 1
        # Create a sequence tensor of shape (batch_size, max_seq_len)
        seq_range = (torch.arange(
            0,
            max_seq_len,
            dtype=audio_feat_lengths.dtype,
            device=audio_feat_lengths.device).unsqueeze(0).expand(
                batch_size, max_seq_len))
        lengths_expand = audio_feat_lengths.unsqueeze(-1).expand(
            batch_size, max_seq_len)
        # Create mask
        padding_mask = seq_range >= lengths_expand

        audio_attention_mask_ = padding_mask.view(
            batch_size, 1, 1, max_seq_len).expand(batch_size, 1, max_seq_len,
                                                  max_seq_len)
        audio_attention_mask = audio_attention_mask_.to(
            dtype=self.audio_tower.conv1.weight.dtype,
            device=self.audio_tower.conv1.weight.device)
        audio_attention_mask[audio_attention_mask_] = float("-inf")

        audio_outputs = self.audio_tower(audio_features,
                                         attention_mask=audio_attention_mask,
                                         check_seq_length=False)
        selected_audio_feature = audio_outputs.last_hidden_state
        audio_features = self.audio_encoder_proj(selected_audio_feature)

        num_audios, max_audio_tokens, embed_dim = audio_features.shape
        audio_feat_out_lengths = audio_feat_out_lengths.unsqueeze(1)
        if self.config.projector_temporal_downsample > 1:
            # ((audio_feat_out_lengths + 2*pad - (ksz - 1) - 1) // proj_stride) + 1
            audio_feat_out_lengths = (audio_feat_out_lengths - 1) // self.config.projector_temporal_downsample + 1
        # Ensure at least 1 token per audio segment to prevent crash in scatter_mm_placeholders
        # when very short audio clips (1-2 mel frames) downsample to 0 tokens
        audio_feat_out_lengths = audio_feat_out_lengths.clamp(min=1, max=audio_features.size(1))
        audio_features_mask = torch.arange(max_audio_tokens).expand(
            num_audios, max_audio_tokens).to(
                audio_feat_out_lengths.device) < audio_feat_out_lengths
        masked_audio_features = audio_features[audio_features_mask].view(
            -1, embed_dim)

        # Split to tuple of embeddings for individual audio input.
        return torch.split(masked_audio_features,
                           audio_feat_out_lengths.flatten().tolist())

    def _process_audio_input(self,
                             audio_input: HiggsAudioInputs) -> torch.Tensor:
        assert self.use_whisper_tokenizer, "HiggsAudio3ForConditionalGeneration only supports Whisper tokenizer"
        audio_features = audio_input["audio_features"]
        audio_feature_attention_mask = audio_input[
            "audio_feature_attention_mask"]

        audio_features, audio_feature_attention_mask = _fix_and_round_time_dim(
            audio_features,
            audio_feature_attention_mask,
            round_to=16,  # FIXME - hard code to 16 to avoid SP issue later
            pad_value=0.0,
        )
        return self._process_whisper_audio_input(
            audio_features, audio_feature_attention_mask)
