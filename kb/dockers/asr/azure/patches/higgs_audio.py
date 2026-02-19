# SPDX-License-Identifier: Apache-2.0
# ruff: noqa
"""Inference-only audio model compatible with HuggingFace weights."""
import copy
import math
import os
import warnings
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from typing import Any, List, Optional, Set, Tuple, TypedDict, Union

import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import (AutoConfig, AutoFeatureExtractor, BatchFeature,
                          ProcessorMixin)
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper import WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer
from transformers.tokenization_utils_base import (PaddingStrategy,
                                                  PreTokenizedInput, TextInput)

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.llama import (LlamaAttention,
                                              LlamaDecoderLayer, LlamaMLP)
from vllm.model_executor.models.utils import (extract_layer_index,
                                              is_pp_missing_parameter,
                                              make_layers,
                                              merge_multimodal_embeddings)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems, NestedTensors
from vllm.multimodal.parse import (AudioProcessorItems, MultiModalDataItems,
                                   MultiModalDataParser)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate, PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors
from vllm.v1.multimodal.metadata import MultimodalMetadata
from vllm.v1.sample.metadata import SamplingMetadata

from .higgs_audio_config import HiggsAudioConfig, HiggsAudioEncoderConfig, HiggsAudio3Config
from .higgs_audio_tokenizer import AudioTokenizer

logger = init_logger(__name__)

_KEYS_TO_MODIFY_MAPPING = {
    "audio_decoder_proj.audio_lm_head": "audio_lm_head",
    "audio_decoder_proj.text_lm_head": "text_lm_head",
    # Backward-compat: some checkpoints may use transformer_layers as the
    # collection name inside audio_decoder_proj. Normalize to .layers.
    "audio_decoder_proj.transformer_layers": "audio_decoder_proj.layers",
}

AutoConfig.register("higgs_audio_encoder", HiggsAudioEncoderConfig)
AutoConfig.register("higgs_audio", HiggsAudioConfig)
AutoConfig.register("higgs_audio_3", HiggsAudio3Config)
AutoFeatureExtractor.register(HiggsAudioConfig, AudioTokenizer)
AutoFeatureExtractor.register(HiggsAudio3Config, AudioTokenizer)
if transformers.__version__.startswith("4.46"):
    transformers._modules.add("AudioTokenizer")
    transformers.AudioTokenizer = AudioTokenizer


# # === Audio Inputs === #
class HiggsAudioInputs(TypedDict):
    # (num_audios, num_mel_bins, 3000)`
    audio_features: torch.Tensor

    # (num_audios, 3000)
    audio_feature_attention_mask: torch.Tensor

    # (num_audios, num_codebooks)
    audio_out_ids: torch.Tensor


def _validate_and_reshape_mm_tensor(
    mm_input: object,
    name: str,
    pad_with: Optional[int] = None,
) -> torch.Tensor:
    if not isinstance(mm_input, (torch.Tensor, list)):
        raise ValueError(f"Incorrect type of {name}. "
                         f"Got type: {type(mm_input)}")
    if isinstance(mm_input, torch.Tensor):
        return torch.concat(list(mm_input))
    else:
        if pad_with is not None:
            max_size = max([tensor.size(-1) for tensor in mm_input
                            ])  # Find max size along the last dimension
            # Step 2: Pad each tensor to the max size along the last
            # dimension
            padded_tensors = []
            for tensor in mm_input:
                pad_size = max_size - tensor.size(
                    -1)  # Calculate how much padding is needed
                if pad_size > 0:
                    # Pad tensor along the last dimension (right side)
                    padded_tensor = torch.nn.functional.pad(
                        tensor, (0, pad_size))
                else:
                    padded_tensor = tensor
                padded_tensors.append(padded_tensor)
            return torch.concat(padded_tensors)
        else:
            return torch.concat(mm_input)


def _build_delay_pattern_mask(
    input_ids: torch.LongTensor,
    bos_token_id: int,
    pad_token_id: int,
):
    """Implement the delay pattern proposed in "Simple and Controllable Music Generation", https://arxiv.org/pdf/2306.05284

    In the delay pattern, each codebook is offset by the previous codebook by
    one. We insert a special delay token at the start of the sequence if its delayed, and append pad token once the sequence finishes.

    Take the example where there are 4 codebooks and audio sequence length=5. After shifting, the output should have length seq_len + num_codebooks - 1

    - [ *,  *,  *,  *,  *,  P,  P,  P]
    - [ B,  *,  *,  *,  *,  *,  P,  P]
    - [ B,  B,  *,  *,  *,  *,  *,  P]
    - [ B,  B,  B,  *,  *,  *,  *,  *]

    where B indicates the delay token id, P is the special padding token id and `*` indicates that the original audio token.

    Now let's consider the case where we have a sequence of audio tokens to condition on.
    The audio tokens were originally in the following non-delayed form:

    - [a, b]
    - [c, d]
    - [e, f]
    - [g, h]

    After conversion, we get the following delayed form:
    - [a, b, -1, -1, -1]
    - [B, c,  d, -1, -1]
    - [B, B,  e,  f, -1]
    - [B, B,  B,  g,  h]

    Note that we have a special token `-1` that indicates it should be replaced by a new token we see in the generation phase.
    In that case, we should override the `-1` tokens in auto-regressive generation.

    Args:
        input_ids (:obj:`torch.LongTensor`):
            The input ids of the prompt. It will have shape (bsz, num_codebooks, seq_len).
        bos_token_id (:obj:`int`):
            The id of the special delay token
        pad_token_id (:obj:`int`):
            The id of the padding token. Should be the same as eos_token_id.

    Returns:
        input_ids (:obj:`torch.LongTensor`):
            The transformed input ids with delay pattern applied. It will have shape (bsz, num_codebooks, seq_len + num_codebooks - 1).
        input_ids_with_gen_mask (:obj:`torch.LongTensor`):
            The transformed input ids with delay pattern applied. The -1 in the output indicates new tokens that should be generated.

    """
    bsz, num_codebooks, seq_len = input_ids.shape

    new_seq_len = seq_len + num_codebooks - 1
    input_ids_with_gen_mask = torch.ones((bsz, num_codebooks, new_seq_len),
                                         dtype=torch.long,
                                         device=input_ids.device)
    bos_mask = torch.tril(input_ids_with_gen_mask, -1) > 0
    eos_mask = torch.triu(input_ids_with_gen_mask, seq_len) > 0
    input_ids_with_gen_mask[bos_mask] = bos_token_id
    input_ids_with_gen_mask[(~bos_mask) & (~eos_mask)] = input_ids.reshape(-1)
    input_ids = input_ids_with_gen_mask.clone()
    input_ids[eos_mask] = pad_token_id
    input_ids_with_gen_mask[eos_mask] = -1
    return input_ids


# Revised on top of transformers.models.qwen2_audio.modeling_qwen2_audio
# with Qwen2AudioEncoder --> HiggsAudioEncoder
# The code was originally borrowed from WhisperEncoder
class HiggsAudioEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* 
    self attention layers. Each layer is a [`WhisperEncoderLayer`].

    Args:
        config: HiggsAudioEncoderConfig
    """

    # Ignore copy
    config_class = HiggsAudioEncoderConfig
    main_input_name = "input_features"
    _no_split_modules = ["WhisperEncoderLayer"]

    def __init__(self, config: HiggsAudioEncoderConfig):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(
            embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins,
                               embed_dim,
                               kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv1d(embed_dim,
                               embed_dim,
                               kernel_size=3,
                               stride=2,
                               padding=1)

        self.embed_positions = nn.Embedding(self.max_source_positions,
                                            embed_dim)
        self.embed_positions.requires_grad_(False)

        self.layers = nn.ModuleList([
            WhisperEncoderLayer(config) for _ in range(config.encoder_layers)
        ])
        self.layer_norm = nn.LayerNorm(config.d_model)
        # Ignore copy
        self.avg_pooler = nn.AvgPool1d(2, stride=2)

        self.gradient_checkpointing = False

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        check_seq_length=True,
    ):
        r"""
        Args:
            input_features (`torch.LongTensor` of shape 
                `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech 
                waveform. Raw speech waveform can be obtained by loading a 
                `.flac` or `.wav` audio file into an array of type 
                `List[float]` or a `numpy.ndarray`, *e.g.* via the soundfile 
                library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used 
                for extracting the mel features, padding and conversion into a 
                tensor of type `torch.FloatTensor`. See 
                [`~WhisperFeatureExtractor.__call__`]
            attention_mask (`torch.Tensor`)`, *optional*):
                HiggsAudio does not support masking of the `input_features`, 
                this argument is preserved for compatibility, but it is not 
                used. By default the silence in the input log mel spectrogram 
                are ignored.
            head_mask (`torch.Tensor` of shape 
                `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. 
                Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all 
                attention layers. See `attentions` under returned tensors 
                for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. 
                See `hidden_states` under returned tensors for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of 
                a plain tuple.
        """

        expected_seq_length = (self.config.max_source_positions *
                               self.conv1.stride[0] * self.conv2.stride[0])
        if check_seq_length and (input_features.shape[-1]
                                 != expected_seq_length):
            raise ValueError(
                f"HiggsAudio expects the mel input features to be of length "
                f"{expected_seq_length}, but found {input_features.shape[-1]}. "
                "Make sure to pad the input mel features to "
                f"{expected_seq_length}.")

        output_attentions = (output_attentions if output_attentions is not None
                             else self.config.output_attentions)
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        return_dict = (return_dict if return_dict is not None else
                       self.config.use_return_dict)

        # Ignore copy
        input_features = input_features.to(dtype=self.conv1.weight.dtype,
                                           device=self.conv1.weight.device)

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)

        T = inputs_embeds.size(1)
        assert T <= self.config.max_source_positions, f"HiggsAudio expects the mel input features to be of length {self.config.max_source_positions}, but found {T}. Make sure to pad the input mel features to {self.config.max_source_positions}."
        embed_pos = self.embed_positions.weight[:T, :].unsqueeze(0)  # (1, T, D)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states,
                                              p=self.dropout,
                                              training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} " \
               f"layers, but it is for {head_mask.size()[0]}."

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states, )
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            # Ignore copy
            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx]
                                         if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1], )

        # Ignore copy
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.avg_pooler(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states, )

        if not return_dict:
            return tuple(
                v for v in [hidden_states, encoder_states, all_attentions]
                if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_states,
                               hidden_states=encoder_states,
                               attentions=all_attentions)

    # Ignore copy
    def _get_feat_extract_output_lengths(self,
                                         input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and 
        the output length of the audio encoder
        """
        # TODO(sxjscience) Double confirm the formula
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths


class HiggsAudioFeatureProjector(nn.Module):
    """
    Projector that maps audio features extracted by Whisper to hidden state of the text model. Two selectable implementations:
      - 'linear' (backward-compatible, old behavior)
      - 'mlp'    (new: optional temporal downsample + 2-layer MLP + activation)
    """
    def __init__(self, vllm_config: VllmConfig):
        super().__init__()
        config = vllm_config.model_config.hf_config
        audio_dim = config.audio_encoder_config.d_model
        llm_hidden_dim = config.text_config.hidden_size
        self.stride = int(getattr(config, "projector_temporal_downsample", 1))
        self.projector_type = getattr(config, "projector_type", "linear")
        if self.projector_type == "linear":
            assert self.stride == 1, "Temporal downsample is not supported for linear projector."
            self.linear = nn.Linear(audio_dim, llm_hidden_dim, bias=True)
        else: 
            if self.stride > 1:
                ksz, pad = 3, 1
                self.temporal = nn.Conv1d(audio_dim, audio_dim, ksz, self.stride, padding=pad, groups=audio_dim, bias=True)
            else: 
                self.temporal = nn.Identity()
            # fix to 2 layer mlp with 2048 hidden and ReLU as https://huggingface.co/stepfun-ai/Step-Audio-2-mini/blob/main/modeling_step_audio_2.py#L266
            hidden = 2048
            self.linear1 = nn.Linear(audio_dim, hidden, bias=True)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(hidden, llm_hidden_dim, bias=True)

    def forward(self, audio_features):
        # Input: (B, T, audio_dim)
        # Output: (B, T', llm_hidden_dim), where T' depends on the downsample stride get from conv
        x = audio_features  # (B, T, C_in)
        if self.projector_type == "linear":
            return self.linear(x)
        else: 
            if self.stride > 1:
                x = x.permute(0, 2, 1)
                x = self.temporal(x)  # apply on the time dimension
                x = x.permute(0, 2, 1)
            x = self.linear1(x)
            x = self.relu(x)
            return self.linear2(x)


class HiggsAudioDecoderProjector(nn.Module):
    """A decoder-only projector applied on top of LLM hidden states.

    Uses vLLM's LlamaDecoderLayer to remain consistent with the main stack.
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self._audio_decoder_proj_num_layers = config.audio_decoder_proj_num_layers
        if self._audio_decoder_proj_num_layers > 0:
            self.layers = nn.ModuleList([
                LlamaDecoderLayer(
                    config.text_config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"audio_decoder_proj.layers.{i}",
                ) for i in range(self._audio_decoder_proj_num_layers)
            ])
            self.norm = RMSNorm(config.text_config.hidden_size,
                                eps=config.text_config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, *,
                positions: torch.Tensor) -> torch.Tensor:
        if self._audio_decoder_proj_num_layers <= 0:
            return hidden_states

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states)

        return hidden_states


def get_processor(cls: type[ProcessorMixin]):
    def _get_processor(
        tokenzier,
        *args,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """Gets a processor for the given model name via HuggingFace.

        Derived from `vllm.transformers_utils.image_processor.get_image_processor`.
        """
        # don't put this import at the top level
        # it will call torch.cuda.device_count()
        from transformers import AutoFeatureExtractor

        # For INPUT audio processing, distinguish between ASR and TTS voice reference.
        # 0.8.3 parity: input feature extractor is controlled by HIGGS_AUDIO_TOKENIZER
        # (default: Whisper). Allow HIGGS_AUDIO_INPUT_TOKENIZER to override when set.
        prefer_audio_tokenizer_for_input = bool(
            kwargs.pop("prefer_audio_tokenizer_for_input", False))
        is_audio_out_model = bool(kwargs.pop("is_audio_out_model", False))

        env_input_tok = os.getenv("HIGGS_AUDIO_INPUT_TOKENIZER")
        env_tok = os.getenv("HIGGS_AUDIO_TOKENIZER")

        audio_stream_bos_id = kwargs.pop("audio_stream_bos_id", None)
        audio_stream_eos_id = kwargs.pop("audio_stream_eos_id", None)

        # Decide default based on model capability (0.8.3 parity):
        # - If env explicitly requests Whisper, use Whisper.
        # - Else if env specifies an audio tokenizer type, use AudioTokenizer.
        # - Else if prefer_audio_tokenizer_for_input is True (TTS generation
        #   models without an audio tower), default to AudioTokenizer.
        # - Otherwise default to Whisper for ASR-style inputs.
        input_tok_name = env_input_tok or env_tok
        use_whisper = False
        if input_tok_name is None:
            if prefer_audio_tokenizer_for_input:
                # Conservative default for Higgs TTS models trained with 4-CB
                input_tok_name = "xcodec_0521_exp_6"
            else:
                input_tok_name = "openai/whisper-large-v3-turbo"

        if input_tok_name == "openai/whisper-large-v3-turbo":
            use_whisper = True

        if use_whisper:
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                input_tok_name,
                *args,
                trust_remote_code=trust_remote_code,
                attn_implementation="sdpa",
                **kwargs,
            )
            try:
                logger.info("HFHiggsAudioProcessor input extractor: Whisper (%s)",
                            input_tok_name)
            except Exception:
                pass
        else:
            HIGGS_AUDIO_TOKENIZER_PATH = os.environ.get(
                "HIGGS_AUDIO_TOKENIZER_PATH", None)
            feature_extractor = AudioTokenizer(
                model=input_tok_name,
                device="cpu",
                downloaded_model_path=HIGGS_AUDIO_TOKENIZER_PATH,
            )
            try:
                logger.info(
                    "HFHiggsAudioProcessor input extractor: AudioTokenizer (%s)",
                    input_tok_name)
            except Exception:
                pass
        processor = cls(
            feature_extractor=feature_extractor,
            tokenizer=tokenzier,
            audio_stream_bos_id=audio_stream_bos_id,
            audio_stream_eos_id=audio_stream_eos_id,
            is_audio_out_model=is_audio_out_model
        )
        logger.info(f"Loaded {cls.__name__}")

        return processor

    return _get_processor


def _get_feat_extract_output_lengths(input_lengths: torch.LongTensor, projector_temporal_downsample: int):
    """
    Computes the output length of the convolutional layers
    and the output length of the audio encoder
    """
    input_lengths = (input_lengths - 1) // 2 + 1
    output_lengths = (input_lengths - 2) // 2 + 1
    if projector_temporal_downsample > 1:
        output_lengths = (output_lengths - 1) // projector_temporal_downsample + 1
    return input_lengths, output_lengths


class HFHiggsAudioProcessor(ProcessorMixin):
    """
    HF Processor class for Higgs audio model. Mostly borrow from 
    processing_qwen2_audio.py.
    """

    attributes = ["feature_extractor", "tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        audio_token="<|AUDIO|>",
        audio_bos_token="<|audio_bos|>",
        audio_eos_token="<|audio_eos|>",
        audio_stream_bos_id=None,
        audio_stream_eos_id=None,
        is_audio_out_model=False,
    ):
        self.is_audio_out_model = is_audio_out_model
        if chat_template is None:
            chat_template = self.default_chat_template
        self.audio_token = (tokenizer.audio_token if hasattr(
            tokenizer, "audio_token") else audio_token)
        self.audio_bos_token = (tokenizer.audio_bos_token if hasattr(
            tokenizer, "audio_bos_token") else audio_bos_token)
        self.audio_eos_token = (tokenizer.audio_eos_token if hasattr(
            tokenizer, "audio_eos_token") else audio_eos_token)

        self.audio_stream_bos_id = audio_stream_bos_id
        self.audio_stream_eos_id = audio_stream_eos_id
        # HACK: Workaround the class check in the base class
        if feature_extractor is not None:
            self.feature_extractor_class = feature_extractor.__class__.__name__
        super().__init__(feature_extractor,
                         tokenizer,
                         chat_template=chat_template)
        self.feat_whisper_padding = False

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput],
                    List[PreTokenizedInput]] = None,
        audio: Union[np.ndarray, List[np.ndarray]] = None,
        audios=None,  # kept for BC
        padding: Union[bool, str, PaddingStrategy] = False,
        sampling_rate: Optional[int] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and 
        audio(s). Borrowed the code from Qwen2 Audio.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence 
                can be a string or a list of strings (pretokenized string). If 
                the sequences are provided as list of strings (pretokenized), 
                you must set `is_split_into_words=True` (to lift the ambiguity 
                with a batch of sequences).
            audios (`np.ndarray`, `List[np.ndarray]`):
                The audio or batch of audios to be prepared. Each audio can be 
                a NumPy array.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, 
                    defaults to `False`):
                Select a strategy to pad the returned sequences (according to 
                the model's padding side and padding index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the 
                  batch (or no padding if only a single sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the 
                  argument `max_length` or to the maximum acceptable input 
                  length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can 
                  output a batch with sequences of different lengths).
            sampling_rate (`int`, defaults to 16000):
                The sampling rate at which the audio files should be 
                digitalized expressed in hertz (Hz).
        """

        # Handle BC when user passes deprecared keyword argument
        if audios is not None and audio is None:
            audio = audios
            warnings.warn(
                "You may have used the keyword argument for the `audio` inputs. "
                "It is strongly recommended to pass inputs with keyword arguments "
                "with keys `audio` and `text`. From transformers v4.55 `audio` "
                "will be the only acceptable keyword argument.",
                FutureWarning,
            )

        if text is None:
            raise ValueError("You need to specify `text` input to process.")
        elif isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError(
                "Invalid input text. Please provide a string, or a list of strings"
            )

        if audio is not None:
            # ensure we have as much audios as audio tokens
            num_audio_tokens = sum(
                sample.count(self.audio_token) for sample in text)
            num_audios = 1 if type(audio) is np.ndarray else len(audio)
            if num_audio_tokens != num_audios:
                raise ValueError(
                    f"Found {num_audio_tokens} {self.audio_token} token"
                    f"{'s' if num_audio_tokens > 1 else ''} "
                    f"in provided text but received {num_audios} audio"
                    f"{'s' if num_audios > 1 else ''}")
            # Some kwargs should not be changed so we can expand text with audio tokens below
            if hasattr(self.feature_extractor, "encode"):
                if isinstance(audio, np.ndarray):
                    audio = [audio]
                audio = [a.astype(np.float32) for a in audio]
                audio_ids = [
                    self.feature_extractor.encode(
                        a, self.feature_extractor.sampling_rate).unsqueeze(0)
                    for a in audio
                ]

                # -2 is the number of codebooks
                num_codebook_dim = -2
                use_delay_pattern = audio_ids[0].shape[num_codebook_dim] > 1
                if use_delay_pattern:
                    for i, audio_id in enumerate(audio_ids):
                        audio_id = torch.cat([
                            torch.full(
                                (1, audio_id.shape[num_codebook_dim], 1),
                                self.audio_stream_bos_id,
                                dtype=torch.long,
                                device=audio_id.device),
                            audio_id,
                            torch.full(
                                (1, audio_id.shape[num_codebook_dim], 1),
                                self.audio_stream_eos_id,
                                dtype=torch.long,
                                device=audio_id.device),
                        ],
                                             dim=-1)
                        audio_ids[i] = \
                            _build_delay_pattern_mask(audio_id,
                                                      bos_token_id=self.audio_stream_bos_id,
                                                      pad_token_id=self.audio_stream_eos_id)

                audio_lengths = [a.shape[-1] for a in audio_ids]
                audio_in_ids_length = torch.tensor(audio_lengths)
                audio_in_ids = _validate_and_reshape_mm_tensor(audio_ids,
                                                               "audio_in_ids",
                                                               pad_with=0)
                audio_feature_attention_mask = torch.arange(
                    audio_in_ids.shape[-1]).expand(
                        audio_in_ids.shape[0], audio_in_ids.shape[-1]).to(
                            audio_in_ids_length.device
                        ) < audio_in_ids_length.unsqueeze(-1)
                audio_inputs = {
                    "input_features": audio_in_ids,
                    "audio_feature_attention_mask":
                    audio_feature_attention_mask,
                }
            else:
                # use_whisper = True
                audio_inputs = self.feature_extractor(
                    audio,
                    sampling_rate=sampling_rate,
                    return_attention_mask=True,
                    # Avoid using max_length to prevent overestimating
                    # the number of feature positions.
                    padding=self.feat_whisper_padding,
                    **kwargs,
                )
                # Rename to audio_feature_attention_mask to prevent conflicts
                # with text attention mask
                audio_inputs[
                    "audio_feature_attention_mask"] = audio_inputs.pop(
                        "attention_mask")
                audio_lengths = audio_inputs[
                    "audio_feature_attention_mask"].sum(-1).tolist()

            # Keep the audio placeholder as a single token in text.
            # vLLM will handle multimodal positions and embeddings via mm_kwargs.
            # Do not expand the placeholder into repeated tokens based on audio
            # length to avoid exploding prompt length.

        # TODO: text could have more than only audio tokens, we need to handle such cases
        text = [f"<|audio_bos|>{t}<|audio_eos|>" for t in text]
        logger.info(f"HiggsAudioProcessor: text before tokenization: {text}")
        inputs = self.tokenizer(text, padding=padding, **kwargs)

        if audio is not None:
            inputs.update(audio_inputs)

        return BatchFeature(data={**inputs})

    @property
    def default_chat_template(self):
        # fmt: off
        if self.is_audio_out_model:
            return (
                "{% set loop_messages = messages %}"
                "{% for message in loop_messages %}"
                    "{% set content = '<|start_header_id|>' + message['role'] + "
                    "'<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
                    "{% if loop.index0 == 0 %}"
                        "{% set content = bos_token + content %}"
                    "{% endif %}"
                    "{% if message['role'] == 'assistant' and '<|audio_bos|><|AUDIO|>' in message['content'] %}"
                        "{% set content = content.replace('<|audio_bos|><|AUDIO|>', '<|audio_out_bos|><|AUDIO|>') %}"
                    "{% endif %}"
                    "{{ content }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n<|audio_out_bos|><|AUDIO_OUT|>' }}"
                "{% endif %}"
            )

        return (
            "{% set loop_messages = messages %}"
            "{% for message in loop_messages %}"
                "{% set content = '<|start_header_id|>' + message['role'] + "
                "'<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
                "{% if loop.index0 == 0 %}"
                "{% set content = bos_token + content %}"
            "{% endif %}"
            "{{ content }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
            "{% endif %}"
        )
        # fmt: on


cached_get_processor = lru_cache(get_processor(HFHiggsAudioProcessor))
HiggsAudioFeatureExtractor = Union[AudioTokenizer, WhisperFeatureExtractor]


class HiggsAudioProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(HiggsAudioConfig)

    def get_hf_processor(
        self,
        *,
        # Ignored in initialization
        sampling_rate: Optional[int] = None,
        **kwargs: object,
    ) -> HFHiggsAudioProcessor:
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

    def get_feature_extractor(
        self,
        *,
        # Ignored in initialization
        sampling_rate: Optional[int] = None,
    ) -> HiggsAudioFeatureExtractor:
        hf_processor = self.get_hf_processor(sampling_rate=sampling_rate)
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        # Warn if tokenizer codebooks mismatch model expectation (common TTS issue)
        try:
            expected_cbs = int(self.get_hf_config().audio_num_codebooks)
            actual_cbs = int(getattr(feature_extractor, "num_codebooks", expected_cbs))
            if actual_cbs != expected_cbs:
                logger.warning(
                    "HiggsAudio: tokenizer num_codebooks=%d != model expected=%d. This can degrade TTS."
                    " Set HIGGS_AUDIO_TOKENIZER to a model with %d codebooks (e.g., xcodec_tps25_0215).",
                    actual_cbs, expected_cbs, expected_cbs)
        except Exception:
            pass
        return feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        hf_config = self.get_hf_config()
        # Determine max tokens based on input tokenizer type
        audio_input_tokenizer = os.getenv(
            "HIGGS_AUDIO_INPUT_TOKENIZER",
            os.getenv("HIGGS_AUDIO_TOKENIZER", "openai/whisper-large-v3-turbo"))
        
        if audio_input_tokenizer == "openai/whisper-large-v3-turbo":
            # Whisper (ASR): based on max source positions
            max_source_position = \
                hf_config.audio_encoder_config.max_source_positions
            max_output_lengths = (max_source_position - 2) // 2 + 1
            if hf_config.projector_temporal_downsample > 1:
                max_output_lengths = (max_output_lengths - 1) // hf_config.projector_temporal_downsample + 1
        else:
            # AudioTokenizer (TTS voice cloning): based on TPS
            try:
                fe = self.get_feature_extractor()
                tps = int(getattr(fe, "tps", 25))
                num_cbs = int(getattr(fe, "num_codebooks", 1))
                max_output_lengths = 30 * tps + num_cbs - 1 + 2  # +2 for BOS/EOS
            except Exception:
                # Fallback to Whisper calculation
                max_source_position = \
                    hf_config.audio_encoder_config.max_source_positions
                max_output_lengths = (max_source_position - 2) // 2 + 1
        
        return {"audio": max_output_lengths}


class HiggsAudioMultiModalProcessor(
        BaseMultiModalProcessor[HiggsAudioProcessingInfo]):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Determine input audio tokenizer: Whisper for ASR, xcodec for TTS voice cloning
        hf_config = self.info.get_hf_config()
        env_input_tok = os.getenv("HIGGS_AUDIO_INPUT_TOKENIZER")
        env_tok = os.getenv("HIGGS_AUDIO_TOKENIZER")
        prefer_audio_tokenizer = bool(getattr(hf_config, "skip_audio_tower", False))
        if env_input_tok:
            self.audio_input_tokenizer = env_input_tok
        elif env_tok:
            self.audio_input_tokenizer = env_tok
        else:
            self.audio_input_tokenizer = (
                "openai/whisper-large-v3-turbo" if not prefer_audio_tokenizer
                else "xcodec")  # sentinel; will trigger AudioTokenizer path
        self.use_whisper_tokenizer = (
            self.audio_input_tokenizer == "openai/whisper-large-v3-turbo")
        self.projector_temporal_downsample = int(getattr(hf_config, "projector_temporal_downsample", 1))

    def _get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(
            target_sr=self.info.get_feature_extractor().sampling_rate)

    # For HiggsAudio we do not rely on the HF processor to expand placeholders
    # into multimodal tokens. We always compute the placeholder ranges and
    # lengths ourselves based on the attention mask and the mel features.
    # Therefore, force vLLM to apply our prompt updates.
    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        **kwargs: Any,
    ) -> BatchFeature:
        # Text-only input not supported in composite processor
        if not mm_data.get("audios", []):
            # Set add_special_tokens=False to avoid
            # adding an extra begin of text token
            prompt_ids = self.info.get_tokenizer().encode(
                prompt, add_special_tokens=False)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            batch_data = BatchFeature(dict(input_ids=[prompt_ids]),
                                      tensor_type="pt")
            return batch_data

        hf_processor_mm_kwargs = kwargs.get("hf_processor_mm_kwargs", {})
        tok_kwargs = kwargs.get("tok_kwargs", {})
        feature_extractor = self.info.get_feature_extractor(
            **hf_processor_mm_kwargs)
        mm_kwargs = dict(
            **hf_processor_mm_kwargs,
            sampling_rate=feature_extractor.sampling_rate,
        )

        batch_data = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        input_features = batch_data.pop("input_features")
        import os
        if os.getenv("VLLM_TTS_DEBUG_MM", "0") == "1":
            try:
                logger.debug(
                    f"HiggsAudioMultiModalProcessor: input_features shape before assignment: {input_features.shape if isinstance(input_features, torch.Tensor) else [f.shape for f in input_features]}"
                )
            except Exception:
                pass

        # No hidden-state stashing here; embeddings are handled downstream.
        batch_data["audio_features"] = input_features
        return batch_data

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            audio_features=MultiModalFieldConfig.batched("audio"),
            audio_feature_attention_mask=MultiModalFieldConfig.batched(
                "audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        # Use getattr with default to be compatible with transformers<4.48
        audio_token = getattr(processor, "audio_token", "<|AUDIO|>")
        # CRITICAL: Align the placeholder token id with model config used later
        # during embedding merge, instead of deriving from tokenizer vocab.
        audio_token_id = int(self.info.get_hf_config().audio_in_token_idx)
        num_audio_items = mm_items.get_all_counts().get("audio", 0)

        def _sum_mask_to_int(mask: torch.Tensor) -> int:
            if mask.dtype not in (torch.long, torch.int64):
                mask = mask.to(dtype=torch.long)
            return int(mask.sum(-1).reshape(-1)[0].item())

        audio_output_lengths: list[int] = []
        mask_infos: list[Optional[dict]] = []
        feat_shapes: list[Optional[tuple]] = []
        audio_items_seq = out_mm_kwargs.get("audio", [])
        for i in range(num_audio_items):
            item_len = 0
            mask_info_i = None
            feat_shape_i: Optional[tuple] = None
            try:
                mm_item = audio_items_seq[i]
            except Exception:
                audio_output_lengths.append(1)
                mask_infos.append(mask_info_i)
                feat_shapes.append(feat_shape_i)
                continue

            mask_elem = mm_item.get("audio_feature_attention_mask")
            feat_elem = mm_item.get("audio_features")
            mask = getattr(mask_elem, "data", None) if mask_elem is not None else None
            feat = getattr(feat_elem, "data", None) if feat_elem is not None else None

            if isinstance(mask, torch.Tensor):
                try:
                    mel_len = _sum_mask_to_int(mask)
                    if self.use_whisper_tokenizer:
                        # Whisper features are downsampled by convs; convert
                        # mel frames to encoder output frames.
                        _, out_len = _get_feat_extract_output_lengths(
                            torch.tensor([mel_len], dtype=torch.long), self.projector_temporal_downsample)
                        item_len = int(out_len.reshape(-1)[0].item())
                    else:
                        # For codebook tokenizers (e.g., xcodec), each mask
                        # position corresponds to one embedding.
                        item_len = int(mel_len)
                    mask_info_i = {
                        "dtype": str(mask.dtype),
                        "shape": tuple(mask.shape),
                        "mel_len": int(mel_len),
                        "out_len": int(item_len),
                    }
                except Exception:
                    item_len = 0

            if (not item_len) and isinstance(feat, torch.Tensor):
                try:
                    f2d = feat
                    if f2d.ndim == 3:
                        f2d = f2d[0]
                    if f2d.ndim == 2:
                        T = f2d.shape[-1]
                        last = f2d[:, -1]
                        tol = 1e-6
                        diffs = (f2d - last.unsqueeze(-1)).abs().max(dim=0).values
                        non_pad = (diffs > tol).nonzero(as_tuple=False)
                        if non_pad.numel() == 0:
                            content_T = T
                        else:
                            content_T = int(non_pad[-1].item()) + 1
                        if self.use_whisper_tokenizer:
                            _, out_len = _get_feat_extract_output_lengths(
                                torch.tensor([content_T], dtype=torch.long), self.projector_temporal_downsample)
                            item_len = int(out_len.reshape(-1)[0].item())
                        else:
                            item_len = int(content_T)
                    feat_shape_i = tuple(feat.shape)
                except Exception:
                    item_len = item_len or 0
            else:
                if isinstance(feat, torch.Tensor):
                    feat_shape_i = tuple(feat.shape)

            audio_output_lengths.append(max(1, int(item_len)))
            mask_infos.append(mask_info_i)
            feat_shapes.append(feat_shape_i)

        # Ensure we have exactly one length per item
        if len(audio_output_lengths) > num_audio_items:
            audio_output_lengths = audio_output_lengths[:num_audio_items]
        elif len(audio_output_lengths) < num_audio_items:
            audio_output_lengths.extend([1] * (num_audio_items - len(audio_output_lengths)))

        max_tokens_cfg = self.info.get_mm_max_tokens_per_item(
            seq_len=0, mm_counts={"audio": num_audio_items}).get("audio")
        if max_tokens_cfg is not None and num_audio_items > 0:
            max_tokens_cfg = int(max_tokens_cfg)
            audio_output_lengths = [
                min(max_tokens_cfg, int(length))
                for length in audio_output_lengths
            ]

        # Debug: log derived lengths and inputs
        try:
            logger.debug(
                "HiggsAudio _get_prompt_updates: num_items=%d, mask_info=%s,\n\tfinal_out_lens=%s",
                num_audio_items, mask_infos, audio_output_lengths)
            logger.debug("HiggsAudio feature shapes per item: %s", feat_shapes)
        except Exception:
            pass

        

        def get_replacement_higgs_audio(item_idx: int):
            # Number of embeddings produced for this audio item
            num_features = int(audio_output_lengths[item_idx])
            if num_features <= 0:
                # Fallback to 1 token to keep the pipeline healthy during
                # profiling or degenerate inputs.
                try:
                    audios = mm_items.get_items("audio", AudioProcessorItems)
                    audio_len = audios.get_audio_length(item_idx)
                except Exception:
                    audio_len = -1
                logger.warning_once(
                    "Encountered non-positive audio feature length (%d). "
                    "Audio raw length: %d. Using 1 placeholder token.",
                    num_features, audio_len)
                num_features = 1

            # Only expand to TOKEN * num_features. Do not insert text BOS/EOS
            # here; they are not part of the embedded audio features.
            full = [int(audio_token_id)] * int(num_features)
            return PromptUpdateDetails.select_token_id(
                full, embed_token_id=int(audio_token_id))

        return [
            PromptReplacement(modality="audio",
                              target=audio_token,
                              replacement=get_replacement_higgs_audio)
        ]


class HiggsAudioDummyInputsBuilder(
        BaseDummyInputsBuilder[HiggsAudioProcessingInfo]):

    def get_dummy_text(
        self,
        mm_counts: Mapping[str, int],
    ) -> str:
        # Provide one audio placeholder per requested audio item
        num_audios = mm_counts.get("audio", 0)
        if num_audios <= 0:
            return "Hello"
        return "".join(["<|AUDIO|>" for _ in range(num_audios)])

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, object]:
        feature_extractor = self.info.get_feature_extractor()
        sampling_rate = getattr(feature_extractor, "sampling_rate", 16000)
        # Use a short fixed length (1s) for dummy audio unless chunk_length is available
        if hasattr(feature_extractor, "chunk_length"):
            audio_len = int(feature_extractor.chunk_length * sampling_rate)
        else:
            audio_len = int(1 * sampling_rate)
        num_audios = mm_counts.get("audio", 0)
        return {
            "audio": self._get_dummy_audios(length=audio_len,
                                            num_audios=num_audios)
        }

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        feature_extractor = self.info.get_feature_extractor()

        sampling_rate = feature_extractor.sampling_rate
        if hasattr(feature_extractor, "chunk_length"):
            audio_len = feature_extractor.chunk_length * sampling_rate
        else:
            # Default to 30 seconds audio
            audio_len = 30 * sampling_rate
        num_audios = mm_counts.get("audio", 0)

        mm_data = {
            "audio":
            self._get_dummy_audios(length=audio_len, num_audios=num_audios)
        }

        return ProcessorInputs(
            prompt="<|AUDIO|>" * num_audios,
            mm_data=mm_data,
        )


class HiggsAudioDualFFNDecoderLayer(nn.Module):
    """We implement a dual-path FFN decoder layer where the audio tokens and 
    text tokens go through separate FFN layers.

    The audio and text tokens share the text-attention layer, but will be 
    encoded with separate feedforward layers. In addition, the audio tokens can
    be configured to go through separate attention layer.

    Following is an illustration:

     t    t    t    a   a     a    t    t    t
                        |
                        | (audio self-attention layer)
                        v
    t    t     t    h'_a h'_a  h'_a  t  t    t
                        |
                        | (shared attention layer)
                        v
    h_t  h_t  h_t  h_a  h_a  h_a  h_t  h_t  h_t
                        |
                        | (separate text/audio hidden states)
                        v
    [h_t  h_t  h_t  h_t  h_t  h_t], [h_a, h_a, h_a]
             |                             |
             | (separate FFNs)             |
             v                             v
    [o_t  o_t  o_t  o_t  o_t  o_t], [o_a, o_a, o_a]
                        |
                        | (reorder)
                        v
    o_t  o_t  o_t  o_a  o_a  o_a  o_t  o_t  o_t

    This has a few advantages:
    1) We are able to use a smaller FFN, or even bypass the FFN for
        audio tokens. This accelerates the inference speed.
    2) The Audio-FFN introduces more trainable parameters to the model.
    This should have the same effect as the mixture-of-expert layer and
       we may expect better performance due to the scaling law.
    3) We can replace the original FFN in LLMs with the dual-path FFN without
       changing the model architecture.


    """

    def __init__(
        self,
        config: HiggsAudioConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        text_config = config.text_config
        self.hidden_size = text_config.hidden_size
        self.layer_idx = extract_layer_index(prefix)
        rope_theta = getattr(text_config, "rope_theta", 10000)
        rope_scaling = getattr(text_config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                text_config, "original_max_position_embeddings", None):
            rope_scaling[
                "original_max_position_embeddings"] = text_config.original_max_position_embeddings
        max_position_embeddings = getattr(text_config,
                                          "max_position_embeddings", 8192)
        attention_bias = getattr(text_config, "attention_bias",
                                 False) or getattr(text_config, "bias", False)
        self.self_attn = LlamaAttention(
            config=text_config,
            hidden_size=self.hidden_size,
            num_heads=text_config.num_attention_heads,
            num_kv_heads=getattr(text_config, "num_key_value_heads",
                                 text_config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            bias_o_proj=attention_bias,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=text_config.intermediate_size,
            hidden_act=text_config.hidden_act,
            quant_config=quant_config,
            bias=getattr(text_config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        self.fast_forward = self.layer_idx not in config.audio_dual_ffn_layers
        self.use_audio_attention = config.use_audio_out_self_attention

        if self.fast_forward or self.use_audio_attention:
            raise NotImplementedError(
                f"The fast-forward and audio-attention mode are not supported in "
                f"HiggsAudioDualFFNDecoderLayer, but got fast_forward={self.fast_forward}"
                f"and use_audio_attention={self.use_audio_attention}.")

        if not self.fast_forward:
            if self.use_audio_attention:
                self.audio_attn = LlamaAttention(
                    config=config,
                    hidden_size=self.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=getattr(config, "num_key_value_heads",
                                         config.num_attention_heads),
                    rope_theta=rope_theta,
                    rope_scaling=rope_scaling,
                    max_position_embeddings=max_position_embeddings,
                    quant_config=quant_config,
                    bias=attention_bias,
                    cache_config=cache_config,
                    prefix=f"{prefix}.self_attn",
                )
                self.audio_post_audio_attn_layer_norm = RMSNorm(
                    text_config.hidden_size, eps=text_config.rms_norm_eps)

            self.audio_mlp = LlamaMLP(
                hidden_size=self.hidden_size,
                intermediate_size=text_config.intermediate_size,
                hidden_act=text_config.hidden_act,
                quant_config=quant_config,
                bias=getattr(text_config, "mlp_bias", False),
                prefix=f"{prefix}.audio_mlp",
            )
            self.audio_input_layernorm = RMSNorm(text_config.hidden_size,
                                                 eps=text_config.rms_norm_eps)
            self.audio_post_attention_layernorm = RMSNorm(
                text_config.hidden_size, eps=text_config.rms_norm_eps)

        self.input_layernorm = RMSNorm(text_config.hidden_size,
                                       eps=text_config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(text_config.hidden_size,
                                                eps=text_config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ):
        assert (
            residual is None
        ), "The residual output is not supported in HiggsAudioDualFFNDecoderLayer."

        residual = hidden_states

        # if self.fast_forward and has_audio_out:
        #     original_hidden_states = hidden_states.clone()

        # Obtain per-token modality map from forward context if available.
        # In v1 engine the metadata may be set after forward; be robust here.
        ctx = get_forward_context()
        mm_meta = getattr(ctx, "multimodal_metadata", None)
        token_mm_map = getattr(mm_meta, "token_mm_map", None)
        seq_len = hidden_states.shape[0]
        if token_mm_map is None:
            mm_map_device = hidden_states.device
            token_mm_map = torch.zeros(seq_len,
                                       dtype=torch.bool,
                                       device=mm_map_device)
        else:
            # Align device and length with the current hidden states.
            token_mm_map = token_mm_map.to(device=hidden_states.device)
            cur_len = token_mm_map.shape[0]
            if cur_len != seq_len:
                if cur_len < seq_len:
                    padded = torch.zeros(seq_len,
                                         dtype=torch.bool,
                                         device=hidden_states.device)
                    padded[:cur_len] = token_mm_map
                    token_mm_map = padded
                else:
                    token_mm_map = token_mm_map[:seq_len]
        audio_out_mask = token_mm_map.unsqueeze(-1)
        # Optional deep layer trace (per-token, per-layer) for debugging
        _trace = os.getenv("HIGGS_TTS_TRACE", "0") == "1"
        if _trace:
            try:
                trace_pos_env = os.getenv("HIGGS_TRACE_POS", "-1")
                tpos = int(trace_pos_env)
                vec = hidden_states[tpos]
                _n = float(torch.linalg.norm(vec).item())
                _mu = float(vec.mean().item())
                _sd = float(vec.std().item())
                _min = float(vec.min().item())
                _max = float(vec.max().item())
                logger.debug(
                    "DualFFN[%d]/pre-ln: pos=%d norm=%.5f mean=%.5f std=%.5f min=%.5f max=%.5f",
                    self.layer_idx, tpos if tpos >= 0 else (hidden_states.shape[0] + tpos),
                    _n, _mu, _sd, _min, _max)
            except Exception:
                pass

        if not self.fast_forward:
            hidden_states = torch.where(
                audio_out_mask,
                self.audio_input_layernorm(hidden_states),
                self.input_layernorm(hidden_states),
            )
        else:
            hidden_states = self.input_layernorm(hidden_states)

        # # Audio Attention
        # if self.use_audio_attention and has_audio_out:
        #     assert (
        #         kv_cache.shape[0] == 4
        #     ), "The KV cache should have shape (4, batch_size, seq_len, hidden_size)"
        #     audio_hidden_states = self.audio_attn(
        #         positions=positions,
        #         hidden_states=hidden_states,
        #         kv_cache=kv_cache[2:4],
        #         attn_metadata=attn_metadata,
        #     )
        #     audio_hidden_states = residual + audio_hidden_states
        #     residual = torch.where(audio_out_mask.unsqueeze(-1),
        #                            audio_hidden_states, residual)
        #     audio_hidden_states = self.audio_post_audio_attn_layer_norm(
        #         audio_hidden_states)
        #     hidden_states = torch.where(audio_out_mask.unsqueeze(-1),
        #                                 audio_hidden_states, hidden_states)

        # Text Attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states = residual + hidden_states

        # Apply Dual-path FFN
        residual = hidden_states

        if _trace:
            try:
                trace_pos_env = os.getenv("HIGGS_TRACE_POS", "-1")
                tpos = int(trace_pos_env)
                vec = hidden_states[tpos]
                _n = float(torch.linalg.norm(vec).item())
                _mu = float(vec.mean().item())
                _sd = float(vec.std().item())
                _min = float(vec.min().item())
                _max = float(vec.max().item())
                logger.debug(
                    "DualFFN[%d]/post-attn: pos=%d norm=%.5f mean=%.5f std=%.5f min=%.5f max=%.5f",
                    self.layer_idx, tpos if tpos >= 0 else (hidden_states.shape[0] + tpos),
                    _n, _mu, _sd, _min, _max)
            except Exception:
                pass

        if not self.fast_forward:
            audio_out_indices = torch.where(audio_out_mask.squeeze(-1))[0]
            text_indices = torch.where(~audio_out_mask.squeeze(-1))[0]

            output_hidden_states = hidden_states.clone()

            if len(text_indices) > 0:
                text_only = hidden_states[text_indices]
                text_only = self.post_attention_layernorm(text_only)
                text_only = self.mlp(text_only)
                output_hidden_states[text_indices] = text_only

            if len(audio_out_indices) > 0:
                audio_only = hidden_states[audio_out_indices]
                audio_only = self.audio_post_attention_layernorm(audio_only)
                audio_only = self.audio_mlp(audio_only)
                output_hidden_states[audio_out_indices] = audio_only

            hidden_states = output_hidden_states
        else:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

        # if self.fast_forward:
        #     hidden_states = torch.where(audio_out_mask.unsqueeze(-1),
        #                                 original_hidden_states, hidden_states)

        # Add a None as the residual output for the compatibility
        if _trace:
            try:
                trace_pos_env = os.getenv("HIGGS_TRACE_POS", "-1")
                tpos = int(trace_pos_env)
                vec = hidden_states[tpos]
                _n = float(torch.linalg.norm(vec).item())
                _mu = float(vec.mean().item())
                _sd = float(vec.std().item())
                _min = float(vec.min().item())
                _max = float(vec.max().item())
                logger.debug(
                    "DualFFN[%d]/post-ffn: pos=%d norm=%.5f mean=%.5f std=%.5f min=%.5f max=%.5f",
                    self.layer_idx, tpos if tpos >= 0 else (hidden_states.shape[0] + tpos),
                    _n, _mu, _sd, _min, _max)
            except Exception:
                pass

        outputs = (hidden_states, None)

        return outputs


@MULTIMODAL_REGISTRY.register_processor(
    HiggsAudioMultiModalProcessor,
    info=HiggsAudioProcessingInfo,
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
class HiggsAudioForConditionalGeneration(nn.Module, SupportsMultiModal):

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        # Ensure chat_utils has a clear placeholder mapping for audio
        if modality.startswith("audio"):
            return "<|AUDIO|>"
        return None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", llm_backbone=LlamaDecoderLayer):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config

        self.multimodal_config = multimodal_config

        # Force to set attention implementation
        config.audio_encoder_config._attn_implementation = "sdpa"
        self.audio_tower = HiggsAudioEncoder(config.audio_encoder_config)

        self.quant_config = quant_config

        self.embed_tokens = nn.Embedding(config.text_config.vocab_size,
                                         config.text_config.hidden_size,
                                         config.pad_token_id)

        if config.audio_adapter_type == "dual_ffn_fast_forward":
            self.start_layer, self.end_layer, self.layers = make_layers(
                config.text_config.num_hidden_layers,
                lambda prefix: HiggsAudioDualFFNDecoderLayer(
                    config=config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers",
                ),
                prefix=f"{prefix}.layers",
            )
        elif config.audio_adapter_type == "stack":
            self.start_layer, self.end_layer, self.layers = make_layers(
                config.text_config.num_hidden_layers,
                lambda prefix: llm_backbone(
                    config=config.text_config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers",
                ),
                prefix=f"{prefix}.layers",
            )
        else:
            raise NotImplementedError(
                f"Audio adapter type {config.audio_adapter_type} not implemented."
            )
        self.norm = RMSNorm(config.text_config.hidden_size,
                            eps=config.text_config.rms_norm_eps)

        is_neox_style = True
        if quant_config is not None and quant_config.get_name() == "gguf":
            is_neox_style = False
        self.rotary_emb = get_rope(
            head_size=config.text_config.head_dim,
            rotary_dim=config.text_config.head_dim,
            max_position=config.text_config.max_position_embeddings,
            base=config.text_config.rope_theta,
            rope_scaling=config.text_config.rope_scaling,
            is_neox_style=is_neox_style,
        )

        self.audio_encoder_proj = HiggsAudioFeatureProjector(vllm_config)
        # We add 1 for the audio_stream_bos token and 1
        # for theaudio_stream_eos token
        self.audio_codebook_size = (config.audio_codebook_size + 2)
        self.audio_num_codebooks = config.audio_num_codebooks

        # HACK: This is a hack to tell if it is a audio generation model
        # FIXME: This should be fixed. We should simply reply on the token
        # history to determine if we should generate audio out tokens.
        self.generate_audio_out_token = config.skip_audio_tower
        # Determine input tokenizer: Whisper for ASR, xcodec for TTS voice cloning
        self.audio_input_tokenizer = os.getenv(
            "HIGGS_AUDIO_INPUT_TOKENIZER",
            os.getenv("HIGGS_AUDIO_TOKENIZER", "openai/whisper-large-v3-turbo"))
        self.use_whisper_tokenizer = (
            self.audio_input_tokenizer == "openai/whisper-large-v3-turbo")

        if config.use_audio_out_embed_projector:
            self.audio_out_embed_projector = nn.Linear(
                config.text_config.hidden_size,
                config.text_config.hidden_size,
                bias=False)

        self.audio_codebook_embeddings = nn.Embedding(
            config.audio_num_codebooks * self.audio_codebook_size,
            config.text_config.hidden_size)

        self.text_lm_head = ParallelLMHead(
            config.text_config.vocab_size,
            config.text_config.hidden_size,
            quant_config=quant_config,
            bias=False,
        )

        self.audio_lm_head = ParallelLMHead(
            config.audio_num_codebooks * self.audio_codebook_size,
            config.text_config.hidden_size,
            quant_config=quant_config,
            bias=False,
        )

        if get_pp_group().is_last_rank:
            self.audio_decoder_proj = HiggsAudioDecoderProjector(vllm_config)
            # Text logits scale (keep config default)
            text_logit_scale = float(getattr(config, "logit_scale", 1.0))
            self.logits_processor = LogitsProcessor(
                config.text_config.vocab_size,
                config.text_config.vocab_size,
                text_logit_scale,
            )

            # Audio logits scale: force 1.0 by default for 0.8.3 parity.
            try:
                import os as _os
                if _os.getenv("HIGGS_AUDIO_LOGIT_SCALE") is not None:
                    audio_logit_scale = float(_os.getenv("HIGGS_AUDIO_LOGIT_SCALE"))
                elif _os.getenv("HIGGS_TTS_STRICT_083", "1") == "1":
                    audio_logit_scale = 1.0
                else:
                    audio_logit_scale = float(getattr(config, "audio_logit_scale", 1.0))
            except Exception:
                audio_logit_scale = 1.0
            self.audio_logits_processor = LogitsProcessor(
                self.audio_lm_head.num_embeddings_padded,
                self.audio_lm_head.org_vocab_size,
                audio_logit_scale,
            )
            self.sampler = get_sampler()
            
        # v0.10.2 workaround: Track last audio tokens for decode steps
        self._last_audio_tokens = None

        # 0.8.3-compat: Install optional per-layer tracing hooks for deep
        # visibility. Enabled via HIGGS_TTS_LAYER_TRACE=1. Limit logs via
        # HIGGS_TTS_LAYER_TRACE_BUDGET (default 200 total hook prints).
        try:
            import os as _os
            if _os.getenv("HIGGS_TTS_LAYER_TRACE", "0") == "1":
                self._layer_trace_budget = int(
                    _os.getenv("HIGGS_TTS_LAYER_TRACE_BUDGET", "200"))

                def _mk_hook(tag: str):
                    def _hook(_module, _inp, _out):
                        try:
                            if self._layer_trace_budget <= 0:
                                return
                            x = _out[0] if isinstance(_out, tuple) else _out
                            if not isinstance(x, torch.Tensor):
                                return
                            s = x.detach()
                            mu = float(s.mean().item())
                            sd = float(s.std().item())
                            mn = float(s.min().item())
                            mx = float(s.max().item())
                            logger.debug(
                                "LAYER_TRACE %s: shape=%s mean=%.5f std=%.5f min=%.5f max=%.5f",
                                tag, tuple(s.shape), mu, sd, mn, mx)
                            self._layer_trace_budget -= 1
                        except Exception:
                            pass
                    return _hook

                # Trace text backbone decoder layers
                for i, layer in enumerate(self.layers):
                    try:
                        layer.register_forward_hook(_mk_hook(f"text.layer[{i}]"))
                    except Exception:
                        pass

                # Trace audio decoder projection stack if present
                try:
                    proj_layers = getattr(self, "audio_decoder_proj", None)
                    if proj_layers is not None and hasattr(proj_layers,
                                                           "layers"):
                        for j, al in enumerate(proj_layers.layers):
                            try:
                                al.register_forward_hook(
                                    _mk_hook(f"audio_proj.layer[{j}]"))
                            except Exception:
                                pass
                except Exception:
                    pass
        except Exception:
            # Tracing is best-effort; never fail initialization for hooks.
            pass

    def _parse_and_validate_audio_input(
            self, **kwargs: object) -> Optional[HiggsAudioInputs]:
        audio_features = kwargs.pop("audio_features", None)
        audio_feature_attention_mask = kwargs.pop(
            "audio_feature_attention_mask", None)
        audio_out_ids = kwargs.pop("audio_out_ids", None)
        if audio_features is None and audio_out_ids is None:
            return None
        if audio_features is not None:
            audio_features = _validate_and_reshape_mm_tensor(
                audio_features,
                "audio_features",
                pad_with=0 if not self.use_whisper_tokenizer else None)
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
        audio_feat_out_lengths = audio_feat_out_lengths.clamp_max(audio_features.size(1))  # safety, might be redundant
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
        audio_features = audio_input["audio_features"]
        audio_feature_attention_mask = audio_input[
            "audio_feature_attention_mask"]

        if self.use_whisper_tokenizer:
            return self._process_whisper_audio_input(
                audio_features, audio_feature_attention_mask)

        audio_features_flattened = audio_features.transpose(1, 0).reshape(
            audio_features.shape[1], -1)
        audio_features_embeddings = self._embed_audio_ids(
            audio_features_flattened)
        audio_features_attention_mask_flattened = (
            audio_feature_attention_mask.flatten())
        masked_audio_features_embeddings = audio_features_embeddings[
            audio_features_attention_mask_flattened]
        audio_features_lens = audio_feature_attention_mask.sum(-1)
        masked_audio_features_embeddings = torch.split(
            masked_audio_features_embeddings, audio_features_lens.tolist())
        return masked_audio_features_embeddings

    def _embed_audio_ids(self, audio_ids):
        """Embed the audio ids

        Args:
            audio_ids: torch.LongTensor of shape (num_codebooks, audio_in_total_length)

        Returns:
            audio_embed: torch.LongTensor of shape (audio_in_total_length, hidden_size)
        """
        # 0.8.3 parity: identity codebook order for embeddings by default.
        # Allow overriding via env HIGGS_AUDIO_EMBED_CB_PERM to keep the
        # embedding codebook order consistent with the decoder-side order.
        try:
            if not hasattr(self, "_embed_cb_perm_cache"):
                self._embed_cb_perm_cache = None  # type: ignore[attr-defined]
            if self._embed_cb_perm_cache is None:
                import os as _os
                val = _os.getenv("HIGGS_AUDIO_EMBED_CB_PERM")
                perm = None
                if val:
                    parts = [p.strip() for p in val.split(",") if p.strip()]
                    if len(parts) == int(self.audio_num_codebooks):
                        perm = torch.tensor([int(x) for x in parts],
                                            dtype=torch.long)
                self._embed_cb_perm_cache = perm  # type: ignore[attr-defined]
            if self._embed_cb_perm_cache is not None:
                order = self._embed_cb_perm_cache.to(device=audio_ids.device)  # type: ignore[attr-defined]
            else:
                order = torch.arange(self.audio_num_codebooks,
                                     device=audio_ids.device)
        except Exception:
            order = torch.arange(self.audio_num_codebooks, device=audio_ids.device)
        codebook_shift = order * self.audio_codebook_size
        codebook_shift = codebook_shift.unsqueeze(-1)

        # Compute per-codebook embeddings.
        # Preserve BOS/EOS embeddings; the model was trained with these
        # discrete symbols and expects their signal in context.
        idx = audio_ids + codebook_shift
        audio_embed_cb = self.audio_codebook_embeddings(idx)
        audio_embed = torch.sum(audio_embed_cb, dim=0)
        if self.config.use_audio_out_embed_projector:
            audio_embed = self.audio_out_embed_projector(audio_embed)
        return audio_embed

    def get_multimodal_embeddings(self, **kwargs) -> Optional[NestedTensors]:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return None
        if audio_input["audio_features"] is not None:
            masked_audio_features = self._process_audio_input(audio_input)
        else:
            masked_audio_features = None
        # audio_out_ids hold the freshly generated codebooks. Encode them and
        # append after the audio-in embeddings so the ordering mirrors 0.8.3.
        audio_out_embeddings = None
        if kwargs.get("audio_out_ids", None) is not None:
            audio_out_ids = kwargs["audio_out_ids"]
            try:
                import os
                if os.getenv("VLLM_TTS_DEBUG_MM", "0") == "1":
                    logger.debug(
                        f"get_multimodal_embeddings: audio_out_ids received (shape={audio_out_ids.shape if hasattr(audio_out_ids, 'shape') else 'unknown'}); embedding them"
                    )
            except Exception:
                pass

            try:
                target_device = self.audio_codebook_embeddings.weight.device
            except Exception:
                target_device = next(self.parameters()).device

            try:
                if not isinstance(audio_out_ids, torch.Tensor):
                    audio_out_ids = torch.tensor(audio_out_ids, dtype=torch.long)
                if audio_out_ids.ndim >= 3:
                    audio_out_ids = audio_out_ids.flatten(0, -2)
                if audio_out_ids.ndim == 1:
                    audio_out_ids = audio_out_ids.unsqueeze(0)
                assert audio_out_ids.shape[-1] == self.audio_num_codebooks, \
                    f"audio_out_ids last dim {audio_out_ids.shape[-1]} != num_codebooks {self.audio_num_codebooks}"
            except Exception:
                audio_out_ids = torch.tensor(audio_out_ids, dtype=torch.long)
                audio_out_ids = audio_out_ids.view(-1, self.audio_num_codebooks)

            # Transpose to [num_codebooks, batch] for _embed_audio_ids
            audio_out_ids = audio_out_ids.to(target_device, non_blocking=True)
            audio_out_flattened = audio_out_ids.transpose(1, 0)

            # Embed: [num_codebooks, batch] -> [batch, hidden]
            audio_out_embeddings = self._embed_audio_ids(audio_out_flattened)

            # Split into list of 2D tensors [1, hidden] per batch item.
            audio_out_embeddings = [emb.unsqueeze(0)
                                    for emb in audio_out_embeddings]

        try:
            import os
            if os.getenv("VLLM_TTS_DEBUG_MM", "0") == "1":
                if audio_out_embeddings is not None:
                    logger.debug(
                        f"get_multimodal_embeddings: encoded {len(audio_out_embeddings)} audio_out frames"
                    )
        except Exception:
            pass

        if masked_audio_features is None:
            masked_audio_features = []
        else:
            if isinstance(masked_audio_features, tuple):
                masked_audio_features = list(masked_audio_features)

        if audio_out_embeddings is not None:
            masked_audio_features.extend(audio_out_embeddings)
        else:
            # Scheduler may omit the -1 placeholder in the same step. Fall back
            # to the cached last frames (recorded by the runner for every
            # scheduled token) so we still feed audio-out embeddings, matching
            # the legacy patches.
            try:
                cached_frames = getattr(self, "last_mm_token_ids_batch", None)
            except Exception:
                cached_frames = None
            if cached_frames:
                try:
                    device = self.audio_codebook_embeddings.weight.device
                except Exception:
                    device = next(self.parameters()).device
                fallback_embeds: list[torch.Tensor] = []
                for frame in cached_frames:
                    if frame is None:
                        continue
                    try:
                        frame_tensor = torch.tensor(frame, dtype=torch.long)
                        frame_tensor = frame_tensor.to(device)
                    except Exception:
                        continue
                    if frame_tensor.ndim == 1:
                        frame_tensor = frame_tensor.unsqueeze(-1)
                    if frame_tensor.shape[0] != self.audio_num_codebooks:
                        continue
                    try:
                        embed = self._embed_audio_ids(frame_tensor)
                        fallback_embeds.append(embed.unsqueeze(0))
                    except Exception:
                        continue
                if fallback_embeds:
                    masked_audio_features.extend(fallback_embeds)

        return masked_audio_features if masked_audio_features else None

    def get_multimodal_output_embeddings(self) -> list:
        """
        Return multimodal output embeddings (audio_out_ids) for the v1 engine.
        
        This is called by the model runner during _execute_encoder to properly
        encode audio output tokens through get_multimodal_embeddings, matching
        the 0.8.3 behavior where audio_out_ids are passed via **kwargs.
        
        Returns:
            list of MultiModalKwargsItem containing audio_out_ids for each request
        """
        # Allow disabling feedback of output embeddings for troubleshooting.
        # If set, we won't feed the last sampled audio codes back as embeddings;
        # the model will condition only on text states, similar to older flows.
        # 0.8.3-compat: feed back last sampled audio codes as embeddings so
        # the next decode step conditions on them.
        from vllm.multimodal.inputs import (MultiModalKwargsItem,
                                            MultiModalFieldElem,
                                            MultiModalBatchedField)
        
        if not self.generate_audio_out_token or not hasattr(self, 'model_runner'):
            return []
        
        mm_outputs = []
        req_input_ids_for_mm_outputs = []
        
        # Access the model runner's request states
        if hasattr(self.model_runner, 'requests') and hasattr(self.model_runner, 'input_batch'):
            for req_id in self.model_runner.input_batch.req_ids:
                req_state = self.model_runner.requests.get(req_id)
                if req_state and hasattr(req_state, 'output_mm_token_ids'):
                    if req_state.output_mm_token_ids and len(req_state.output_mm_token_ids) > 0:
                        # Get the last frame of audio tokens
                        last_frame = req_state.output_mm_token_ids[-1]
                        # Convert to [num_codebooks] tensor (NOT [1, num_codebooks])
                        # because group_mm_kwargs_by_modality will add batch dim
                        # to match 0.8.3's expected shape
                        # Get device from model parameters
                        audio_out_ids = torch.tensor(last_frame,
                                                     dtype=torch.long)
                        
                        import os
                        if os.getenv("VLLM_TTS_DEBUG_MM", "0") == "1":
                            from vllm.logger import init_logger
                            logger = init_logger(__name__)
                            logger.debug(f"get_multimodal_output_embeddings: creating audio_out_ids from last_frame={last_frame}, shape={audio_out_ids.shape}")
                        
                        # Create MultiModalFieldElem for audio_out_ids.
                        # Use a batched field with a single item: shape [1, num_codebooks]
                        # so that get_multimodal_embeddings receives a batch dimension.
                        field_elem = MultiModalFieldElem(
                            modality="audio",
                            key="audio_out_ids",
                            data=audio_out_ids.unsqueeze(0),
                            field=MultiModalBatchedField()
                        )
                        
                        mm_outputs.append(MultiModalKwargsItem.from_elems([field_elem]))
                        # Use -1 as the input_id to indicate this is an output embedding
                        req_input_ids_for_mm_outputs.append((req_id, -1))
        
        # Store for get_req_input_ids_for_mm_outputs
        self._req_input_ids_for_mm_outputs = req_input_ids_for_mm_outputs
        return mm_outputs

    def get_req_input_ids_for_mm_outputs(self) -> list:
        """Return the (req_id, input_id) tuples for multimodal output embeddings."""
        return getattr(self, '_req_input_ids_for_mm_outputs', [])

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> torch.Tensor:
        # Optional prompt-level tracing to understand how many mm placeholders
        # we are about to merge. Enable via HIGGS_TTS_TRACE=1.
        try:
            if os.getenv("HIGGS_TTS_TRACE", "0") == "1":
                mm_count = 0
                if multimodal_embeddings is not None:
                    if isinstance(multimodal_embeddings, (list, tuple)):
                        mm_count = sum(
                            1 for _e in multimodal_embeddings if isinstance(
                                _e, torch.Tensor))
                    elif isinstance(multimodal_embeddings, torch.Tensor):
                        mm_count = int(multimodal_embeddings.shape[0] > 0)
                logger.debug(
                    "get_input_embeddings: input_ids.shape=%s mm_count=%d",
                    tuple(input_ids.shape), mm_count)
        except Exception:
            pass
        inputs_embeds = self.embed_tokens(input_ids)

        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                [
                    int(self.config.audio_in_token_idx),
                    int(self.config.audio_out_token_idx),
                ],
            )

        return inputs_embeds

    def get_input_mm_map(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.isin(
            input_ids,
            torch.tensor([
                self.config.audio_in_token_idx, self.config.audio_out_token_idx
            ],
                         device=input_ids.device))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # Use a local alias for os to avoid accidental shadowing issues
        # from any nested imports within this function scope.
        import os as _os
        if get_pp_group().is_first_rank:
            # 0.8.3 parity: Always compute embeddings from input_ids for TTS
            # decode steps. For engine profiling, input_ids is None, so we
            # must use the provided inputs_embeds to avoid a crash.
            if input_ids is not None:
                # Debug: Check if we're in a decode step with audio tokens
                try:
                    import os as _os
                    if _os.getenv("VLLM_TTS_DEBUG_MM", "0") == "1" and self.generate_audio_out_token:
                        audio_out_token_idx = int(self.config.audio_out_token_idx)
                        if input_ids.numel() > 0 and (input_ids == audio_out_token_idx).any():
                            logger.debug(f"forward: Found audio_out tokens in input_ids, shape={input_ids.shape}")
                            # For decode steps with audio tokens, we need to ensure
                            # audio_out_ids are passed for proper embedding
                            if kwargs.get("audio_out_ids") is None:
                                logger.debug("forward: audio_out tokens present but no audio_out_ids in kwargs! Will use stored tokens.")
                except Exception:
                    pass
                
                multimodal_embeddings = self.get_multimodal_embeddings(**kwargs)
                hidden_states = self.get_input_embeddings(
                    input_ids, multimodal_embeddings)
            else:
                hidden_states = inputs_embeds
            
            residual = None

            # Optional embedding-level trace
            if _os.getenv("HIGGS_TTS_TRACE", "0") == "1":
                try:
                    tpos = int(_os.getenv("HIGGS_TRACE_POS", "-1"))
                    vec = hidden_states[tpos]
                    _n = float(torch.linalg.norm(vec).item())
                    _mu = float(vec.mean().item())
                    _sd = float(vec.std().item())
                    _min = float(vec.min().item())
                    _max = float(vec.max().item())
                    logger.debug(
                        "Higgs.forward/embed: pos=%d norm=%.5f mean=%.5f std=%.5f min=%.5f max=%.5f",
                        tpos if tpos >= 0 else (hidden_states.shape[0] + tpos),
                        _n, _mu, _sd, _min, _max)
                except Exception:
                    pass
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            _trace = _os.getenv("HIGGS_TTS_TRACE", "0") == "1"
            if _trace:
                try:
                    tpos = int(_os.getenv("HIGGS_TRACE_POS", "-1"))
                    vec = hidden_states[tpos]
                    logger.debug(
                        "Higgs.layer[%d]/in: pos=%d L2=%.5f", i,
                        tpos if tpos >= 0 else (hidden_states.shape[0] + tpos),
                        float(torch.linalg.norm(vec).item()))
                except Exception:
                    pass
            if isinstance(layer, HiggsAudioDualFFNDecoderLayer):
                hidden_states, _ = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=None,
                )
            else:
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    residual,
                )
            if _trace:
                try:
                    tpos = int(os.getenv("HIGGS_TRACE_POS", "-1"))
                    vec = hidden_states[tpos]
                    logger.debug(
                        "Higgs.layer[%d]/out: pos=%d L2=%.5f", i,
                        tpos if tpos >= 0 else (hidden_states.shape[0] + tpos),
                        float(torch.linalg.norm(vec).item()))
                except Exception:
                    pass

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states)

        # Record positions for use during logits computation (audio projector).
        try:
            self._last_positions = positions
        except Exception:
            self._last_positions = None

        if os.getenv("HIGGS_TTS_TRACE", "0") == "1":
            try:
                tpos = int(os.getenv("HIGGS_TRACE_POS", "-1"))
                vec = hidden_states[tpos]
                _n = float(torch.linalg.norm(vec).item())
                _mu = float(vec.mean().item())
                _sd = float(vec.std().item())
                _min = float(vec.min().item())
                _max = float(vec.max().item())
                logger.debug(
                    "Higgs.forward/post-norm: pos=%d norm=%.5f mean=%.5f std=%.5f min=%.5f max=%.5f",
                    tpos if tpos >= 0 else (hidden_states.shape[0] + tpos),
                    _n, _mu, _sd, _min, _max)
            except Exception:
                pass

        # Optionally apply the audio decoder projector to the states used for
        # audio logits only. Some checkpoints expect the projector not to
        # modify the text head states. Default keeps previous behavior (both).
        audio_hidden_states = hidden_states
        if self.generate_audio_out_token and hasattr(self, "audio_decoder_proj") and \
           getattr(self.audio_decoder_proj, "_audio_decoder_proj_num_layers", 0) > 0:
            try:
                disable_proj = os.getenv("HIGGS_TTS_DISABLE_AUDIO_DECODER_PROJ", "0") == "1"
            except Exception:
                disable_proj = False
            try:
                apply_to_text = os.getenv("HIGGS_TTS_APPLY_AUDIO_DECODER_PROJ_FOR_TEXT", "1") == "1"
            except Exception:
                apply_to_text = True
            try:
                if not disable_proj:
                    audio_hidden_states = self.audio_decoder_proj(hidden_states, positions=positions)
                    if apply_to_text:
                        hidden_states = audio_hidden_states
            except Exception:
                pass

        # Optional sanitization to prevent NaN propagation from model core.
        try:
            import os as _os, torch as _t
            if _os.getenv("HIGGS_TTS_SANITIZE", "0") == "1":
                if hidden_states.is_floating_point() and not _t.isfinite(hidden_states).all():
                    hidden_states = _t.nan_to_num(hidden_states, nan=0.0, posinf=1e4, neginf=-1e4)
                    
        except Exception:
            pass

        # Stash audio-specific states for compute_logits (0.8.3 parity)
        try:
            self._last_audio_hidden_states = audio_hidden_states
        except Exception:
            self._last_audio_hidden_states = None

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 0.8.3 parity: compute both heads on the (already projected) hidden states.
        text_logits = self.logits_processor(self.text_lm_head, hidden_states,
                                            sampling_metadata)
        # Optional text logits top-k for traced token
        if os.getenv("HIGGS_TTS_TRACE", "0") == "1":
            try:
                # Select the row corresponding to the traced position in the batch-flattened dimension
                row = -1
                topv, topi = torch.topk(text_logits[row], k=min(10, text_logits.shape[-1]))
                logger.debug("Higgs.text_logits topk: idx=%s val=%s",
                             topi.detach().tolist(), [float(x) for x in topv.detach().tolist()])
            except Exception:
                pass

        if self.generate_audio_out_token:
            # audio_logits should NOT use sampling_metadata (temperature/top_p).
            # Match 0.8.3: compute on the same sampled hidden_states and pass
            # None to skip sampling transformations.
            try:
                if os.getenv("VLLM_TTS_DEBUG_MM", "0") == "1":
                    mu = float(hidden_states.mean().item())
                    sd = float(hidden_states.std().item())
                    logger.debug(
                        "compute_logits: audio_hidden_states stats mean=%.5f std=%.5f",
                        mu, sd)
            except Exception:
                pass
            audio_logits = self.audio_logits_processor(
                self.audio_lm_head, hidden_states, None)

            # Reshape robustly to [batch, n_cb, cb_vocab]. Accept padded vocab.
            try:
                batch_size, total_vocab = audio_logits.shape[0], audio_logits.shape[1]
                if total_vocab % self.audio_num_codebooks == 0:
                    per_cb = total_vocab // self.audio_num_codebooks
                    audio_logits = audio_logits.view(batch_size, self.audio_num_codebooks, per_cb).float()
                else:
                    # Fallback: best-effort view; keep last dim as-is
                    audio_logits = audio_logits.view(batch_size, self.audio_num_codebooks, -1).float()
            except Exception:
                audio_logits = audio_logits.unsqueeze(-1).repeat(1, 1, self.audio_codebook_size).float()

            # Debug: log audio logits in compute_logits (matches 0.8.3 behavior)
            try:
                import os as _os
                if _os.getenv("VLLM_TTS_DEBUG_MM", "0") == "1":
                    logger.debug(
                        "audio_logits debug: shape=%s, dtype=%s finite=%s",
                        tuple(audio_logits.shape), str(audio_logits.dtype),
                        bool(torch.isfinite(audio_logits).all()))
            except Exception:
                pass
        else:
            audio_logits = None
        # Stash audio_logits for sample_with_multimodal_metadata to retrieve.
        # Return only text_logits so the default sampler (used during profiling
        # dummy runs) receives a plain tensor, not a tuple.
        self._last_audio_logits = audio_logits
        return text_logits

    def sample(self, logits: torch.Tensor,
               sampling_metadata: SamplingMetadata) -> Optional[SamplerOutput]:
        raise NotImplementedError(
            "HiggsAudio expects sample_with_multimodal_metadata to be used.")

    def sample_with_multimodal_metadata(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        multimodal_metadata: MultimodalMetadata,
    ) -> Optional[SamplerOutput]:
        # audio_logits are stashed by compute_logits (no longer returned as tuple)
        audio_logits = getattr(self, '_last_audio_logits', None)
        next_tokens = self.sampler(logits, sampling_metadata)
        try:
            _orig_text_tokens = next_tokens.sampled_token_ids.clone()
        except Exception:
            _orig_text_tokens = None
        next_mm_tokens = None
        n_reqs = logits.shape[0]

        if n_reqs != len(multimodal_metadata.num_audio_eos):
            return next_tokens, next_mm_tokens

        if self.generate_audio_out_token:
            audio_generation_mode = [0] * n_reqs
            for i in range(n_reqs):
                last_prompt_token_id = multimodal_metadata.last_prompt_token_ids[i]
                output_token_ids = sampling_metadata.output_token_ids[i]
                if (output_token_ids and
                        output_token_ids[-1] == self.config.audio_out_bos_token_id) or (
                        not output_token_ids and
                        last_prompt_token_id == self.config.audio_out_bos_token_id):
                    audio_generation_mode[i] = 1
                elif (output_token_ids and
                      output_token_ids[-1] == self.config.audio_out_token_idx):
                    audio_generation_mode[i] = 2

            assert audio_logits is not None
            audio_logits = audio_logits.reshape(-1, self.audio_codebook_size)
            mm_sampling_metadata = self.prepare_mm_sampling_metadata(
                sampling_metadata)
            next_mm_tokens = self.sampler(audio_logits, mm_sampling_metadata)
            next_mm_tokens.sampled_token_ids = next_mm_tokens.sampled_token_ids.reshape(
                -1, self.audio_num_codebooks)

            if next_mm_tokens.sampled_token_ids.numel() > 0:
                self._last_audio_tokens = next_mm_tokens.sampled_token_ids.clone()

            bos_val = int(self.config.audio_stream_bos_id)
            eos_val = int(self.config.audio_stream_eos_id)
            text_eos = int(self.config.audio_eos_token_id)

            if next_mm_tokens.sampled_token_ids.numel() > 0:
                bos_tensor = torch.tensor(bos_val,
                                          device=next_mm_tokens.sampled_token_ids.device)
                eos_tensor = torch.tensor(eos_val,
                                          device=next_mm_tokens.sampled_token_ids.device)
                invalid_mask = (
                    (next_mm_tokens.sampled_token_ids < 0)
                    | (next_mm_tokens.sampled_token_ids >= self.audio_codebook_size))
                special_mask = (next_mm_tokens.sampled_token_ids == bos_tensor) \
                    | (next_mm_tokens.sampled_token_ids == eos_tensor)
                sanitized = invalid_mask & (~special_mask)
                if sanitized.any():
                    next_mm_tokens.sampled_token_ids[sanitized] = bos_val

            for i in range(n_reqs):
                if audio_generation_mode[i] == 1:
                    next_tokens.sampled_token_ids[i] = self.config.audio_out_token_idx
                    next_mm_tokens.sampled_token_ids[i] = bos_val
                elif audio_generation_mode[i] == 2:
                    next_tokens.sampled_token_ids[i] = self.config.audio_out_token_idx
                    num_audio_delay = multimodal_metadata.num_audio_delays[i]
                    num_audio_eos = multimodal_metadata.num_audio_eos[i]

                    if num_audio_delay < self.audio_num_codebooks:
                        next_mm_tokens.sampled_token_ids[i][num_audio_delay:] = bos_val

                    if num_audio_eos < self.audio_num_codebooks:
                        eos_idx = torch.where(
                            next_mm_tokens.sampled_token_ids[i] == eos_val)[0]
                        if eos_idx.numel() > 0:
                            last_eos_index = int(eos_idx[-1].item())
                            # 0.8.3 behavior: promote all EARLIER codebooks to EOS
                            # once a later codebook hits EOS. This causes the
                            # scheduler to accumulate num_audio_eos and terminate
                            # promptly while leaving higher codebooks untouched
                            # for continued generation.
                            if last_eos_index + 1 < self.audio_num_codebooks:
                                next_mm_tokens.sampled_token_ids[i][last_eos_index + 1:] = eos_val
                            # if last_eos_index > 0:
                            #     next_mm_tokens.sampled_token_ids[i][:last_eos_index] = eos_val
                    elif num_audio_eos == self.audio_num_codebooks:
                        next_tokens.sampled_token_ids[i] = text_eos
                        next_mm_tokens.sampled_token_ids[i] = -1
                    else:
                        # Text-driven stop: if text head sampled <|audio_eos|> and
                        # we've produced at least HIGGS_TTS_MIN_FRAMES frames, honor it.
                        try:
                            import os as _os
                            if _orig_text_tokens is not None and \
                               int(_orig_text_tokens[i].item()) == text_eos:
                                min_frames = int(_os.getenv("HIGGS_TTS_MIN_FRAMES", "0"))
                                out_ids = sampling_metadata.output_token_ids[i]
                                ao = int(self.config.audio_out_token_idx)
                                frames_emitted = sum(1 for t in out_ids if t == ao) \
                                    if isinstance(out_ids, (list, tuple)) else 0
                                if frames_emitted >= min_frames:
                                    next_tokens.sampled_token_ids[i] = text_eos
                                    next_mm_tokens.sampled_token_ids[i] = -1
                        except Exception:
                            pass
                else:
                    if next_mm_tokens is not None:
                        next_mm_tokens.sampled_token_ids[i] = -1

        return next_tokens, next_mm_tokens

    def prepare_mm_sampling_metadata(
            self, sampling_metadata: SamplingMetadata) -> SamplingMetadata:
        mm_sampling_metadata = copy.copy(sampling_metadata)
        if sampling_metadata.top_k is not None:
            mm_sampling_metadata.top_k = sampling_metadata.top_k.clip(
                max=self.audio_codebook_size).repeat_interleave(
                    self.audio_num_codebooks)
        if sampling_metadata.top_p is not None:
            mm_sampling_metadata.top_p = sampling_metadata.top_p.repeat_interleave(
                self.audio_num_codebooks)
        if sampling_metadata.temperature is not None:
            mm_sampling_metadata.temperature = sampling_metadata.temperature.repeat_interleave(
                self.audio_num_codebooks)
        return mm_sampling_metadata

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loaded_counters = {"audio_lm_head": 0, "text_lm_head": 0}
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.config.audio_adapter_type == "stack":
                audio_param_names = [
                    "audio_attn", "audio_input_layernorm", "audio_mlp",
                    "audio_post_attention_layernorm"
                ]
                if any(p in name for p in audio_param_names):
                    continue
            for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in name:
                    name = name.replace(key_to_modify, new_key)

            if (
                    self.quant_config is not None
                    and (scale_name := self.quant_config.get_cache_scale(name))
            ):  # Loading kv cache scales for compressed-tensors quantization
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = loaded_weight[0]
                weight_loader(param, loaded_weight)
                continue

            if "audio_tower" in name:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                if name.startswith("audio_lm_head"):
                    loaded_counters["audio_lm_head"] += 1
                elif name.startswith("text_lm_head"):
                    loaded_counters["text_lm_head"] += 1

                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                if name.startswith("audio_lm_head"):
                    loaded_counters["audio_lm_head"] += 1
                elif name.startswith("text_lm_head"):
                    loaded_counters["text_lm_head"] += 1

        try:
            logger.info("Loaded lm_head params: audio=%s, text=%s",
                        loaded_counters["audio_lm_head"],
                        loaded_counters["text_lm_head"])
            if hasattr(self, "audio_lm_head") and hasattr(self.audio_lm_head,
                                                           "weight"):
                w = self.audio_lm_head.weight
                logger.info("audio_lm_head.weight: mean=%.6f std=%.6f",
                            float(w.mean().item()), float(w.std().item()))
            # Probe text embedding weights as a sanity check; zeroed embeddings
            # will collapse hidden states to zero across the stack.
            try:
                if hasattr(self, "embed_tokens") and \
                   hasattr(self.embed_tokens, "weight"):
                    ew = self.embed_tokens.weight
                    logger.info("embed_tokens.weight: mean=%.6f std=%.6f",
                                float(ew.mean().item()), float(ew.std().item()))
            except Exception:
                pass
            if hasattr(self, "audio_decoder_proj") and \
               getattr(self.audio_decoder_proj, "_audio_decoder_proj_num_layers", 0) > 0:
                # Probe a couple of weights for sanity.
                try:
                    layer0 = self.audio_decoder_proj.layers[0]
                    qkv = getattr(layer0.self_attn.qkv_proj, "weight", None)
                    mlp_up = getattr(layer0.mlp.gate_up_proj, "weight", None)
                    if qkv is not None:
                        logger.info("audio_decoder_proj.layer0.qkv_proj.weight: mean=%.6f std=%.6f",
                                    float(qkv.mean().item()), float(qkv.std().item()))
                    if mlp_up is not None:
                        logger.info("audio_decoder_proj.layer0.gate_up_proj.weight: mean=%.6f std=%.6f",
                                    float(mlp_up.mean().item()), float(mlp_up.std().item()))
                except Exception:
                    pass
        except Exception:
            pass

        # Optional: tie audio_codebook_embeddings to audio_lm_head.
        # 0.8.3 parity: do NOT tie by default. Enable only if explicitly
        # requested via HIGGS_TTS_ENABLE_TIE_AUDIO_EMBED=1.
        try:
            if hasattr(self, "audio_codebook_embeddings") and \
               hasattr(self, "audio_lm_head") and \
               hasattr(self.audio_codebook_embeddings, "weight") and \
               hasattr(self.audio_lm_head, "weight"):
                emb_w = self.audio_codebook_embeddings.weight
                head_w = self.audio_lm_head.weight
                if emb_w.shape == head_w.shape:
                    try:
                        import os as _os
                        enable_tie = _os.getenv("HIGGS_TTS_ENABLE_TIE_AUDIO_EMBED",
                                                 "0") == "1"
                    except Exception:
                        enable_tie = False
                    if enable_tie:
                        with torch.no_grad():
                            emb_w.copy_(head_w)
                        try:
                            emb_std = float(emb_w.std().detach().cpu().item())
                            head_std = float(head_w.std().detach().cpu().item())
                            logger.info(
                                "audio_codebook_embeddings tied to audio_lm_head (std: emb=%.6f head=%.6f)",
                                emb_std, head_std)
                        except Exception:
                            pass
        except Exception:
            # Best-effort; do not fail weight loading if tying fails
            pass
