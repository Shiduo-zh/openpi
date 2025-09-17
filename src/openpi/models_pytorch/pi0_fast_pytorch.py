import dataclasses
from typing import Any

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.models.auto import CONFIG_MAPPING

import openpi.models.gemma as _gemma
from openpi.models import model as _model
from openpi.models.pi0_fast import Pi0FASTConfig
from openpi.models_pytorch import preprocessing_pytorch as _preprocessing


def make_attn_mask(input_mask, mask_ar):
    """
    Creates a 2D attention mask for sequence processing.

    This function is a PyTorch adaptation of a JAX utility used in the original
    pi0_fast model. It constructs a causal attention mask that allows tokens to
    attend to preceding tokens based on an auto-regressive mask (`mask_ar`).
    This is essential for creating prefix-LM or causal attention patterns.

    Args:
        input_mask (torch.Tensor): A boolean tensor of shape `(B, N)` where `True`
            indicates valid input tokens and `False` indicates padding.
        mask_ar (torch.Tensor): A boolean tensor of shape `(B, N)` that controls
            the auto-regressive behavior. A `True` value at a position indicates
            the start of a new causal block.

    Returns:
        torch.Tensor: A boolean tensor of shape `(B, N, N)` representing the
            final attention mask, where `True` allows attention between tokens.
    """
    if mask_ar.shape != input_mask.shape:
        mask_ar = mask_ar.expand_as(input_mask)

    cumsum = torch.cumsum(mask_ar.long(), dim=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] & input_mask[:, :, None]

    return attn_mask & valid_mask


class PI0FastPytorch(nn.Module):
    def __init__(self, config: Pi0FASTConfig, use_adarms = None):
        """
        Initializes the PI0FastPytorch model.

        This constructor builds a `PaliGemmaForConditionalGeneration` model from
        scratch based on the provided `Pi0FASTConfig`. It meticulously maps
        parameters from the project's internal configuration to the Hugging
        Face `transformers` configuration, ensuring the architecture is
        identical to the JAX counterpart. This allows for correct loading of
        local pre-trained weights.

        Args:
            config (Pi0FASTConfig): The configuration object containing model
                parameters and settings.
        """
        super().__init__()
        self.config = config
        
        vlm_config = _gemma.get_config(config.paligemma_variant)
        # 1. Get the detailed JAX-style Gemma configuration
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        # 6. Instantiate the model from the fully-specified configuration.
        # This creates the correct architecture with uninitialized weights.
        self.paligemma = transformers.PaliGemmaForConditionalGeneration(
            config=vlm_config_hf
        )
        
        # 7. Set the model's precision based on the training configuration.
        if config.dtype == "bfloat16":
            self.paligemma = self.paligemma.to(torch.bfloat16)

    def embed_inputs(
        self, obs: _model.Observation
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Embeds images and the combined text-action prompt into a single sequence.

        This method processes multiple images and a single tokenized prompt.
        Crucially, for pi0_fast, `obs.tokenized_prompt` is expected to already
        contain the concatenation of the language instruction and the tokenized
        action sequence, as prepared by the data loader.

        Args:
            obs (_model.Observation): An observation object containing images,
                image masks, and the combined tokenized prompt data.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - The concatenated token embeddings (`B, S, E`).
                - The input mask (`B, S`), indicating valid tokens.
                - The auto-regressive mask (`B, S`), for controlling attention flow.
        """
        token_embeddings = []
        input_mask = []
        ar_mask = []

        # 1. Embed and append image tokens
        for name in obs.images:
            image_token_embeddings = self.paligemma.model.get_image_features(
                obs.images[name]
            )
            token_embeddings.append(image_token_embeddings)

            current_input_mask = einops.repeat(
                obs.image_masks[name],
                "b -> b s",
                s=image_token_embeddings.shape[1],
            )
            input_mask.append(current_input_mask)

            ar_mask.append(
                torch.zeros_like(current_input_mask, dtype=torch.bool)
            )

        # 2. Embed and append the combined prompt (text + action) tokens
        text_action_embeddings = self.paligemma.language_model.embed_tokens(
            obs.tokenized_prompt
        )
        token_embeddings.append(text_action_embeddings)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask.append(obs.token_ar_mask.bool())

        # 3. Concatenate all parts into single tensors
        final_embeddings = torch.cat(token_embeddings, dim=1)
        final_input_mask = torch.cat(input_mask, dim=1)
        final_ar_mask = torch.cat(ar_mask, dim=1)

        return final_embeddings, final_input_mask, final_ar_mask

    def forward(self, observation: _model.Observation, actions: torch.Tensor = None) -> torch.Tensor:
        """
        Performs a forward pass and computes the language modeling loss.

        This method treats the combined text and action tokens as a single language
        modeling task. It computes the cross-entropy loss for predicting the next
        token in the sequence. The `token_loss_mask` (provided in the observation)
        ensures that loss is only computed on the action tokens.

        Args:
            observation (_model.Observation): The input observation, containing images
                and the combined text-action prompt.
            actions (torch.Tensor, optional): Unused for this model, but included for
                API compatibility with the trainer.

        Returns:
            torch.Tensor: A tensor of shape `(B,)` containing the loss for each
                example in the batch. The training script will then take the mean.
        """
        # 1. Preprocess the observation (e.g., image augmentations)
        observation = _preprocessing.preprocess_observation_pytorch(
            observation, train=self.training
        )

        # 2. Get embeddings and masks for the full input sequence
        input_token_embeddings, input_mask, ar_mask = self.embed_inputs(observation)

        # 3. Create the 2D attention mask from the 1D masks
        attn_mask = make_attn_mask(input_mask, ar_mask)

        # 4. Prepare targets and loss mask
        # The target for each token is the *next* token in the sequence.
        targets = observation.tokenized_prompt[:, 1:]
        loss_mask = observation.token_loss_mask[:, 1:]

        # 5. Run the transformer to get pre-logits (last hidden state)
        # We don't feed the last token, as it has no target to predict.
        pre_logits = self.paligemma.model.language_model(
            inputs_embeds=input_token_embeddings[:, :-1],
            attention_mask=attn_mask[:, None, :-1, :-1],  # Add head dimension for HF model
        ).last_hidden_state

        # 6. Memory Optimization: Apply the final projection head (`lm_head`) only
        # on the hidden states that correspond to our target tokens.
        num_targets = targets.shape[1]
        pre_logits_for_loss = pre_logits[:, -num_targets:]
        logits = self.paligemma.lm_head(pre_logits_for_loss)

        # 7. Compute Cross-Entropy Loss
        vocab_size = logits.shape[-1]
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
            reduction="none",
        )
        loss = loss.view(targets.shape)  # Reshape back to (B, S)

        # 8. Apply the loss mask and normalize by the number of target tokens
        loss = loss * loss_mask
        final_loss_per_example = loss.sum(dim=-1) / loss_mask.sum(dim=-1).clamp(min=1)

        return final_loss_per_example