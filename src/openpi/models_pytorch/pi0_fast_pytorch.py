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

PALIGEMMA_EOS_TOKEN = 1

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


def left_to_right_align(
    x: torch.Tensor, input_mask: torch.Tensor, attn_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts input from left-aligned to right-aligned. PyTorch version.

    This function mirrors the `vmap`ped JAX implementation. It iterates through
    each example in the batch to calculate its true sequence length and then
    rolls the tensors to move padding from the right to the left. This is
    necessary for efficient KV cache prefilling.

    Args:
        x (torch.Tensor): The input embeddings of shape `(B, S, E)`.
        input_mask (torch.Tensor): The boolean input mask of shape `(B, S)`.
        attn_mask (torch.Tensor): The boolean attention mask of shape `(B, S, S)`.

    Returns:
        A tuple containing the right-aligned `x`, `input_mask`, and `attn_mask`.
    """
    out_x = []
    out_input_mask = []
    out_attn_mask = []

    # The JAX version is vmapped, which means it processes each item in the
    # batch independently. A simple loop is the most direct translation.
    for i in range(x.shape[0]):
        # Get the length of the valid sequence for this example.
        seqlen = input_mask[i].sum()

        # Roll the tensors to move the padding to the left.
        # The shift is negative because we want to roll to the left.
        # The number of elements is seq_len, so we shift by total_len - seqlen
        # to align to the right. Or equivalently, shift by -seqlen to move
        # the valid part to the left end and then it will wrap around.
        # Let's verify the JAX logic: jnp.roll(x, -seqlen, axis=0) moves the first
        # `seqlen` elements to the end. This aligns the valid tokens to the right.
        shift = -int(seqlen.item())
        out_x.append(torch.roll(x[i], shifts=shift, dims=0))
        out_input_mask.append(torch.roll(input_mask[i], shifts=shift, dims=0))
        out_attn_mask.append(torch.roll(attn_mask[i], shifts=(shift, shift), dims=(0, 1)))

    return torch.stack(out_x), torch.stack(out_input_mask), torch.stack(out_attn_mask)


def put_along_last_axis(
    arr: torch.Tensor, indices: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """
    PyTorch equivalent of the JAX `put_along_last_axis` utility.

    Places `values` into `arr` at positions specified by `indices` along the
    last axis.

    Args:
        arr (torch.Tensor): The source tensor.
        indices (torch.Tensor): The indices where values should be placed.
            Must be broadcastable to the shape of `values`.
        values (torch.Tensor): The values to be placed into `arr`.

    Returns:
        A new tensor with the values placed at the specified indices.
    """
    # Ensure indices can be broadcast for scatter. It needs to match the dim of values.
    if len(indices.shape) < len(values.shape):
        indices = indices.unsqueeze(-1)
    if arr.ndim != indices.ndim:
        indices = indices.expand_as(arr[..., : indices.shape[-1]])

    return arr.scatter(-1, indices, values.to(arr.device))


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
    
    
    # (The rest of the file remains the same, only sample_actions is updated)

    @torch.no_grad()
    def sample_actions(
        self,
        observation: _model.Observation,
        *,
        max_decoding_steps: int = 256,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """
        Generates action sequences autoregressively based on an observation.

        This method is a PyTorch translation of the original JAX implementation.
        It performs manual decoding without using the high-level `generate`
        method to ensure logic parity.

        The process involves:
        1. Embedding the input observation (images and text prompt).
        2. Right-aligning the sequences to handle padding efficiently.
        3. A 'prefill' forward pass to compute the KV cache for the prompt.
        4. A step-by-step decoding loop to generate one token at a time.

        Args:
            observation (_model.Observation): The input observation.
            max_decoding_steps (int): The maximum number of tokens to generate.
            temperature (float): The temperature for sampling. 0.0 means greedy.

        Returns:
            torch.Tensor: A tensor of shape `(B, max_decoding_steps)` containing
                the generated token IDs for the actions.
        """
        # Ensure model is in evaluation mode
        self.eval()

        # We can infer the device from the model's parameters.
        device = next(self.paligemma.parameters()).device
        model_dtype = next(self.paligemma.parameters()).dtype

        # 1. Preprocess and embed all inputs (prefix)
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=False)
        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_inputs(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)

        # 2. Left-to-right align all input token sequences for efficient prefill
        prefix_token_embeddings, prefix_mask, prefix_attn_mask = left_to_right_align(
            prefix_token_embeddings, prefix_mask, prefix_attn_mask
        )

        prefill_size = prefix_token_embeddings.shape[1]
        prefill_len = prefix_mask.sum(dim=-1)
        
        # 3. Prefill the KV cache with a forward pass of the prefix.
        # The Hugging Face model expects a 4D attention mask [B, 1, S, S].
        # The position IDs are required by PaliGemma.
        prefix_positions = torch.cumsum(prefix_mask.long(), dim=-1) - 1
        prefix_positions = prefix_positions.to(device)

        # **CORRECTED LOGIC: Call language_model, not the top-level model.**
        # This returns hidden states and the KV cache.
        outputs = self.paligemma.language_model(
            inputs_embeds=prefix_token_embeddings,
            attention_mask=prefix_attn_mask[:, None, :, :],
            position_ids=prefix_positions,
            use_cache=True,
        )
        hidden_states = outputs.last_hidden_state
        kv_cache = outputs.past_key_values

        # 4. Prepare for decoding loop
        # The hidden state for the *next* token is from the last valid input token.
        # We gather these last hidden states for each item in the batch.
        last_hidden_state = hidden_states[
            torch.arange(len(prefill_len), device=device), prefill_len - 1, :
        ]
        
        # **CORRECTED LOGIC: Manually apply lm_head.**
        last_logit = self.paligemma.lm_head(last_hidden_state)
        last_logit = last_logit.unsqueeze(1)  # Shape: [B, 1, V]

        batch_size = last_logit.shape[0]
        output_tokens = torch.zeros(
            (batch_size, max_decoding_steps), dtype=torch.long, device=device
        )
        all_eos = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_decoding_steps):
            if all_eos.all():
                break

            # 5. Sample a token from the last logit
            if temperature > 0.0:
                probs = F.softmax(last_logit / temperature, dim=-1)
                # Squeeze to remove the sequence length of 1 for multinomial
                token = torch.multinomial(probs.squeeze(1), num_samples=1)
            else:
                token = torch.argmax(last_logit, dim=-1)

            # Mask out generation for sequences that have already finished
            token[all_eos] = 0

            # Update the output tokens tensor
            output_tokens = put_along_last_axis(
                output_tokens,
                torch.full((batch_size, 1), step, device=device, dtype=torch.long),
                token,
            )

            # Check for early stopping
            has_eos = (token == PALIGEMMA_EOS_TOKEN).squeeze(-1)
            all_eos = all_eos | has_eos

            # 6. Decode one step
            token_embedding = self.paligemma.language_model.embed_tokens(token)

            # Position IDs for the new token
            positions = prefill_len + step
            positions = positions.to(device)
            
            # The transformers implementation of the decoder with KV cache handles
            # the attention mask internally. We just need to provide the new token,
            # its position, and the cache.
            outputs = self.paligemma.language_model(
                inputs_embeds=token_embedding,
                position_ids=positions,
                past_key_values=kv_cache,
                use_cache=True,
            )
            
            # **CORRECTED LOGIC: Manually apply lm_head.**
            last_logit = self.paligemma.lm_head(outputs.last_hidden_state)
            kv_cache = outputs.past_key_values

        return output_tokens