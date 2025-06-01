import torch
import warnings
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger
from typing import Optional, Tuple

from .configuration_smt import SMTConfig

from transformers import ConvNextConfig, ConvNextModel, ViTConfig, PreTrainedModel
from transformers.models.vit.modeling_vit import ViTEncoder 
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class PositionalEncoding2D(nn.Module):

    def __init__(self, dim, h_max, w_max):
        super(PositionalEncoding2D, self).__init__()
        self.h_max = h_max
        self.max_w = w_max
        self.dim = dim

        self.pe: torch.Tensor
        self.register_buffer("pe", torch.zeros((dim, h_max, w_max), requires_grad=False), persistent=False)

        div = torch.exp(-torch.arange(0., dim // 2, 2) / dim * torch.log(torch.tensor(1e+4))).unsqueeze(1)
        w_pos = torch.arange(0., w_max) * div
        h_pos = torch.arange(0., h_max) * div
        self.pe[:dim // 2:2] = torch.sin(h_pos).unsqueeze(2).repeat(1, 1, w_max)
        self.pe[1:dim // 2:2] = torch.cos(h_pos).unsqueeze(2).repeat(1, 1, w_max)
        self.pe[dim // 2::2] = torch.sin(w_pos).unsqueeze(1).repeat(1, h_max, 1)
        self.pe[dim // 2 + 1::2] = torch.cos(w_pos).unsqueeze(1).repeat(1, h_max, 1)

    def forward(self, x):
        """
        Add 2D positional encoding to x
        x: Tensor(B, C, H, W)
        returns:
        - Tensor(B, C, H, W)
        """
        return x + self.get_pe_by_size(x.size(-2), x.size(-1))

    def get_pe_by_size(self, h, w):
        return self.pe[:, :h, :w]


class PositionalEncoding1D(nn.Module):

    def __init__(self, dim, len_max):
        super(PositionalEncoding1D, self).__init__()
        self.len_max = len_max
        self.dim = dim
        self.pe: torch.Tensor
        self.register_buffer("pe", torch.zeros((len_max, dim), requires_grad=False), persistent=False)

        div = torch.exp(-torch.arange(0., dim, 2) / dim * torch.log(torch.tensor(1e+4)))
        l_pos = torch.arange(0., len_max).unsqueeze(1) * div
        self.pe[:, ::2] = torch.sin(l_pos)
        self.pe[:, 1::2] = torch.cos(l_pos)

    def forward(self, x, start = 0):
        """
        Add 1D positional encoding to x
        x: Tensor(B, L, C)
        start: index for x[:, 0, :]
        returns:
        - Tensor(B, L, C)
        """
        if isinstance(start, int):
            return x + self.pe[start:start+x.size(-2)]
        else:
            for i in range(x.size(0)):
                x[i] = x[i] + self.pe[start[i]:start[i]+x.size(-2)]
            return x

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model:int, num_heads:int, dropout: float = 0.1,
                 bias:bool = True):
        super().__init__()

        assert(d_model % num_heads == 0), logger.error("The embeddings depth must be divisible by the number of heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = self.d_head ** -0.5

        self.has_flash_attn = hasattr(F, 'scaled_dot_product_attention')
        if not self.has_flash_attn:
            logger.warning("This program cannot run Flash Attention, for optimal computing, check your GPU driver and your PyTorch version")

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self._init_parameters()

        self.dropout = nn.Dropout(dropout)


    def _init_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)


    def _split_heads(self, tensor:torch.Tensor) -> torch.Tensor:
        """Split the heads and put them into a batch-first format."""
        batch_size, seq_len, _ = tensor.shape
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.d_head)
        return tensor.transpose(1,2) # (batch_size, num_heads, seq_len, d_head)


    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Merge heads and transpose back to batch-first format."""
        batch_size = tensor.shape[0]
        tensor = tensor.transpose(1, 2)
        return tensor.reshape(batch_size, -1, self.d_model).contiguous()


    def compute_flash_attn(self,
                           q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           attn_mask:Optional[torch.Tensor] = None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if attn_mask is not None:
                attn_mask = ~attn_mask
            return F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal = False, scale=self.scale
            )

    def _compute_regular_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None
    ):
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_weights.masked_fill_(~attn_mask, float('-inf'))
            else:
                attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights.masked_fill_(
                ~key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_output = attn_probs @ v

        return attn_output, attn_weights

    def forward(self,
                query: torch.Tensor, key: Optional[torch.Tensor] = None, value:Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if key is None and value is None:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        else:
            if key is None or value is None:
                raise ValueError("Both key and value must be provided for cross-attention")

            # We manually multiply each weight section from qkv_proj to their respective vectors
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

        # Split the heads in q, k and v
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        use_flash_attn = self.has_flash_attn and not need_weights

        if use_flash_attn:
            attn_output = self.compute_flash_attn(q, k, v, attn_mask=attn_mask)
            attn_weights = None
        else:
            attn_output, attn_weights = self._compute_regular_attention(q, k, v, key_padding_mask, attn_mask)

        output = self._merge_heads(attn_output)
        output = self.out_proj(output)

        if need_weights:
            return output, attn_weights

        return output, None

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_ff:int,
                 dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)

        self.activation = nn.ReLU() if activation.lower() == "relu" else nn.GELU()

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model)
        )

        self.norm_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(3)])

    def forward(self,
                x: torch.Tensor,
                encoder_output_key: torch.Tensor,
                encoder_output_value: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                return_weights=False):


        attn_output, self_weights = self.self_attn(
            query=x, key=None, value=None,
            key_padding_mask=tgt_key_padding_mask, attn_mask=tgt_mask
        )

        x = x + self.dropout_layers[0](attn_output)
        x = self.norm_layers[0](x)

        attn_output, cross_weights = self.cross_attn(
            query=x,
            key=encoder_output_key,
            value=encoder_output_value,
            key_padding_mask=memory_key_padding_mask
        )

        x = x + self.dropout_layers[1](attn_output)
        x = self.norm_layers[1](x)

        ffn_output = self.ffn(x)
        x = x + self.dropout_layers[2](ffn_output)
        x = self.norm_layers[2](x)

        if return_weights:
            return x, [self_weights, cross_weights]

        return x, None

class DecoderStack(nn.Module):
    def __init__(self, num_dec_layers:int,
                 d_model:int, dim_ff:int, num_heads:int,
                 dropout:float):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  num_heads=num_heads,
                                                  dim_ff=dim_ff) for _ in range(num_dec_layers)])
    def forward(self,
                x:torch.Tensor, encoder_output_2D:torch.Tensor, encoder_output_raw:torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                return_weights=False):

        output = x
        all_weights = {
            "self_attn": [],
            "cross_attn": []
        }

        for i, dec_layer in enumerate(self.layers):
            output, weights = dec_layer(x=output,
                                        encoder_output_key=encoder_output_2D,
                                        encoder_output_value=encoder_output_raw,
                                        tgt_mask=tgt_mask,
                                        tgt_key_padding_mask=tgt_key_padding_mask,
                                        memory_key_padding_mask=memory_key_padding_mask,
                                        return_weights=return_weights)
            if return_weights:
                all_weights["self_attn"].append(weights[0])
                all_weights["cross_attn"].append(weights[1])

        if return_weights:
            return output, all_weights

        return output, None

class Decoder(nn.Module):
    def __init__(self, num_dec_layers:int,
                 d_model:int, dim_ff:int, n_heads:int,
                 max_seq_length:int, out_categories:int, dropout:float = 0.1):

        super().__init__()

        self.decoder = DecoderStack(num_dec_layers=num_dec_layers,
                                    d_model=d_model, dim_ff=dim_ff, num_heads=n_heads,
                                    dropout=dropout)

        self.embedding = nn.Embedding(num_embeddings=out_categories, embedding_dim=d_model)

        self.position_encoding = PositionalEncoding1D(dim=d_model, len_max=max_seq_length)

        self.vocab_projection = nn.Linear(in_features=d_model, out_features=out_categories)

        self.dropout = nn.Dropout(dropout)

    def forward(self, decoder_input:torch.Tensor,
                encoder_output_2D:torch.Tensor, encoder_output_raw:torch.Tensor,
                tgt_mask:Optional[torch.Tensor] = None,
                tgt_key_padding_mask:Optional[torch.Tensor] = None,
                memory_key_padding_mask:Optional[torch.Tensor] = None,
                return_weights = False):

        decoder_input = self.embedding(decoder_input)
        decoder_input = self.position_encoding(decoder_input)

        output, weights = self.decoder(x=decoder_input, encoder_output_2D=encoder_output_2D,
                                       encoder_output_raw=encoder_output_raw,
                                       tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                                       memory_key_padding_mask=memory_key_padding_mask,
                                       return_weights=return_weights)

        output = self.dropout(output)

        predictions = self.vocab_projection(output)

        return output, predictions, weights

class SMTOutput(CausalLMOutputWithCrossAttentions):
    """Output wrapper for the SMT"""

class SMTModelForCausalLM(PreTrainedModel):
    config_class = SMTConfig

    def __init__(self, config: SMTConfig):
        super().__init__(config)

        # 1. ConvNeXt as initial feature extractor (CNN backbone)
        conv_next_stages = 3 # Number of stages in ConvNeXt, can be configured
        convnext_config = ConvNextConfig(
            num_channels=config.in_channels,
            num_stages=conv_next_stages,
            hidden_sizes=[96, 192, 384], # Example output channels for each stage
            depths=[3, 3, 9] # Example depths for each stage
        )
        self.convnext_backbone = ConvNextModel(convnext_config)

        # Determine the output channels of ConvNeXt's last stage
        convnext_output_channels = convnext_config.hidden_sizes[-1] # e.g., 384

        # Calculate image downsampling factor by ConvNeXt
        # ConvNeXt's patch_embed has stride 4, and each subsequent stage (from stage 1 onwards)
        # has a downsample layer with stride 2.
        # Total reduction = 4 * (2^(num_stages - 1)) for num_stages > 0
        # If num_stages = 3, reduction = 4 * (2^(3-1)) = 4 * 4 = 16
        self.width_reduction = 4 * (2 ** (conv_next_stages - 1)) if conv_next_stages > 0 else 1
        self.height_reduction = 4 * (2 ** (conv_next_stages - 1)) if conv_next_stages > 0 else 1

        # 2. Linear projection layer to match ConvNeXt output channels to ViT's hidden_size (d_model)
        self.vit_projection = nn.Linear(convnext_output_channels, config.d_model)

        # 3. 2D Positional Encoding for the projected ConvNeXt features
        # Applied before flattening for the ViT encoder
        self.pos2D = PositionalEncoding2D(dim=config.d_model,
                                          h_max=config.maxh // self.height_reduction,
                                          w_max=config.maxw // self.width_reduction)

        # 4. Vision Transformer Encoder
        # The ViTEncoder itself is just the stack of transformer layers, it expects pre-processed embeddings.
        vit_config = ViTConfig(
            image_size=1, # Dummy, not used as we feed a token sequence
            patch_size=1, # Dummy
            num_channels=config.d_model, # Dummy, refers to the feature dimension after projection
            hidden_size=config.d_model, # The actual embedding dimension for ViT tokens
            num_hidden_layers=config.num_vit_layers, # From SMTConfig
            num_attention_heads=config.num_attn_heads,
            intermediate_size=config.dim_ff, # Feed-forward dimension for ViT blocks
            hidden_act="gelu",
            dropout=config.dropout,
            attention_probs_dropout_prob=config.dropout,
            # We don't need a class token for this feature extraction setup
            # We don't need the pooling layer as it's not for classification here
            add_pooling_layer=False
        )
        self.vit_encoder = ViTEncoder(vit_config) # ViTEncoder handles self-attention and FFN

        # 5. Decoder (remains unchanged)
        self.decoder = Decoder(num_dec_layers=config.num_dec_layers,
                               d_model=config.d_model, dim_ff=config.dim_ff, n_heads=config.num_attn_heads,
                               max_seq_length=config.maxlen, out_categories=config.out_categories)

        self.padding_token = config.padding_token
        self.loss = nn.CrossEntropyLoss(ignore_index=config.padding_token)
        self.w2i = config.w2i
        self.i2w = config.i2w
        self.maxlen = int(config.maxlen)

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the input image through ConvNeXt and then the Vision Transformer encoder.

        Args:
            x (torch.Tensor): Input image tensor (B, C_in, H_img, W_img).

        Returns:
            torch.Tensor: Encoded features from the Vision Transformer (B, N_patches, D_model).
        """
        # 1. ConvNeXt backbone to extract initial features
        # Output shape: (B, convnext_output_channels, H_reduced, W_reduced)
        convnext_features = self.convnext_backbone(pixel_values=x).last_hidden_state

        # 2. Project ConvNeXt channels to d_model for ViT compatibility
        # Linear layer expects the last dimension as features, so permute first:
        # (B, C, H, W) -> (B, H, W, C) -> (B, H, W, D_model)
        convnext_features_proj = self.vit_projection(convnext_features.permute(0, 2, 3, 1))
        # Permute back to (B, D_model, H, W) for PositionalEncoding2D
        convnext_features_proj = convnext_features_proj.permute(0, 3, 1, 2)

        # 3. Add 2D positional encoding to the projected features
        # Output shape: (B, D_model, H_reduced, W_reduced)
        features_with_pe = self.pos2D(convnext_features_proj)

        # 4. Flatten the 2D features into a sequence of tokens for the ViT encoder
        # Output shape: (B, N_patches, D_model) where N_patches = H_reduced * W_reduced
        vit_input_sequence = features_with_pe.flatten(start_dim=2).permute(0, 2, 1)

        # 5. Pass through the Vision Transformer encoder layers
        # The ViTEncoder returns a BaseModelOutput, access `last_hidden_state`.
        # Output shape: (B, N_patches, D_model)
        vit_encoded_features = self.vit_encoder(hidden_states=vit_input_sequence).last_hidden_state

        return vit_encoded_features

    def forward_decoder(
        self,
        encoder_output: torch.Tensor, # This is `vit_encoded_features` now
        last_predictions: torch.Tensor,
        get_weights: bool = False
    ) -> SMTOutput:
        """
        Forward pass for the decoder.

        Args:
            encoder_output (torch.Tensor): Encoded features from the Vision Transformer (B, N_patches, D_model).
            last_predictions (torch.Tensor): Previous token predictions for decoder input.
            get_weights (bool): Whether to return attention weights.

        Returns:
            SMTOutput: Decoder output.
        """
        # `encoder_output` is already the `vit_encoded_features` from `forward_encoder`.
        # It's a 3D tensor (B, N_patches, D_model) and contains positional information
        # implicitly learned by the ViT's self-attention and explicitly added by `pos2D`.
        # Therefore, we can use it for both key and value in the cross-attention.
        encoder_output_2D = encoder_output  # Used as key for cross-attention
        encoder_features = encoder_output   # Used as value for cross-attention

        # Generate masks for decoder self-attention and cross-attention
        batch_size, _ = last_predictions.size()
        key_target_mask = self._generate_token_mask(
            [lp.shape[0] for lp in last_predictions], # This line needs `lp` to be actual lengths for batching
            last_predictions.size(),
            device=last_predictions.device
        )
        causal_mask = self._generate_causal_mask(last_predictions.size(1), last_predictions.device)

        # Note: The original code commented "[TODO] This only works with one sample per batch" for
        # `memory_key_padding_mask`. If your `encoder_output` (N_patches) length can vary within a batch
        # (e.g., variable input image sizes), you would need to generate and pass a mask here.
        # Assuming fixed input image sizes (maxh, maxw) means N_patches is constant.
        memory_key_padding_mask = None # For fixed input image size, no padding mask is needed for encoder output

        output, predictions, weights = self.decoder(
            decoder_input=last_predictions,
            encoder_output_2D=encoder_output_2D,
            encoder_output_raw=encoder_features,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=key_target_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            return_weights=get_weights
        )

        return SMTOutput(
            logits=predictions,
            hidden_states=output,
            attentions=None if weights is None else weights["self_attn"],
            cross_attentions=None if weights is None else weights["cross_attn"]
        )

    def forward(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor, labels: Optional[torch.Tensor] = None) -> SMTOutput:
        """
        Main forward pass of the SMT model.

        Args:
            encoder_input (torch.Tensor): Input image tensor.
            decoder_input (torch.Tensor): Decoder input sequence (e.g., token IDs including <bos>).
            labels (Optional[torch.Tensor]): Target labels for loss computation (shifted decoder input).

        Returns:
            SMTOutput: Model output including logits and optionally loss.
        """
        x = self.forward_encoder(encoder_input)
        output = self.forward_decoder(x, decoder_input)

        if labels is not None:
            # Shift the labels for causal language modeling loss calculation
            output.loss = self.loss(output.logits.permute(0, 2, 1).contiguous(), labels[:, :-1])

        return output

    @torch.no_grad() # Use @torch.no_grad() decorator for prediction
    def predict(self, input: torch.Tensor, convert_to_str: bool = False, return_weights: bool = False) -> Tuple[list, SMTOutput]:
        """
        Generates a sequence of tokens from an input image.

        Args:
            input (torch.Tensor): Input image tensor (e.g., preprocessed sheet music).
            convert_to_str (bool): If True, convert predicted token IDs to their string representations.
            return_weights (bool): If True, return attention weights from the decoder.

        Returns:
            Tuple[list, SMTOutput]: A tuple containing the generated text sequence (list of tokens)
                                    and the model's output object.
        """
        predicted_sequence = torch.tensor([[self.w2i['<bos>']]], dtype=torch.long, device=input.device)
        encoder_output = self.forward_encoder(input)
        text_sequence = []

        for i in range(self.maxlen - 1): # Maxlen - 1 to account for <bos>
            output = self.forward_decoder(encoder_output=encoder_output,
                                          last_predictions=predicted_sequence,
                                          get_weights=return_weights)

            # Get the predicted token ID for the last position
            predicted_token_id = torch.argmax(output.logits[:, -1, :], dim=-1).item()

            # Append the new token to the sequence
            predicted_sequence = torch.cat([predicted_sequence, torch.tensor([[predicted_token_id]], dtype=torch.long, device=input.device)], dim=1)

            # Break if <eos> token is predicted
            if self.i2w[predicted_token_id] == '<eos>':
                break

            # Add token to result list
            token_representation = self.i2w[predicted_token_id] if convert_to_str else predicted_token_id
            text_sequence.append(token_representation)

        return text_sequence, output

    def _generate_token_mask(self, token_len: list, total_size: Tuple[int, int], device: torch.device) -> torch.Tensor:
        """
        Generates a padding mask for a batch of sequences.

        Args:
            token_len (list): List of actual lengths of sequences in the batch.
            total_size (Tuple[int, int]): (batch_size, max_seq_length).
            device (torch.device): Device to place the mask on.

        Returns:
            torch.Tensor: Boolean mask where False indicates padded tokens.
        """
        batch_size, len_mask = total_size
        mask = torch.zeros((batch_size, len_mask), dtype=torch.bool, device=device)
        for i, length in enumerate(token_len):
            mask[i, :length] = True
        return mask

    def _generate_causal_mask(self, token_len: int, device: torch.device) -> torch.Tensor:
        """
        Generates a causal (look-ahead) mask for self-attention.

        Args:
            token_len (int): Length of the sequence.
            device (torch.device): Device to place the mask on.

        Returns:
            torch.Tensor: Boolean mask where True allows attention and False masks it.
        """
        causal_mask = torch.triu(
            torch.ones(token_len, token_len, dtype=torch.bool, device=device),
            diagonal=1 # Mask out future positions
        )
        return causal_mask
