import torch
import wandb
import random
import numpy as np
import torch.nn as nn
import lightning.pytorch as L

from torchinfo import summary
from eval_functions import compute_poliphony_metrics
from smt_model import SMTConfig
from smt_model import SMTModelForCausalLM

class SMT_Trainer(L.LightningModule):
    def __init__(self, maxh: int, maxw: int, maxlen: int, out_categories: int,
                 padding_token: int, in_channels: int, w2i: dict, i2w: dict,
                 d_model: int = 256, dim_ff: int = 256, num_dec_layers: int = 8,
                 # NEW: Add dropout and num_vit_layers to trainer's init
                 dropout: float = 0.1,
                 num_vit_layers: int = 6,
                 num_attn_heads: int = 8, # Make this explicit and match SMTConfig name
                 learning_rate: float = 1e-4 # Add learning rate as a parameter for configurability
                 ):
        super().__init__()

        # Pass all relevant parameters to SMTConfig
        self.config = SMTConfig(
            maxh=maxh,
            maxw=maxw,
            maxlen=maxlen,
            out_categories=out_categories,
            padding_token=padding_token,
            in_channels=in_channels,
            w2i=w2i,
            i2w=i2w,
            d_model=d_model,
            dim_ff=dim_ff,
            num_attn_heads=num_attn_heads, # Corrected name from 'attn_heads'
            num_dec_layers=num_dec_layers,
            dropout=dropout, # Pass dropout from trainer's init
            num_vit_layers=num_vit_layers # Pass num_vit_layers from trainer's init
            # Removed use_flash_attn=True as it's not a parameter in the SMTConfig provided
        )

        self.model = SMTModelForCausalLM(self.config)
        self.padding_token = padding_token
        self.learning_rate = learning_rate # Store learning rate

        self.preds = []
        self.grtrs = []

        self.save_hyperparameters() # Saves all __init__ arguments as hyperparameters

        # Updated summary input_size for decoder_input to be (1, maxlen-1) as it's typically shifted
        # And ensure the data types are correct
        summary(self.model, input_size=[(1, in_channels, self.config.maxh, self.config.maxw),
                                        (1, self.config.maxlen)],
                dtypes=[torch.float, torch.long])


    def configure_optimizers(self):
        # Collect parameters from all *actual* sub-modules of the encoder
        # SMTModelForCausalLM now has convnext_backbone, vit_projection, vit_encoder, and decoder
        optimizer_params = list(self.model.convnext_backbone.parameters()) + \
                           list(self.model.vit_projection.parameters()) + \
                           list(self.model.vit_encoder.parameters()) + \
                           list(self.model.decoder.parameters())

        return torch.optim.Adam(optimizer_params, lr=self.learning_rate, amsgrad=False)

    def forward(self, input, last_preds, labels=None): # Added labels for consistent forward method
        # It's better to call the model's main forward method directly,
        # which handles both encoder and decoder.
        return self.model(encoder_input=input, decoder_input=last_preds, labels=labels)

    def training_step(self, batch, batch_idx): # batch_idx is a required arg for lightning
        x, di, y = batch
        # Call the trainer's forward method, which in turn calls model's forward
        outputs = self(input=x, last_preds=di[:, :-1], labels=y)
        loss = outputs.loss
        self.log('loss', loss, on_epoch=True, batch_size=1, prog_bar=True)

        return loss


    def validation_step(self, val_batch, batch_idx): # batch_idx is a required arg for lightning
        x, _, y = val_batch # Note: di is ignored here for validation, but for predict it's not needed.
        # Ensure 'input' here is the full image, and 'y' is the ground truth sequence for CER/SER
        predicted_sequence_ids, _ = self.model.predict(input=x) # model.predict returns sequence of IDs or strings
        # Convert predicted IDs to strings using i2w from config
        predicted_sequence_str = [self.config.i2w[token_id] for token_id in predicted_sequence_ids]

        dec = "".join(predicted_sequence_str)
        dec = dec.replace("<t>", "\t")
        dec = dec.replace("<b>", "\n")
        dec = dec.replace("<s>", " ")

        gt = "".join([self.config.i2w[token.item()] for token in y.squeeze(0)[:-1]])
        gt = gt.replace("<t>", "\t")
        gt = gt.replace("<b>", "\n")
        gt = gt.replace("<s>", " ")

        self.preds.append(dec)
        self.grtrs.append(gt)

    def on_validation_epoch_end(self, metric_name="val") -> None:
        

        cer, ser, ler = compute_poliphony_metrics(self.preds, self.grtrs)

        random_index = random.randint(0, len(self.preds)-1)
        predtoshow = self.preds[random_index]
        gttoshow = self.grtrs[random_index]
        print(f"\n[Prediction] - {predtoshow}") # Added \n for better readability
        print(f"[GT] - {gttoshow}")

        self.log(f'{metric_name}_CER', cer, on_epoch=True, prog_bar=True)
        self.log(f'{metric_name}_SER', ser, on_epoch=True, prog_bar=True)
        self.log(f'{metric_name}_LER', ler, on_epoch=True, prog_bar=True)

        self.preds = []
        self.grtrs = []

        return ser # Returning a metric is common practice for epoch end callbacks

    def test_step(self, test_batch, batch_idx): # batch_idx is a required arg for lightning
        return self.validation_step(test_batch, batch_idx)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end("test")