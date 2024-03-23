from typing import Optional

from torch import Tensor
from transformers import ErniePreTrainedModel, PretrainedConfig, ErnieModel
import torch


class UIE(ErniePreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super(UIE, self).__init__(config)
        # when converting paddle model to pytorch, the parameters are start with encoder.
        # so here we must use self.encoder, and encoder can't be changed
        self.encoder = ErnieModel(config)
        self.linear_start = torch.nn.Linear(config.hidden_size, 1)
        self.linear_end = torch.nn.Linear(config.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(
            self,
            input_ids: Optional[Tensor] = None,
            token_type_ids: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            return_dict: Optional[Tensor] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        sequence_output = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=return_dict,
        )
        start_logits = self.linear_start(sequence_output.last_hidden_state)
        start_logits = torch.squeeze(start_logits, -1)
        start_prob = self.sigmoid(start_logits)
        end_logits = self.linear_end(sequence_output.last_hidden_state)
        end_logits = torch.squeeze(end_logits, -1)
        end_prob = self.sigmoid(end_logits)
        return start_prob, end_prob
