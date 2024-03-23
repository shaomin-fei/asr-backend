from os import path
import lightning as L
from typing import Any
from models.ie.UIEModel import UIE
from transformers import PretrainedConfig

class UIELightningPredictMoudle(L.LightningModule):
    def __init__(self, pretrained_config_path):
        super().__init__()
        pretrained_config=PretrainedConfig.from_json_file(path.join(pretrained_config_path,"config.json"))
        # pretrained_config=UIE.config_class.from_pretrained(pretrained_config_path,return_unused_kwargs=True,)
        self.model =UIE(pretrained_config)

    def forward(self, *args, **kwargs) -> Any:
        # predict will call forward
        start_prob, end_prob = self.model(input_ids=kwargs["input_ids"], token_type_ids=kwargs["token_type_ids"],
                                          attention_mask=kwargs["attention_mask"])
        return start_prob,end_prob