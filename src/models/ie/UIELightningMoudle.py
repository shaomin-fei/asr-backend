import json
import os.path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT

from models.ie.UIEModel import UIE
from models.ie.SpanEvaluator import SpanEvaluator


class UIELightningMoudle(L.LightningModule):
    def __init__(self, model_path, lr,model_name="pytorch_model.bin"):
        super().__init__()
        # config_path=f"{model_path}/config.json"
        # if not os.path.isfile(config_path):
        #     raise Exception(f"config file in {config_path} not found")
        # with open(config_path) as reader:
        #     self.config=json.load(reader)
        self.model = UIE.from_pretrained(model_path)
        #self.model.load_state_dict(torch.load(f"{model_path}/{model_name}"))
        self.lr = lr
        self.loss_fn = torch.nn.BCELoss()
        self.loss_lst=[]
        self.val_loss_lst=[]
        self.metric=SpanEvaluator()
        # with this code, model_path, lr,model_name are saved in checkpoint. checkpoint["hyper_parameters"]
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, start_pos, end_pos=batch
        start_prob, end_prob=self.model(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)
        start_pos=start_pos.type(torch.float32)
        end_pos=end_pos.type(torch.float32)
        loss_start=self.loss_fn(start_prob,start_pos)
        loss_end=self.loss_fn(end_prob,end_pos)
        loss=(loss_start+loss_end)*0.5
        self.loss_lst.append(float(loss))
        print(f"train loss = {loss}, batch index = {batch_idx}")
        return loss

    def on_validation_start(self) -> None:
        print("val start......")
        self.metric.reset()

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> STEP_OUTPUT:
        input_ids, token_type_ids, attention_mask, start_pos, end_pos = batch
        start_prob, end_prob = self.model(input_ids=input_ids, token_type_ids=token_type_ids,
                                          attention_mask=attention_mask)
        loss_start = self.loss_fn(start_prob, start_pos.type(torch.float32) )
        loss_end = self.loss_fn(end_prob, end_pos.type(torch.float32))
        loss = (loss_start + loss_end) * 0.5
        loss=float(loss)
        self.val_loss_lst.append(loss)
        loss_avg=sum(self.val_loss_lst)/len(self.val_loss_lst)
        num_correct, num_infer, num_label = self.metric.compute(start_prob, end_prob,
                                                           start_pos, end_pos)
        self.metric.update(num_correct, num_infer, num_label)

        precision, recall, f1 = self.metric.accumulate()

        print(f"val loss = {loss}, batch index = {batch_idx}, precision={precision},recall={recall},f1={f1}")
        # log info so the early stop can find the metric
        self.log("val_f1", f1)
        self.log("val_loss", loss)
        self.log("val_precision", precision)
        return {
            "val_loss":loss,
            "val_f1":f1,
            "val_precision":precision,
            "val_recall":recall
        }

    def forward(self, *args, **kwargs) -> Any:
        # predict will call forward
        start_prob, end_prob = self.model(input_ids=kwargs["input_ids"], token_type_ids=kwargs["token_type_ids"],
                                          attention_mask=kwargs["attention_mask"])
        return start_prob,end_prob






    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(lr=self.lr, params=self.parameters())
        return optimizer
