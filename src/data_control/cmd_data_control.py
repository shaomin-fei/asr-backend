import numpy as np
import torch
from data_control.base_speech_data_control import BaseSpeechDataControl
from models.wakeup_model import WakeupModel

from enum import Enum

class CmdStatus(Enum):
    Waiting_WakeUp = 0
    Waiting_Command = 1
    Receiving_Command = 2
    Waiting_Command_Timeout = 3
    End_Command = 4

class CmdDataControl(BaseSpeechDataControl):
    def __init__(self, config:dict):
        super().__init__(config=config)
        self.end_command_after_seconds=config.get("end_command_after_seconds",1.5)
        self.end_listen_after_seconds=config.get("end_listen_after_seconds",10)
        self.wakeup_model = WakeupModel(config)
        self.status = CmdStatus.Waiting_WakeUp

    def data_arrive(self,audio_data:np.ndarray):
        status=self.status
        if self.status==CmdStatus.Waiting_WakeUp:
            is_wake_up=self.wakeup_model.predict(audio_data)
            if is_wake_up:
                self.status=CmdStatus.Waiting_Command
                self.speech_lifetime.set_lifetime(self.end_listen_after_seconds)
        elif self.status==CmdStatus.Waiting_Command:
            self.speech_lifetime.set_lifetime(self.end_listen_after_seconds)
        elif self.status==CmdStatus.Receiving_Command:
            self.speech_lifetime.set_lifetime(self.end_command_after_seconds)
            
        data=torch.tensor(audio_data,dtype=torch.float32)
        result=self.handle_data(data,raw_data=audio_data)
        if result["has_speech"]:
            self.status=CmdStatus.Receiving_Command
        elif result["timeout"]:
            if self.status==CmdStatus.Receiving_Command:
                status=CmdStatus.End_Command
            elif self.status==CmdStatus.Waiting_Command:
                status=CmdStatus.Waiting_Command_Timeout
        return {"status":status,"text":result["text"]}

    def reset(self):
        super().reset()
        self.status=CmdStatus.Waiting_WakeUp