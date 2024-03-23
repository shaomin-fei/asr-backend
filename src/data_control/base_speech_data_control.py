import numpy as np
import torch
from models.asr_model import ASRModel
from life_time.speech_lifetime import SpeechLifeTime
from models.vad_model import VADModel


class BaseSpeechDataControl:
    def __init__(self,config:dict):
        interval_between_sentence=config.get("interval_between_sentence",1)
        self.speech_lifetime = SpeechLifeTime(speech_lifetime_second=interval_between_sentence)
        vad_path = config.get("vad_path", None)
        self.vad_model = VADModel(local_path=vad_path,config=config)
        self.asr_model=ASRModel(config=config) 
        self.data_tensor=None
        self.decoded_txt=""

    def data_arrive(self,audio_data:np.ndarray):
        data=torch.tensor(audio_data,dtype=torch.float32)
        return self.handle_data(data,raw_data=audio_data)

    def handle_data(self,audio_data:torch.tensor,raw_data:np.ndarray):
        timeout=False
        has_speech=self.vad_model.has_speech(audio_data)
        result_txt=""
        if has_speech:
            self.speech_lifetime.start()
            self.add_data(audio_data)
            # process speech recognize
            self.decoded_txt=self.asr_model.decode_stream(self.data_tensor.numpy())["text"]
            result_txt=self.decoded_txt
        else:
            if self.speech_lifetime.is_start():
                if self.speech_lifetime.is_end(len(audio_data), self.vad_model.sample_rate):
                    result_txt=self.decoded_txt
                    self.reset()
                    timeout=True
                else:
                    self.add_data(audio_data)
        return {"timeout":timeout,"has_speech":has_speech,"text":result_txt}

    def add_data(self,audio_data:torch.tensor):
        # concatenate the audio data
            if self.data_tensor is None:
                self.data_tensor=audio_data
            else:
                self.data_tensor=torch.cat((self.data_tensor,audio_data),0)

    def reset(self):
        self.speech_lifetime.reset()
        self.data_tensor=None
        self.decoded_txt=""