import os
from eff_word_net.engine import HotwordDetector
from eff_word_net.audio_processing import Resnet50_Arc_loss
from eff_word_net import samples_loc
import numpy as np

class WakeupModel:
    def __init__(self,config:dict) -> None:
        ref_path=config.get("ref_path",samples_loc)
        self.base_model = Resnet50_Arc_loss()
        # sample_rate=16000*1.5second
        self.data_len=24000
        # sample_rate=16000*1.5second*0.5
        self.window_size=12000
        self.hey_skywalker = HotwordDetector(
        hotword="hey_skywalker",
        model=self.base_model,
        reference_file=os.path.join(ref_path, "hey-skywalker_ref.json"),
        threshold=config.get("threshold",0.7),
        relaxation_time=2
    )
    def predict(self,audio:np.ndarray):
        # each chunk should be 240001
        chunk_size=24000
        offset=12000
        chunks=[]
        audio_len=audio.shape[0]
        last_chunk=False
        if audio_len<chunk_size:
            audio=np.pad(audio,(0,chunk_size-audio_len))
            audio_len=chunk_size
            chunks.append(audio)
        else:
            for i in range(0,audio_len,offset):
                if i+chunk_size>audio_len:
                    i=audio_len-chunk_size
                    last_chunk=True
                chunks.append(audio[i:i+chunk_size])
                if last_chunk:
                    break
        for chunk in chunks:
            #the data should be (24000,), and the type is float64
            result=self.hey_skywalker.scoreFrame(chunk,True)
            if(result["match"]):
                print("Wakeword uttered",result["confidence"])
                return True
        return False
            