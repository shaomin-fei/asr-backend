import os
from eff_word_net.engine import HotwordDetector,MultiHotwordDetector
from eff_word_net.audio_processing import Resnet50_Arc_loss
from eff_word_net import samples_loc
import numpy as np

class WakeUpModelMulti:
    def __init__(self,config:dict) -> None:
        ref_path=config.get("ref_path",samples_loc)
        hotwords=config.get("hotwords",{})
        self.base_model = Resnet50_Arc_loss()
        # sample_rate=16000*1.5second
        self.data_len=24000
        # sample_rate=16000*1.5second*0.5
        self.window_size=12000
        self.hotwords_detector=[]
        for key in hotwords:
            hw=HotwordDetector(
                hotword=key,
                model=self.base_model,
                reference_file=os.path.join(ref_path,hotwords[key]),
                threshold=0.7,
                relaxation_time=2
            )
            self.hotwords_detector.append(hw)
        self.multi_hotword_detector = MultiHotwordDetector(
        self.hotwords_detector,
        model=self.base_model,
        continuous=True,
        )

    def predict(self,audio:np.ndarray):
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
            result=self.multi_hotword_detector.findBestMatch(chunk)
            if(None not in result):
                print(result[0],f",Confidence {result[1]:0.4f}")
                return True
        return False
        
        