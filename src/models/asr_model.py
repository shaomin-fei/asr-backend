import numpy as np
import torchaudio
import torch 
import torchaudio.compliance.kaldi as kaldi
import wenet
from wenet.transformer.search import (attention_rescoring,
                                      ctc_prefix_beam_search, DecodeResult)
from wenet.utils.context_graph import ContextGraph
from wenet.utils.ctc_utils import (force_align, gen_ctc_peak_time,
                                   gen_timestamps_from_peak)



class ASRModel:
    def __init__(self, config:dict):
        model_path=config.get("model_path", "models/wenet/")
        context_path=config.get("context_path", None)
        self.model_wrapper = wenet.load_model("chinese", model_dir=model_path,context_path=context_path)
        self.replace_words = config.get("replace_words", {"死看":"SCAN","死干":"SCAN"})
        # the model requires 16000 sample rate
        self.sample_rate = 16000
        self.recording_sample_rate = config.get("recording_sample_rate", 16000) # we need it when doing stream prediction
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict_file(self, file_path: str) -> str:
        decoded= self.model_wrapper.transcribe(file_path)
        if decoded is None:
            return "Error, No text output!"
        txt=decoded["text"]
        for k, v in self.replace_words.items():
            txt = txt.replace(k, v)
        return txt
    
    def predict_stream(self,data:np.ndarray,recording_sample_rate=None,tokens_info: bool = False) -> str:
        if recording_sample_rate is None:
            recording_sample_rate=self.recording_sample_rate
        decoded= self.decode_stream(data,recording_sample_rate,tokens_info)
        if decoded is None:
            return "Error, No text output!"
        txt=decoded["text"]
        for k, v in self.replace_words.items():
            txt = txt.replace(k, v)
        return {"text":txt,"confidence":decoded["confidence"]}
    
    def decode_stream(self, data:np.ndarray,recording_sample_rate=16000,tokens_info: bool = False,label: str = None) -> str:
        feats=self.compute_feats(data,recording_sample_rate)
        encoder_out, _, _ = self.model_wrapper.model.forward_encoder_chunk(feats, 0, -1)
        encoder_lens = torch.tensor([encoder_out.size(1)],
                                    dtype=torch.long,
                                    device=encoder_out.device)
        ctc_probs = self.model_wrapper.model.ctc_activation(encoder_out)
        if label is None:
            ctc_prefix_results = ctc_prefix_beam_search(
                ctc_probs,
                encoder_lens,
                self.model_wrapper.beam,
                context_graph=self.model_wrapper.context_graph)
        else:  # force align mode, construct ctc prefix result from alignment
            label_t = self.model_wrapper.tokenize(label)
            alignment = force_align(ctc_probs.squeeze(0),
                                    torch.tensor(label_t, dtype=torch.long))
            peaks = gen_ctc_peak_time(alignment)
            ctc_prefix_results = [
                DecodeResult(tokens=label_t,
                             score=0.0,
                             times=peaks,
                             nbest=[label_t],
                             nbest_scores=[0.0],
                             nbest_times=[peaks])
            ]
        rescoring_results = attention_rescoring(self.model_wrapper.model, ctc_prefix_results,
                                                encoder_out, encoder_lens, 0.3,
                                                0.5)
        res = rescoring_results[0]
        result = {}
        result['text'] = ''.join([self.model_wrapper.char_dict[x] for x in res.tokens])
        result['confidence'] = res.confidence

        if tokens_info:
            frame_rate = self.model_wrapper.model.subsampling_rate(
            ) * 0.01  # 0.01 seconds per frame
            max_duration = encoder_out.size(1) * frame_rate
            times = gen_timestamps_from_peak(res.times, max_duration,
                                             frame_rate, 1.0)
            tokens_info = []
            for i, x in enumerate(res.tokens):
                tokens_info.append({
                    'token': self.model_wrapper.char_dict[x],
                    'start': times[i][0],
                    'end': times[i][1],
                    'confidence': res.tokens_confidence[i]
                })
            result['tokens'] = tokens_info
        return result
           
    def compute_feats(self, data:np.ndarray,recording_sample_rate=16000) -> torch.Tensor:
        waveform=torch.tensor(data)
        waveform = waveform.to(torch.float)
        waveform=waveform.reshape(1,-1)
        if recording_sample_rate != self.sample_rate:
            waveform = torch.transforms.Resample(
                orig_freq=recording_sample_rate, new_freq=self.sample_rate)(waveform)
        waveform.to(self.device)
        feats=kaldi.fbank(waveform,
                          num_mel_bins=80,
                          frame_length=25,
                          frame_shift=10,
                          energy_floor=0.0,
                          sample_frequency=self.sample_rate)
        feats=feats.unsqueeze(0)
        return feats
    

    