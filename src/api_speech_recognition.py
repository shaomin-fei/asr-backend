import json
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from models.ie.predictor import UIEPredictor
from models.asr_model import ASRModel

from models.wakeup_model import WakeupModel
from models.wake_up_model_multi import WakeUpModelMulti
from models.utc_model import UTCModel
from itn.chinese.inverse_normalizer import InverseNormalizer

cmd_categories=["无线电实时监测","无线电查询正常监测站数量","无线电查询异常监测站数量","无线电查询非法信号监测站数量","无线电查询非法信号数量",
                "无线电查询信号数量","无线电查询合法信号数量","无线电查询航空干扰数量"]
# for test reason, 
test_data={
    "无线电实时监测":["任务执行成功!"],
    "无线电查询正常监测站数量":["一共有5个监测站正常工作。","一共有15个监测站正常工作。"],
    "无线电查询异常监测站数量":["有15个监测站异常。","有2个监测站异常。"],
    "无线电查询非法信号监测站数量":["两个监测站发现异常信号。","没有监测站发现异常信号。"],
    "无线电查询非法信号数量":["发现100个非法信号。","发现10个非法信号。"],
    "无线电查询信号数量":["发现10个信号。","发现20个信号。"],
    "无线电查询合法信号数量":["发现11个合法信号。","发现20个合法信号。"],
    "无线电查询航空干扰数量":["到目前为止，今天一共发现100个航空干扰。","一共发现12个航空干扰。"]
}
########################
base_path_windows="C:\\test\\asr_integration"
base_path_linux="/home/fsm/Desktop/ml/asr-integration-pretrained-models"
base_path=base_path_linux
config={
    "recording_sample_rate":44100, # gradio's recording sample rate is 44100
    #vad configuration
    "vad_path":f"{base_path}/vad",
    "has_speech_threshold":0.6,
    "window_size_samples":512*3,
    "asr":{
        "model_path":f"{base_path}/asr",
        "context_path":f"{base_path}/asr/context.txt",
    },
    
    # wakeup
    "wakeup":{
    "use-multi-hotwords":True,
    "ref_path":f"{base_path}/wakeup/samples",
    "threshold":{"Hey-Hunter":0.65,"Hi-Hunter":0.65}, # the length of hunter is much less, so we reduce the threshold. Default threshold is 0.7
    "hotwords":{"Hey-Hunter":"Hey_Hunter_ref.json","Hi-Hunter":"Hi_Hunter_ref.json","Hey-skywalker":"hey-skywalker_ref.json","Hi-Skywalker":"Hi_skywalker_ref.json"},
    },
    "ie":{
    "model_path":f"{base_path}/ie/best.ckpt",
    "pretrained_model":f"{base_path}/ie",
    "pretrained_tokenizer":f"{base_path}/ie",
    "device":"cpu",
    "positon_prob":0.5,
    "max_seq_len":512,
    "batch_size":63,
    "split_sentence":False,
    "schema":["监测站","省监测中心","任务","市监测中心","中心频率","带宽","起始频率","截止频率","步进","设备","解调方式","声音开关","自动增益","频段扫描类别","时间"]
    },
    "utc":{
        "task_path":f"{base_path}/utc",
        "task_name":"zero_shot_text_classification",
        "pre_trained_model":"utc-base",
        "schema":cmd_categories
    }

}

invnormalizer = InverseNormalizer(enable_0_to_9=True,enable_million=True)

if(config["wakeup"]):
    wakeup_model=WakeUpModelMulti(config["wakeup"])
else:
    wakeup_model=WakeupModel(config["wakeup"])
asr_model=ASRModel(config["asr"])
ie_model=UIEPredictor(config["ie"])
utc_model=UTCModel(config["utc"])

app=Flask(__name__)
CORS(app)

@app.route('/api/wakeup', methods=['POST'])
def is_wake_up():
    sample_rate =int(request.form.get('sampleRate', 16000)) 
    data = request.files.get('audio').read()
    #data.save("audio.wav")
    #data.flush()
    #data.close()
    data=np.frombuffer(data,dtype=np.float32)
    data=data.astype(np.float64)
    res=wakeup_model.predict(data)
    return {"isWakeUp":res}

@app.route('/api/recognize/cmd', methods=['POST'])
def recognize_cmd():
    result={"text":"","parameters":{}, "cmdType":""}
    if request.method == "OPTIONS":
         return result
    sample_rate =int(request.form.get('sampleRate', 16000)) 
    data = request.files.get('audio').read()
    #data.save("audio.wav")
    #data.flush()
    #data.close()
    data=np.frombuffer(data,dtype=np.float32)
    result=asr_model.predict_stream(data,sample_rate)
    result["text"]=invnormalizer.normalize(result["text"])
    print(f"decode result: text={result['text']},confident={result['confidence']}")
    parameters={}
    ttsWords=""
    if result["text"]:
        # classify first
        category=utc_model.predict(result["text"])
        if category["text"] not in cmd_categories:
            cmd_type="Unknown"
            ttsWords="无法识别的指令"
        else:
            cmd_type=category["text"]
            extracted=ie_model(result["text"])
            parameters=extracted[0]
            # test, randomly return tts words
            index=np.random.randint(0,len(test_data[cmd_type]),1)
            ttsWords=test_data[cmd_type][index[0]]
        result["cmdType"]=cmd_type
        result["parameters"]=parameters if parameters is not None else []
        result["success"]=True
        station=result["parameters"].get("监测站",[{"text":""}])
        result["ttsMessage"]=f"{station[0]['text']}{ttsWords}"
    return result


@app.route('/api/recognize/stream', methods=['POST'])
def recognize_stream():
    result={"text":""}
    sample_rate =int(request.form.get('sampleRate', 16000)) 
    data = request.files.get('audio').read()
    #data.save("audio.wav")
    #data.flush()
    #data.close()
    data=np.frombuffer(data,dtype=np.float32)
    result=asr_model.predict_stream(data,sample_rate)
    print(f"decode result: text={result['text']},confident={result['confidence']}")
    return result

@app.route('/api/tts', methods=['POST'])
def tts(words:str):
    pass

all_ip_address="0.0.0.0"
local_ip_address="127.0.0.1"
ip_address=all_ip_address
if __name__ == "__main__":
    # run on https, need to pip install pyopenssl, and pip install cryptography
    # otherwise we will get error like: code 400, message Bad request version when receiving request
    # BUT, I have problem with https, maybe we need to generate cert by ourselves.
    # So I stick to HTTP on the client side instead of HTTPS
    app.run(host=ip_address)