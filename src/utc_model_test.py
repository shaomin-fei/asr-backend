from models.utc_model import UTCModel


cmd_categories=["无线电实时监测","无线电查询正常监测站数量","无线电查询异常监测站数量","无线电查询非法信号监测站数量","无线电查询非法信号数量","无线电查询信号数量","无线电查询合法信号数量","无线电查询航空干扰数量"]
config={
     "task_path":"C:\\test\\asr_integration\\utc",
     "task_name":"zero_shot_text_classification",
     "pre_trained_model":"utc-base",
     "schema":cmd_categories
}

utc_model=UTCModel(config)

result=utc_model.predict("查一下有多少个站有非法信号")
label=result["text"]
assert(label=="无线电查询非法信号监测站数量")

result=utc_model.predict("查一下有多少个站有非法信号")
label=result["text"]
assert(label=="无线电查询非法信号监测站数量")

result=utc_model.predict("衡山站监测到了多少合法信号")
label=result["text"]
assert(label=="无线电查询合法信号数量")

result=utc_model.predict("衡山站监测到了多少非法信号")
label=result["text"]
assert(label=="无线电查询非法信号数量")

result=utc_model.predict("今天有多少非法信号在莫斯科站")
label=result["text"]
assert(label=="无线电查询非法信号数量")