import negfflow
from negfflow.flowgen import FlowGen
import json

with open('flow.json') as j:
    config = json.loads(j.read())
config = negfflow.normalize(config)
wf = FlowGen(config, True)
wf.submit()