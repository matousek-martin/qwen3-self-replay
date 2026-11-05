"""Small, non-benchmark fixtures for GPU plumbing, never efficacy evidence."""
import json

def fixtures():
    specs=[('math','What is 1 + 1? Put the final number in \\boxed{}.',{'task':'math','value':'2'}),
           ('coding','Write Python code that prints hello. Put your final code in a Python code block.',{'task':'coding','tests':{'inputs':[''],'outputs':['hello']}}),
           ('tools','The user wants to book a flight. The only available function is add(a, b), which adds two numbers. Output only a JSON array of calls after reasoning, or [] if no available function can fulfill the request.',{'task':'tools','calls':[]}),
           ('instructions','Write the single word hello in lowercase as your final answer.',{'task':'instructions','checks':['lowercase']})]
    return [dict(id='technical-'+task,task=task,prompt=[{'role':'user','content':prompt}],answer=json.dumps(spec)) for task,prompt,spec in specs]
