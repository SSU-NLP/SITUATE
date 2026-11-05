import importlib.util,json,pathlib,collections
from tokenizers import Tokenizer
root=pathlib.Path('/home/namchailin/Projects/Hi-OnTop');p=root/'benchmarks/STAGE_v0/evaluation/scripts/run_task1_state_update_evaluation.py';s=importlib.util.spec_from_file_location('judge',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
t=Tokenizer.from_file('/home/namchailin/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json')
def n(x): return len(t.encode(x).ids)
def js(x): return json.dumps(x,ensure_ascii=False)
audit=[]
for p in (root/'outputs/experiments/2026-09-26_stage_situate/pre_api_backup').glob('*/predictions/preflight*.json'): audit+=json.loads(p.read_text())
out={'exact_counts':dict(sum((collections.Counter(x['counts']) for x in audit),collections.Counter())),'movies':len(audit)}
st=collections.Counter()
for p in (root/'data/stage_task1_inputs').glob('*/task_1_reference_state_update.json'):
 a=json.loads(p.read_text())
 for tr in a['trajectories']:
  for cp in tr['checkpoints']:
   ref=m._localize_reference(cp['evaluation_target']); pred=[]
   for field,kind,cap in [('current_state','current_state',8),('developments_since_previous_checkpoint','development',4)]:
    for row in cp['evaluation_target'][field][:cap]: pred.append({**row,'prediction_type':kind,'local_id':f'P{len(pred)+1}'})
   kw=dict(character=tr['character'],previous_scene_order=cp['screenplay_interval']['start_scene_order_exclusive'],current_scene_order=cp['screenplay_interval']['end_scene_order_inclusive'])
   sy,us=m.PROMPTS.render(m._prompt_key(a['language'],m.COVERAGE_PROMPT_NAME),**kw,reference_claims=[[r['local_id'],'s' if r['claim_type']=='current_state' else 'd',r['claim']] for r in ref],predicted_claims=[[r['local_id'],'s' if r['prediction_type']=='current_state' else 'd',r['claim']] for r in pred],allowed_reference_ids=[r['local_id'] for r in ref],allowed_prediction_ids=[r['local_id'] for r in pred])
   st['coverage_input_tokens']+=n(sy)+n(us)+16;st['coverage_calls']+=1
   st['coverage_output_tokens']+=n(js({'reference_results':[{'reference_id':r['local_id'],'status':'full','prediction_ids':['P1']} for r in ref]}))
   st['reference_claims']+=len(ref);st['surrogate_prediction_claims']+=len(pred)
   for row in pred:
    evidence=[{'scene_order':order,'title':'Scene title','content':'[Excerpt 1]\n'+' evidence'*100} for order in row['evidence_scene_orders']]
    sy,us=m.PROMPTS.render(m._prompt_key(a['language'],m.EVIDENCE_PROMPT_NAME),**kw,predicted_claim={'type':row['prediction_type'],'claim':row['claim'],'evidence_scene_orders':row['evidence_scene_orders']},evidence_scenes=evidence,prediction_id=row['local_id'])
    st['validity_input_tokens']+=n(sy)+n(us)+16;st['validity_calls']+=1
    st['validity_output_tokens']+=n(js(dict(prediction_id=row['local_id'],supported=True,evidence_grounded=True,currently_valid=True,salient=True,development_coherent=True,contradiction=False)))
out['single_setting_reference_shape_proxy']=dict(st);out['four_methods_two_settings_proxy']={k:v*8 for k,v in st.items()}
I=8*(st['coverage_input_tokens']+st['validity_input_tokens'])/1e6;O=8*(st['coverage_output_tokens']+st['validity_output_tokens'])/1e6
out['api_input_million_proxy']=I;out['api_output_million_proxy']=O;out['api_usd_proxy']={'public_min_0348_0696':I*.348+O*.696,'provider_174_348':I*1.74+O*3.48}
out['assumptions']=['Reference claims stand in for unknown model outputs solely to estimate lengths; capped at 8 states and 4 developments.','Evidence placeholder 100 Qwen tokens per cited scene; actual tokenizer, carried provenance and predictions differ.','No retries, no cache discount, no reasoning tokens, no proxy markup; all four methods and both settings separately judged.']
path=root/'outputs/experiments/2026-09-26_stage_situate/cost_estimate.json';path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False,indent=2))
