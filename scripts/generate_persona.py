import argparse, hashlib, json, os, time
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bank',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--revision',required=True)
    parser.add_argument('--adapter')
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    set_seed(42)
    model_id='HuggingFaceTB/SmolLM2-360M-Instruct'
    tokenizer=AutoTokenizer.from_pretrained(model_id,revision=args.revision)
    model=AutoModelForCausalLM.from_pretrained(model_id,revision=args.revision,torch_dtype=torch.float32).to('cuda:0')
    if args.adapter:
        from peft import PeftModel
        adapter=Path(args.adapter)
        assert (adapter/'adapter_model.safetensors').is_file() and (adapter/'adapter_config.json').is_file()
        model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False)
    model.eval()
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    rows=json.loads(Path(args.bank).read_text())
    if args.limit: rows=rows[:args.limit]
    settings={'do_sample':False,'max_new_tokens':256,'repetition_penalty':1.1,'no_repeat_ngram_size':0}
    run={'status':'running','model_id':model_id,'revision':args.revision,'adapter':args.adapter,'bank_sha256':hashlib.sha256(Path(args.bank).read_bytes()).hexdigest(),'decoding':settings,'precision':'float32','prompt_format':'native_chat_template_no_system','seed':42,'count':len(rows)}
    if args.adapter:run['adapter_sha256']=hashlib.sha256((Path(args.adapter)/'adapter_model.safetensors').read_bytes()).hexdigest()
    def state(): (output/'run.json').write_text(json.dumps(run,indent=2)+'\n')
    state();answers=[]
    try:
        for i,row in enumerate(rows):
            text=tokenizer.apply_chat_template([{'role':'user','content':row['instruction']}],tokenize=False,add_generation_prompt=True)
            tokens=tokenizer(text,return_tensors='pt',add_special_tokens=False).to(model.device)
            assert tokens.input_ids.shape[1]+256<=model.config.max_position_embeddings
            with torch.inference_mode(): generated=model.generate(**tokens,**settings,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
            new=generated[0,tokens.input_ids.shape[1]:].tolist()
            answer=tokenizer.decode(new,skip_special_tokens=True)
            record={**row,'output':answer,'generated_tokens':len(new),'finish_reason':'eos' if new and new[-1]==tokenizer.eos_token_id else 'length','blank':not answer.strip()}
            answers.append(record)
            with (output/'outputs.jsonl').open('a') as handle:
                handle.write(json.dumps(record,ensure_ascii=False)+'\n');handle.flush();os.fsync(handle.fileno())
            run['completed']=i+1;state()
            print(f'[PERSONA GENERATE {i+1}/{len(rows)}] '+row['id'],flush=True)
            if i<4:print(json.dumps(record,ensure_ascii=False),flush=True)
        (output/'outputs.json').write_text(json.dumps(answers,ensure_ascii=False,indent=2)+'\n')
        run.update(status='completed',blanks=sum(a['blank'] for a in answers),outputs_sha256=hashlib.sha256((output/'outputs.json').read_bytes()).hexdigest());state()
    except BaseException as exc:
        run.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)});state();raise
if __name__=='__main__':main()
