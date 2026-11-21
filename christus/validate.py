"""Real bounded engineering runs. Every arm and failure is retained on disk."""
from __future__ import annotations
import argparse, copy, hashlib, json, sqlite3, subprocess, sys, traceback
from pathlib import Path
import numpy as np
from .runtime import create, record, ROOT

def write(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def cognitive_state(s):
    # Wall times / database episode bookkeeping are intentionally excluded,
    # but all causal model, scheduler phase, memory evidence and RNG retained.
    memory=s.memory.snapshot()
    def clean(x):
        if isinstance(x,dict):
            return {k:clean(v) for k,v in x.items() if k not in
                ('created_at','updated_at','last_access','timestamp','memory_id')}
        if isinstance(x,(list,tuple)): return [clean(v) for v in x]
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        return x
    return clean({'step':s.clock.step,'sim_time':s.clock.sim_time,
        'rng':s.rng.bit_generator.state,'core':s.core.state_arrays(),
        'modules':s.registry.snapshot(),'shared':s.shared.snapshot(),
        'memory':memory,'world':s.world.snapshot()})

def summary(rows):
    terminal=[r for r in rows if r['outcome']]
    return {'steps':len(rows),'episodes':len(terminal),
        'caught':sum(r['outcome']=='caught' for r in terminal),
        'reward':sum(r['reward'] for r in rows),'cost':sum(r['cost'] for r in rows),
        'body_mae':float(np.mean([r['body_error'] for r in rows])),
        'body_updates':rows[-1]['body_updates'],'memory_updates':rows[-1]['memory_updates'],
        'model_used':sum(r['model_used'] for r in rows),
        'faults':max(r['faults'] for r in rows),
        'learning_gain_range':[min(r['learning_gain'] for r in rows),max(r['learning_gain'] for r in rows)],
        'phases':{name:{'n':len(part),'body_mae':float(np.mean([r['body_error'] for r in part])),
                              'cost':sum(r['cost'] for r in part)}
                  for name,part in [('normal',rows[:400]),('changed',rows[400:750]),('return',rows[750:])]
                  if part}}

def run_arm(out, disabled=(), steps=1000, profile='integration'):
    s=create(out,profile=profile,disabled=disabled)
    s.config.save(out/'config.json')
    rows=[]
    try:
        with (out/'events.jsonl').open('x',encoding='utf-8') as f:
            for i in range(steps):
                row=record(s,s.step()); rows.append(row)
                f.write(json.dumps(row,ensure_ascii=False)+'\n'); f.flush()
                if (i+1)%250==0: s.save_checkpoint('periodic engineering acceptance')
        s.save_checkpoint('final engineering acceptance')
        report=summary(rows)
        report['health']=[h.to_dict() for h in s.registry.health_report()]
        report['disabled']=list(disabled)
        s.flush_events()
        db=sqlite3.connect(s.config.resolve_path('paths.database'))
        report['database_events']=db.execute('select count(*) from events').fetchone()[0]
        report['distinct_event_ids']=db.execute('select count(distinct event_id) from events').fetchone()[0]
        db.close()
        write(out/'summary.json',report)
        return report,rows
    finally: s.close()

def resume_child(out, dest, steps):
    s=create(out,profile='integration',restore=True)
    rows=[record(s,s.step()) for _ in range(steps)]
    write(dest,{'rows':rows,'state':cognitive_state(s)})
    s.close()

def resume_test(out):
    s=create(out,profile='integration')
    try:
        for _ in range(127): s.step()
        s.save_checkpoint('resume fork at pending-capable transition boundary')
        before=cognitive_state(s)
        import shutil
        child_out=out.parent/'resume_child'
        shutil.copytree(out,child_out)
        # Use SQLite online backup, not an unsafe copy of a live WAL database.
        srcdb=sqlite3.connect(s.config.resolve_path('paths.database'))
        dstdb=sqlite3.connect(child_out/'storage/wphr.db')
        srcdb.backup(dstdb); dstdb.close(); srcdb.close()
        # Uninterrupted instance continues; child separately loads checkpoint.
        rows=[record(s,s.step()) for _ in range(50)]
        state=cognitive_state(s)
        s.flush_events()
        # Fork the persistence workspace, retaining a sole active runtime per DB.
        cp=subprocess.run([sys.executable,'-m','christus.validate','--child',str(child_out),
                           '--dest',str(out/'child.json')],cwd=ROOT,capture_output=True,text=True)
        (out/'child.stdout.log').write_text(cp.stdout,encoding='utf-8')
        (out/'child.stderr.log').write_text(cp.stderr,encoding='utf-8')
        if cp.returncode: raise RuntimeError(cp.stderr)
        child=json.loads((out/'child.json').read_text(encoding='utf-8'))
        # Event IDs retained; the child DB was forked before continuation.
        result={'start_step':before['step'],'continued_steps':50,
                'rows_equal':rows==child['rows'],'causal_state_equal':state==child['state']}
        write(out/'live.json',{'rows':rows,'state':state})
        write(out/'comparison.json',result)
        return result
    finally: s.close()

def correction_test(out):
    s=create(out,profile='integration')
    try:
        for _ in range(20): s.step()
        candidates=[(eid,e) for eid,e in s.bridge._state['observed_events'].items() if e['cue']=='ball']
        if not candidates: raise RuntimeError('no observed ball for grounding')
        eid=candidates[-1][0]; m=s.registry.get('concept_language'); p=s.registry.get('practice')
        before=m.belief('ball')
        scores_before=p._scores('none',[0.,0.])
        accepted=s.correction(event_id=eid,text='avoid ball',polarity=-1,source='test_teacher',
                              reliability=.9,feedback_id='integration-correction-1')
        after=m.belief('ball'); scores_after=p._scores('none',[0.,0.])
        dupe=s.correction(event_id=eid,text='avoid ball',polarity=-1,source='test_teacher',
                         reliability=.9,feedback_id='integration-correction-1')
        # Candidate scores, not a measured executed-action counterfactual.
        result={'source_kind':'synthetic_test_feedback','evidence_id':eid,
                'belief_before':before,'belief_after':after,'accepted':accepted,
                'duplicate':dupe,'scores_before':scores_before,'scores_after':scores_after,
                'decision_scores_changed':scores_before!=scores_after,
                'words':m.lexicon.snapshot()}
        write(out/'correction.json',result)
        return result
    finally:s.close()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path)
    ap.add_argument('--child',type=Path); ap.add_argument('--dest',type=Path)
    args=ap.parse_args()
    if args.child: return resume_child(args.child,args.dest,50)
    out=args.out.resolve()
    if out.exists(): raise RuntimeError('Use a new validation directory; preserve old failures')
    out.mkdir(parents=True)
    results={}; rows_by_arm={}
    arms={'full':(), 'safe':(), **{'without_'+k:(k,) for k in
        ['world','world02','practice','affect','temporal','concept_language','social','volition','memory','action_model']}}
    for name,disabled in arms.items():
        print('RUN '+name,flush=True)
        try:
            results[name],rows_by_arm[name]=run_arm(out/name,disabled,profile='safe' if name=='safe' else 'integration')
        except Exception:
            results[name]={'error':traceback.format_exc()}; write(out/name/'failure.json',results[name])
        write(out/'summary.json',results)
    for name,fn in [('resume',resume_test),('correction',correction_test)]:
        try: results[name]=fn(out/name)
        except Exception: results[name]={'error':traceback.format_exc()}
        write(out/'summary.json',results)
    base=rows_by_arm.get('full',[])
    for name,rows in rows_by_arm.items():
        results[name]['action_differences_vs_full']=sum(a['action']!=b['action'] for a,b in zip(base,rows))
    write(out/'summary.json',results)
    print(json.dumps(results,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
