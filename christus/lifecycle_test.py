"""One targeted rejection test; actual environment steps, no success fabrication."""
import json, sqlite3, subprocess, sys
from .runtime import ROOT, create

def main():
    out=ROOT/'results/uncommitted_tail01'
    if out.exists(): raise RuntimeError('Preserve existing test; do not rerun')
    s=create(out)
    for _ in range(5): s.step()
    s.save_checkpoint('committed boundary')
    for _ in range(3): s.step()
    s.flush_events()
    s.close() # deliberately no final checkpoint; interruption fixture, not a kill test
    def count():
        with sqlite3.connect(out/'storage/wphr.db') as db:
            return db.execute('select count(*),count(distinct event_id),max(step) from events').fetchone()
    before=count()
    cp=subprocess.run([sys.executable,'-m','christus','--out',str(out),'--steps','1','--resume'],
                      cwd=ROOT,capture_output=True,text=True)
    (out/'resume.stdout.log').write_text(cp.stdout,encoding='utf-8')
    (out/'resume.stderr.log').write_text(cp.stderr,encoding='utf-8')
    result={'kind':'real_steps_with_deliberately_uncommitted_tail','checkpoint_step':5,
            'executed_steps':8,'returncode':cp.returncode,'before':before,'after':count(),
            'rejected_without_new_learning':cp.returncode!=0 and 'uncheckpointed tail' in cp.stderr and before==count()}
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))
    if not result['rejected_without_new_learning']: raise RuntimeError('Recovery guard failed')

if __name__=='__main__': main()
