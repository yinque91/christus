"""One local bounded runtime. No camera/network/background service by default."""
import argparse, json, os
from pathlib import Path
from .runtime import create, record
from wphr.runtime.service_manager import ServiceManager

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--seed',type=int,default=31100)
    p.add_argument('--profile',choices=['safe','integration'],default='safe')
    p.add_argument('--disable',action='append',default=[])
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    lock=args.out/'runtime.lock'
    if lock.exists():
        try:
            pid=int(lock.read_text()); os.kill(pid,0)
        except (OSError,ValueError): lock.unlink()
        else: raise RuntimeError(f'Another runtime owns this directory: {pid}')
    with lock.open('x') as f: f.write(str(os.getpid()))
    s=None
    service=ServiceManager(args.out)
    service.install_signals()
    try:
        if (args.out/'events.jsonl').exists() and not args.resume:
            raise RuntimeError('Existing run; use --resume or a fresh output directory')
        s=create(args.out,args.seed,args.profile,args.disable,args.resume)
        if args.resume:
            # Never silently append replayed transitions after a crash. Recovery
            # of an uncommitted tail needs an explicit journal reconciliation.
            path=args.out/'events.jsonl'
            with s.store.connect() as db:
                last_event_step=db.execute('SELECT MAX(step) FROM events').fetchone()[0]
            if last_event_step is not None and last_event_step > s.clock.step:
                raise RuntimeError('Database contains an uncheckpointed tail; preserved for recovery, no steps executed')
            if path.exists():
                with path.open(encoding='utf-8') as prior:
                    last=next(reversed(prior.readlines()), '')
                if last and json.loads(last)['sequence'] != s.world.total:
                    raise RuntimeError('Log/checkpoint boundary differs; preserved tail requires recovery, no steps executed')
        s.config.save(args.out/'config.json')
        service.write_status({'state':'running','step':s.clock.step})
        with (args.out/'events.jsonl').open('a',encoding='utf-8') as f:
            for _ in range(args.steps):
                if s.control.pending() in ('pause','stop') or service.stop_requested() or service.paused(): break
                r=s.step(); f.write(json.dumps(record(s,r),ensure_ascii=False)+'\n'); f.flush()
                if s.clock.step%100==0:
                    s.save_checkpoint('periodic')
                    service.write_status({'state':'running','step':s.clock.step})
        checkpoint=s.save_checkpoint('normal exit')
        service.write_status({'state':'stopped','step':s.clock.step,'checkpoint':str(checkpoint.path)})
        print(json.dumps({'steps':s.clock.step,'checkpoint':str(checkpoint.path),
                          'memory_updates':s.bridge._state['memory_updates']}))
    except BaseException as exc:
        service.write_status({'state':'failed','error':repr(exc),'step':None if s is None else s.clock.step})
        raise
    finally:
        if s is not None: s.close()
        lock.unlink(missing_ok=True)

if __name__=='__main__': main()
