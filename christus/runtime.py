"""CHRISTUS is an integration of WPHR.Subject, not a new cognitive core."""
from __future__ import annotations
import copy
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from wphr.config import Config
from wphr.runtime.subject import Subject
from wphr.core.module import BaseModule
from wphr.core.event import EventType, Source
from wphr.modules.practice import PracticeModule, ACTIONS
from wphr.memory.layers import MemoryRecord
from wphr.memory.store import Store
from wphr.cognition.smctrn import CoreMetrics
from .session_world import SessionWorld, COMMAND
from .core_readout import CoreActionReadout

ROOT = Path(__file__).resolve().parents[1]

def configuration(out, seed=31100, profile='safe', disabled=(), arm=None):
    cfg = Config.load(root=ROOT)
    for key in cfg.section('paths'):
        raw = cfg.get('paths.'+key)
        cfg.set('paths.'+key, str((Path(out)/raw).resolve()))
    cfg.set('world.id', 'christus_session')
    cfg.set('sensors.enabled', False)
    cfg.set('runtime.seed', seed)
    cfg.set('runtime.auto_reset_episodes', False)
    cfg.set('runtime.status_every', 0)
    cfg.set('memory.retention.auto_archive_every', 0)
    cfg.set('memory.retention.prune_on_start', False)
    cfg.set('memory.enabled', 'memory' not in disabled)
    cfg.set('planning.enabled', False)  # no grid-navigation plan in catcher world
    cfg.set('modules.disabled', [x for x in disabled if x not in ('memory','world02','action_model','core')])
    cfg.set('affect.ablation', 'full' if profile=='integration' and 'affect' not in disabled else 'none')
    cfg.set('language.enabled', profile=='integration' and 'concept_language' not in disabled)
    cfg.set('christus', {'profile':profile, 'disabled':list(disabled),
                        'action_model':profile=='integration' and 'action_model' not in disabled,
                        'world02_learning':'world02' not in disabled,
                        'advanced_history':False, 'post03_prior':False,
                        'language_action':profile=='integration',
                        'devices':False, 'network':False})
    if arm is not None:
        # CORE-READOUT-01 controlled arms. All four share the one-dimensional
        # observation bridge and the disabled hand-coded scent prior; only the
        # readout mode differs. SAFE/integration leave these keys absent.
        if arm not in ('A0', 'A1', 'A2', 'A3'):
            raise ValueError(f'unknown core-readout arm: {arm!r}')
        modes = {'A0': 'active', 'A1': 'active', 'A2': 'masked', 'A3': 'shuffled'}
        cfg.set('practice.navigation_gain', 0.0)
        cfg.set('core.modules.world.input_dim', 11)
        cfg.set('core.extra_features', {'world': ['scent_x']})
        cfg.set('christus.core_readout', {
            'arm': arm,
            'enabled': arm != 'A0',
            'mode': modes[arm],
            'gain': 2.0,
            'lr': 0.05,
            'deadzone': 0.02,
            'shuffle_buffer': 1000,
            'shuffle_kind': 'permute',
            'shuffle_min_lag': 300,
        })
    cfg.ensure_dirs()
    return cfg

class IntegrationBridge(BaseModule):
    name, group = 'integration', 'fast'
    def reset(self):
        self._state = {'event_sequence':0, 'accepted':[], 'experience':[],
                       'outcomes':0, 'memory_updates':0, 'corrections':[],
                       'sources':{}, 'observed_events':{}, 'duplicates':0}
    def on_event(self, event):
        super().on_event(event)
        if event.event_type == EventType.OBSERVATION.value:
            cue = (event.observation or {}).get('derived',{}).get('cue','none')
            self._state['observed_events'][event.event_id] = {'cue':cue,'step':event.step}
        if event.event_type != EventType.OUTCOME.value: return
        info = (event.actual_outcome or {}).get('info',{})
        if 'body_error' not in info: return
        self._state['outcomes'] += 1
        if not self.ctx.memory.enabled: return
        scale = float(self.ctx.get_shared('affect.memory_importance_gain', 1))
        experience = {'event_id':event.event_id, 'step':event.step,
            'source':event.source, 'scope':'simulated:catcher:body',
            'prediction':info['body_prediction'], 'actual':info['body_actual'],
            'error':info['body_error'], 'action':(event.actual_outcome or {}).get('action'),
            'importance':min(1., (.1+info['body_error'])*scale)}
        self._state['experience'].append(experience)
        self._state['experience'] = self._state['experience'][-40:]
        self._state['memory_updates'] += 1
        if self.ctx.memory.store is not None:
            rec = MemoryRecord(memory_id='exp:'+event.event_id, layer='semantic',
                kind='prediction_feedback', key=experience['scope'], content=experience,
                source=event.provenance, confidence=event.confidence,
                importance=experience['importance'], support_count=1,
                scope=experience['scope'], created_step=event.step, updated_step=event.step)
            self.ctx.memory.store.upsert_memory(rec)
    def recent(self, action, step):
        if not self.ctx.memory.enabled: return None
        # Simple applicability guard; no claim of advanced transfer superiority.
        return next((e for e in reversed(self._state['experience'])
                     if e['action']==action and step-e['step']<=40 and
                     e['scope']=='simulated:catcher:body'), None)

class IntegratedPractice(PracticeModule):
    """Existing action-value/exploration policy plus explicit experimental adapter."""
    def _scores(self, cue, scent):
        scores = super()._scores(cue, scent)
        w = self.session_world
        self.last_model_scores = [0.]*len(ACTIONS)
        self.last_model_used = False
        # The post-core readout re-ranks exactly these scores. Keeping the
        # snapshot makes the counterfactual (with vs without readout) auditable
        # without duplicating Practice's Q/navigation logic.
        self.last_scores = [float(v) for v in scores]
        if not self.ctx.config.get('christus.action_model', False): return scores
        if not w.measured.get('visible') or w.body.current.n < 5: return scores
        target, paddle = w.measured['x'], w.measured['paddle']
        belief = float((self.ctx.get_shared('concept.beliefs',{}) or {}).get('ball',0))
        sign = -1. if belief < -.1 and self.ctx.config.get('christus.language_action',False) else 1.
        vals = []
        for i,action in enumerate(ACTIONS):
            signed = COMMAND[action]-1
            predicted = w.body.predict(signed)
            recent = self.bridge.recent(action, self.ctx.clock.step)
            if recent is not None: predicted = .5*predicted + .5*recent['actual']
            improvement = (abs(paddle-target)-abs(paddle+predicted-target)) * sign
            vals.append(float(improvement))
            scores[i] += improvement
        self.last_model_scores = vals
        self.last_model_used = True
        self.last_scores = [float(v) for v in scores]
        return scores

class ChristusSubject(Subject):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        old = self.registry.get('practice')
        self.bridge = IntegrationBridge(old.ctx)
        self.registry.register(self.bridge)
        integrated = IntegratedPractice(old.ctx)
        integrated.restore(old.snapshot())
        integrated.session_world = self.world
        integrated.bridge = self.bridge
        self.registry._modules['practice'] = integrated
        self.readout = CoreActionReadout(old.ctx)
        self.registry.register(self.readout)
        self.readout.attach(self.core, integrated)
        readout_cfg = config.get('christus.core_readout', {}) or {}
        self.readout.enabled = bool(readout_cfg.get('enabled', False))
        # Only an enabled readout is allowed to re-rank Practice's action.
        # A0 leaves this None, so its path is the historical SAFE path plus the
        # (unused) observation bridge present in every arm.
        self.action_hook = self.readout.action if self.readout.enabled else None
        self.planner.enabled = False
        if 'core' in config.get('christus.disabled', []):
            # Ablation only: retain native architecture/serialization, freeze dynamics.
            self.core.step = lambda inputs: CoreMetrics(
                float(np.linalg.norm(self.core.h)), float(np.linalg.norm(self.core.c)),
                0.0, self.core.faults, self.core.step_index)

    def _event(self, event_type, modality, **kwargs):
        e = super()._event(event_type, modality, **kwargs)
        self.bridge._state['event_sequence'] += 1
        return replace(e, event_id=f'{self.subject_state.subject_id}:{self.bridge._state["event_sequence"]}')

    def publish(self, event):
        accepted = self.bridge._state['accepted']
        if event.event_id in accepted:
            self.bridge._state['duplicates'] += 1
            return []
        if event.event_type == EventType.OBSERVATION.value:
            payload = copy.deepcopy(event.observation)
            detected = self.world.measured.get('visible',False)
            payload['derived']['cue'] = 'ball' if detected else 'none'
            event = replace(event, observation=payload)
        if event.event_type == EventType.OUTCOME.value:
            info = (event.actual_outcome or {}).get('info',{})
            if 'body_error' in info:
                event = replace(event, prediction_error=float(info['body_error']),
                                expected_outcome={'body_delta':info['body_prediction']},
                                memory_refs=tuple(e['event_id'] for e in self.bridge._state['experience'][-3:]))
        result = super().publish(event)
        accepted.append(event.event_id)
        if self.bus.errors or any(m.last_error for m in self.registry):
            raise RuntimeError('Integration event failure: '+repr(self.bus.errors)+repr(
                {m.name:m.last_error for m in self.registry if m.last_error}))
        return result

    def correction(self, *, event_id, text, polarity, source, reliability, feedback_id):
        if 'concept_language' in self.disabled_modules or not self.config.get('language.enabled'):
            return {'status':'disabled'}
        if not 0<=reliability<=1: raise ValueError('reliability must be in [0,1]')
        if feedback_id in [x['feedback_id'] for x in self.bridge._state['corrections']]:
            return {'status':'duplicate'}
        evidence = self.bridge._state['observed_events'].get(event_id)
        if not evidence or evidence['cue']=='none': raise ValueError('correction requires an observed referent')
        module = self.registry.get('concept_language')
        previous = module._last_cue
        module._last_cue = evidence['cue']
        try:
            self.publish(self._event(EventType.LANGUAGE_CLAIM.value, 'language',
                source=Source.USER_PROVIDED.value, source_device=source,
                source_reliability=reliability, observation={'words':text.split(),
                    'text':text,'polarity':polarity,'speaker':source,'evidence_id':event_id},
                provenance='christus.explicit_correction'))
        finally: module._last_cue = previous
        module.update(self.clock.step)
        row={'feedback_id':feedback_id,'evidence_id':event_id,'source':source,
             'reliability':reliability,'text':text,'polarity':polarity}
        self.bridge._state['corrections'].append(row)
        self.bridge._state['sources'][source]={'reliability':reliability,'last_evidence':event_id}
        return {'status':'accepted', **row}

def create(out, seed=31100, profile='safe', disabled=(), restore=False, arm=None):
    cfg = configuration(out, seed, profile, disabled, arm=arm)
    store = Store(cfg.resolve_path('paths.database'))
    world = SessionWorld(seed, learn=cfg.get('christus.world02_learning'))
    subject = ChristusSubject(cfg, world=world, store=store, seed=seed, persist=True)
    subject.initialise()
    if restore: subject.restore()
    return subject

def record(subject, result, *, compact=False):
    practice = subject.registry.get('practice')
    readout = subject.registry.get('core_readout') if 'core_readout' in subject.registry else None
    scent = subject.shared.get('world.scent', [0.0, 0.0]) or [0.0, 0.0]
    payload = {**result.to_dict(), 'source_kind':'simulation', **copy.deepcopy(subject.world.last),
        'event_sequence':subject.bridge._state['event_sequence'],
        'memory_updates':subject.bridge._state['memory_updates'],
        'learning_gain':subject.shared.get('affect.learning_gain',1.),
        'exploration':subject.shared.get('practice.epsilon',0),
        'model_used':getattr(practice,'last_model_used',False),
        'model_scores':getattr(practice,'last_model_scores',[]),
        'concept_beliefs':subject.shared.get('concept.beliefs',{}),
        'scent_x':float(scent[0]),
        'scent_y':float(scent[1]),
        'visible':bool(subject.world.measured.get('visible',False)),
        'ball_x':float(subject.world.measured.get('x',0.0)),
        'paddle':float(subject.world.measured.get('paddle',0.0)),
        'readout':readout.metrics() if readout is not None else {},
        'readout_mode':str(subject.config.get('christus.core_readout.mode','off')) if readout is not None else 'off',
        'readout_pred':float(subject.shared.get('readout.pred',0.0)),
        'readout_bias':float(subject.shared.get('readout.bias',0.0)),
        'readout_base_action':subject.shared.get('readout.base_action',None),
        'readout_final_action':subject.shared.get('readout.final_action',None),
        'readout_override':bool(subject.shared.get('readout.override',False)),
        'readout_aligned':bool(subject.shared.get('readout.aligned',False)),
        'readout_error':float(subject.shared.get('readout.error',0.0)),
        'module_health':[h.to_dict() for h in subject.registry.health_report()]}
    if compact:
        # Per-step module health is redundant with the run-end health block and
        # dominates the JSONL size. The formal driver keeps the final health in
        # summary.json and drops the per-step copy.
        payload.pop('module_health', None)
    return payload
