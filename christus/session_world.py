"""WORLD02 Session adapter: the WPHR Subject is the sole decision maker.

Session owns execution; its policy here is an actuator/event adapter, not a
second autonomous agent. Only images and proprioceptive positions feed models.
"""
from __future__ import annotations
import copy
from dataclasses import asdict
import numpy as np
from catcher.session import Session
from catcher.env import EpisodeConfig
from catcher.detect import detect_ball
from catcher.world_cognition import Tracked, BodyRel, ExternalRel, fit
from wphr.world.base import WorldObservation, WorldResult

COMMAND = {'move_west': 0, 'move_east': 2, 'move_north': 1,
           'move_south': 1, 'interact': 1, 'wait': 1}

class RecentRelation(Tracked):
    """WORLD02 regression/representation, recent-experience baseline only.

    Candidate confirmation and phase-labelled retrieval are NOT defaulted on.
    Deduplication uses a global monotonically increasing transition sequence.
    """
    def __init__(self, cls, name):
        super().__init__(cls, name, 'persist')
        self.last_event = -1

    def update_recent(self, x, y, eid, enabled=True):
        if eid <= self.last_event:
            return False
        self.last_event = eid
        if not enabled:
            return False
        self.samples.append((x, float(y)))
        self.samples = self.samples[-40:]
        if len(self.samples) >= 5:
            fit(self.current, self.samples)
            self.updates += 1
        return True

    def export(self):
        return {'samples': copy.deepcopy(self.samples), 'coef': self.current.coef.tolist(),
                'n': self.current.n, 'updates': self.updates, 'last_event': self.last_event}

    def restore(self, s):
        self.samples = copy.deepcopy(s['samples'])
        self.current.coef = np.asarray(s['coef'], float)
        self.current.n = s['n']
        self.updates, self.last_event = s['updates'], s['last_event']

class SessionPort:
    name = 'christus_actuator_port'
    def __init__(self, owner): self.owner = owner
    def reset(self): pass
    def on_world_pre(self, obs, paddle):
        o = self.owner
        o.before = float(paddle)
        o.pred_body = o.body.predict(o.command - 1)
        o.body_version = o.body.updates
    def act(self, obs): return self.owner.command
    def on_world_post(self, obs, action, before, after, contact, t):
        o = self.owner
        o.actual_body = float(after - before)
        o.body_error = abs(o.pred_body - o.actual_body)
        o.body.update_recent(action - 1, o.actual_body, o.total + 1, o.learn)
    def add_step_reward(self, reward): pass
    def finish_episode(self, reward): pass
    def diagnostics(self): return {'skill': 'catch', 'source': 'WPHR.Subject'}
    def state(self): return {}
    def load_state(self, state): pass

class SessionWorld:
    world_id = 'christus_session'
    def __init__(self, seed=31100, learn=True):
        self.seed, self.learn = seed, learn
        self.total, self.episode, self.energy = 0, -1, 1.0
        self.body = RecentRelation(BodyRel, 'body')
        self.external = RecentRelation(ExternalRel, 'external')
        self.port = SessionPort(self)
        self.command = 1
        self.last = {}
        self.previous_visual = None
        self.observation = None
        self.measured = {}
        self.external_pending = None
        self.reset()

    def actions(self): return tuple(COMMAND)

    def reset(self, seed=None):
        if seed is not None: self.seed = seed
        self.episode += 1
        self.session = Session(EpisodeConfig(seed=self.seed + self.episode, bar_width=8.0), self.port)
        self.previous_visual = None
        self.external_pending = None
        self.observation = None

    @property
    def done(self): return False  # adapter handles episodes, never resets the Subject

    def observe(self):
        if self.observation is not None: return self.observation
        frame = self.session.world.observe()
        d = detect_ball(frame)
        x, y = (d.x, d.y) if d.present else (0.0, 0.0)
        vx = 0.0
        if d.present and self.previous_visual is not None:
            pt, px, py = self.previous_visual
            vx = (x - px) / max(1, self.total - pt)
        self.measured = {'x': x, 'y': y, 'visible': d.present,
                         'paddle': float(self.session.world.paddle), 'vx': vx}
        vals = [self.measured['paddle']/64, y/64, self.energy,
                (x-self.measured['paddle'])/64 if d.present else 0.0, 0.0]
        self.observation = WorldObservation('vision', vals, self.energy, self.total,
            source_device='simulated:WORLD02.Session', confidence=d.quality if d.present else 0.0,
            meta={'source_kind': 'simulation', 'detection': d.as_dict(), 'episode': self.episode})
        return self.observation

    def execute(self, action):
        if action not in COMMAND: raise ValueError(action)
        self.observe()
        before_visual = copy.deepcopy(self.measured)
        self.command = COMMAND[action]
        external_features = None
        ext_pred = None
        if before_visual['visible']:
            external_features = [before_visual['x']/64, before_visual['y']/64,
                                 before_visual['vx']/4, self.command-1]
            ext_pred = self.external.predict(external_features)
        # Evaluator-only perturbation, never passed as model inputs or phase labels.
        gain = 1.0 if self.total < 400 or self.total >= 750 else 0.45
        self.session.world.cfg.gain = gain
        old_cost = self.session.move_cost
        result = self.session.step()  # actual WORLD02 pre/action/post hooks
        cost = self.session.move_cost - old_cost
        self.total += 1
        self.energy = float(np.clip(self.energy - cost*.1 + (.003 if self.command == 1 else 0), .05, 1))
        ext_error = None
        # Detect actual post-action frame, never use ball ground-truth as a label.
        d = detect_ball(self.session.world.observe())
        if before_visual['visible'] and d.present:
            ext_error = abs(ext_pred-d.x/64)
            self.external.update_recent(external_features, d.x/64, self.total, self.learn)
        terminal = result['outcome']
        reward = -cost + (1.0 if terminal == 'caught' else (-1.0 if terminal else 0))
        self.last = {'sequence': self.total, 'episode': self.episode, 'command': self.command,
                     'body_prediction': self.pred_body, 'body_actual': self.actual_body,
                     'body_error': self.body_error, 'body_version_before': self.body_version,
                     'body_updates': self.body.updates, 'external_prediction': ext_pred,
                     'external_error': ext_error, 'external_updates': self.external.updates,
                     'cost': cost, 'outcome': terminal, 'source_kind': 'simulation',
                     'observed': before_visual}
        if before_visual['visible']:
            self.previous_visual = (self.total-1, before_visual['x'], before_visual['y'])
        self.observation = None
        if terminal: self.reset()
        return WorldResult(action, True, reward, self.energy, terminal=='caught',
                           {'cue':'ball' if before_visual['visible'] else 'none', **self.last})

    def snapshot(self):
        s = self.session.state()
        # Preserve fields omitted by the legacy Session serializer, without
        # changing the archived source implementation.
        extra = {k: copy.deepcopy(v) for k,v in vars(self.session).items()
                 if k not in ('world','policy','cfg','frames')}
        return {'seed': self.seed, 'total': self.total, 'episode': self.episode,
                'energy': self.energy, 'learn': self.learn, 'body': self.body.export(),
                'external': self.external.export(), 'session': s, 'session_extra':extra,
                'world_cfg':asdict(self.session.world.cfg), 'last':copy.deepcopy(self.last),
                'previous_visual':self.previous_visual, 'measured':copy.deepcopy(self.measured),
                'command':self.command,
                'observation':None if self.observation is None else asdict(self.observation)}

    def restore(self, s):
        for k in ('seed','total','episode','energy','learn','last','previous_visual','measured','command'):
            setattr(self, k, copy.deepcopy(s[k]))
        self.body.restore(s['body']); self.external.restore(s['external'])
        self.session = Session.from_state(s['session'], self.port)
        for k,v in s['session_extra'].items(): setattr(self.session,k,copy.deepcopy(v))
        self.session.world.cfg = EpisodeConfig(**s['world_cfg'])
        self.session.cfg = self.session.world.cfg
        self.observation = None if s['observation'] is None else WorldObservation(**s['observation'])
