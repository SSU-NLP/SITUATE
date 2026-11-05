"""Deterministic state-to-change targets, isolated from model inference."""
import numpy as np
from . import annotation_prompts as p

def targets(states, profile='pretrained-i2'):
    if profile != 'pretrained-i2':
        raise ValueError('Unknown label profile')
    if set(states) != set(p.AXES):
        raise ValueError('Exactly five axes are required')
    n = len(states['protagonist'])
    if n < 2 or any(len(states[a]) != n for a in p.AXES):
        raise ValueError('Aligned state sequences of at least two sentences are required')
    axis = {a:{i:dict(row) for i,row in enumerate(states[a])} for a in p.AXES}
    axis['space'] = p.carry_forward(axis['space'], 'location', 'unspecified')
    rules = [p.reset_protagonist_present,p.reset_space,p.reset_time,p.reset_causality,
             lambda a,b:int(a['active_goal']=='none' and b['active_goal']!='none')]
    resets = np.array([[rule(axis[a][i-1],axis[a][i]) for a,rule in zip(p.AXES,rules)]
                       for i in range(1,n)],dtype=np.int64)
    absolute = np.array([[p.PERSPECTIVE.index(states['protagonist'][i]['perspective']),
                          p.TEMPORAL_SHIFT.index(states['time'][i]['temporal_shift_marker']),
                          p.EVENT_TYPE.index(states['causality'][i]['event_type'])]
                         for i in range(n)],dtype=np.int64)
    return resets, absolute
