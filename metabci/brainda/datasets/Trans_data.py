# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def transform_data(arr4d, arr4y):

    X = np.delete(arr4d, (0), axis=0)
    y = np.delete(arr4y, (0), axis=0)

    trial_id = []
    subject = []
    session = []
    run = []
    dataset = []
    Event = []
    for i in range(len(X)):
        trial_id.append(i)
        subject.append(1)
        session.append('session_0')
        run.append('run_0')
        dataset.append('mieeg')
        if y[i] == 1:
            Event.append('Rest')
        else:
            Event.append('SRF')

    state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(state)
    np.random.shuffle(y)
    np.random.shuffle(Event)
    np.random.shuffle(trial_id)

    df = pd.DataFrame(
        {'subject': subject, 'session': session, 'run': run, 'event': Event, 'trial_id': trial_id,
         'dataset': dataset})

    return X, y, df