from hyperopt import fmin, atpe, space_eval, hp, STATUS_OK, Trials
import pickle
import argparse
import hpo_main


hpo_trials = Trials()
hpo_space = {'lr': hp.quniform('lr', 3, 19, 1),'drop_in': hp.uniform('drop_in',0.2, 0.5), 'drop_out': hp.uniform('drop_out',0.2, 0.5), 
                'val_q': hp.uniform('val_q',0.01, 0.5), 'apperture': hp.quniform('apperture', 0, 2000, 1)}
best = fmin(hpo_main.objective_func, hpo_space, algo=atpe.suggest, max_evals = 30,  verbose=False, trials=hpo_trials)

with open('hpo_trial.pkl', 'wb') as f:
    pickle.dump(hpo_trials, f)