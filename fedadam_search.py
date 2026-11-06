"""CNN sample-weighted FedAdam: 20 server LRs on validation, then one final test."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from common import ROOT, dump, digest_file, protocol_digest
from models import make_model, adam
import notebook_workflow as baseline

DEFAULT_CONFIG = ROOT/'configs/fedadam_weighted.json'


def settings(path=None):
    path = Path(path or DEFAULT_CONFIG).resolve()
    options = json.loads(path.read_text(encoding='utf-8-sig'))
    required = {'model':'cnn', 'task':'binary', 'scenario':'fedadam_weighted',
        'client_weighting':'number_of_examples', 'server_optimizer':'adam',
        'selection':'mean_federated_client_validation_attack_f1',
        'client_learning_rate_policy':'keep_selected_cnn_client_learning_rate',
        'checkpoint_selection':'final_round',
        'test_policy':'select_on_validation_then_retrain_from_scratch_and_test_once'}
    if any(options.get(k) != v for k,v in required.items()):
        raise ValueError('Only CNN binary sample-weighted FedAdam with validation selection is supported.')
    values = options['server_learning_rates']
    if not values or len(set(values)) != len(values) or any(
            not np.isfinite(v) or v <= 0 for v in values):
        raise ValueError('Invalid or duplicate server_learning_rates.')
    if options['expected_trials'] != 20 or len(values) != 20:
        raise ValueError('This experiment requires exactly 20 distinct server learning rates.')
    if 'client_learning_rates' in options or 'client_learning_rate' in options:
        raise ValueError('Client LR must remain the one from the selected CNN.')
    for key in ('server_beta_1','server_beta_2'):
        if not np.isfinite(options[key]) or not 0 <= options[key] < 1:
            raise ValueError(f'Invalid {key}.')
    if not np.isfinite(options['server_epsilon']) or options['server_epsilon'] <= 0:
        raise ValueError('Invalid server_epsilon.')
    if options['tie_break'] != ['minimum_client_validation_f1','mean_client_validation_recall',
                               'lower_mean_validation_far','earlier_trial']:
        raise ValueError('Unexpected tie-break policy.')
    options['_config_path'] = str(path)
    return options


def source(options):
    c = baseline.settings(ROOT/options['baseline_config'])
    chosen = baseline.selected_params(c,'cnn','binary')
    return c,chosen


def search_identity(options,c,chosen):
    return protocol_digest({'options':{k:v for k,v in options.items() if not k.startswith('_')},
        'baseline_study':baseline.study_identity(c),'frozen_cnn_selection':chosen,
        'experiment_code_sha256':digest_file(Path(__file__))})


def search_folder(c):
    return baseline.run_root(c)/'tuning/fedadam_weighted/cnn'


def destination(c):
    return baseline.run_root(c)/'results/binary/cnn/fedadam_weighted'


def parameters(chosen):
    return dict(chosen['params'])


def validation_score(evaluation):
    # Each client has one vote for model selection, independent of sample counts.
    # Training aggregation still uses NUM_EXAMPLES.
    metrics = list(evaluation['datasets'].values())
    if len(metrics) != 3:
        raise ValueError('Expected exactly three validation clients.')
    return {'mean_validation_f1':float(np.mean([m['f1'] for m in metrics])),
        'minimum_validation_f1':float(min(m['f1'] for m in metrics)),
        'mean_validation_recall':float(np.mean([m['recall'] for m in metrics])),
        'mean_validation_far':float(np.mean([m['far'] for m in metrics])),
        'pooled_validation_f1':float(evaluation['pooled']['f1'])}


def ranking(result):
    s = result['validation_score']
    return (s['mean_validation_f1'],s['minimum_validation_f1'],s['mean_validation_recall'],
            -s['mean_validation_far'],-result['trial'])


def make_process(c,params,options,server_rate):
    from runtime import tf,tff
    def model_fn():
        return tff.learning.models.from_keras_model(make_model('cnn',params,'binary',c['seed']),
            input_spec=(tf.TensorSpec([None,39],tf.float32),tf.TensorSpec([None,1],tf.float32)),
            loss=tf.keras.losses.BinaryCrossentropy(),metrics=[])
    return tff.learning.algorithms.build_weighted_fed_avg(model_fn=model_fn,
        client_optimizer_fn=adam(params['client_learning_rate'],c),
        server_optimizer_fn=tff.learning.optimizers.build_adam(
            learning_rate=server_rate,beta_1=options['server_beta_1'],
            beta_2=options['server_beta_2'],epsilon=options['server_epsilon']),
        client_weighting=tff.learning.ClientWeighting.NUM_EXAMPLES)


def completed_trials(options,c,chosen):
    identity = search_identity(options,c,chosen)
    results = []
    for number,server_rate in enumerate(options['server_learning_rates'],1):
        path = search_folder(c)/f'trial_{number:03d}/result.json'
        if not path.exists():
            continue
        r = json.loads(path.read_text(encoding='utf-8'))
        if (r['identity'] != identity or r['trial'] != number or r['mode'] != 'validation'
                or r['test_evaluated'] or r['completed_steps'] != c['rounds']
                or r['params'] != parameters(chosen) or r['server_learning_rate'] != server_rate):
            raise ValueError(f'Existing trial does not match the frozen search: {path}')
        results.append(r)
    return results


def write_selection(options,c,chosen):
    results = completed_trials(options,c,chosen)
    if not results:
        return None
    best = max(results,key=ranking)
    selected = {'identity':search_identity(options,c,chosen),'model':'cnn','task':'binary',
        'scenario':'fedadam_weighted','trial':best['trial'],'params':best['params'],
        'server_learning_rate':best['server_learning_rate'],'validation_score':best['validation_score'],
        'completed_trials':len(results),'required_trials':20,'ready_for_final':len(results)==20,
        'selection':options['selection'],'tie_break':options['tie_break'],
        'test_used_for_selection':False,'baseline_selection':chosen}
    dump(search_folder(c)/'best_params.json',selected)
    rows = [{'trial':r['trial'],'client_learning_rate':r['params']['client_learning_rate'],
        'server_learning_rate':r['server_learning_rate'],**r['validation_score'],
        'selected':r['trial']==best['trial']} for r in sorted(results,key=ranking,reverse=True)]
    frame = pd.DataFrame(rows)
    frame.insert(0,'rank',range(1,len(frame)+1))
    frame.to_csv(search_folder(c)/'trials.csv',index=False)
    dump(search_folder(c)/'progress.json',{'completed_trials':len(results),'total_trials':20,
        'best_trial':best['trial'],'best_validation_f1':best['validation_score']['mean_validation_f1'],
        'ready_for_final':len(results)==20,'test_used_for_selection':False})
    return selected


def _train(options,c,chosen,mode,number=None):
    identity = search_identity(options,c,chosen)
    if mode == 'validation':
        if number is None or not 1 <= number <= 20:
            raise ValueError('Provide a trial number in 1..20.')
        server_rate = options['server_learning_rates'][number-1]
        params = parameters(chosen)
        folder = search_folder(c)/f'trial_{number:03d}'
        selection = None
    elif mode == 'test':
        selection = write_selection(options,c,chosen)
        if selection is None or not selection['ready_for_final']:
            raise RuntimeError('Complete all 20 validation trials before final training/test.')
        number = selection['trial'];params = selection['params']
        server_rate = selection['server_learning_rate'];folder = destination(c)
    else:
        raise ValueError('Expected validation or test mode.')
    folder.mkdir(parents=True,exist_ok=True)
    marker = folder/'result.json'
    if marker.exists():
        r = json.loads(marker.read_text(encoding='utf-8'))
        if (r['identity'] != identity or r['mode'] != mode or r['trial'] != number
                or r['params'] != params or r['server_learning_rate'] != server_rate):
            raise ValueError('Completed result identity/selection differs.')
        return r
    dump(folder/'selection.json',selection or {'trial':number,'params':params,
        'server_learning_rate':server_rate,'mode':'validation','test_used_for_selection':False})
    from runtime import configure,tf,tff
    hardware = configure()
    tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(c['seed'])
    process = make_process(c,params,options,server_rate)
    state = process.initialize()
    model = make_model('cnn',params,'binary',c['seed'])
    state = process.set_model_weights(state,tff.learning.models.ModelWeights.from_model(model))
    initial = hashlib.sha256(b''.join(v.tobytes() for v in model.get_weights())).hexdigest()
    control = baseline.run_root(c)/'results/binary/cnn/fedavg_weighted/result.json'
    if control.exists():
        previous = json.loads(control.read_text(encoding='utf-8'))
        frozen = {k:v for k,v in previous['params'].items() if k != 'client_learning_rate'}
        if frozen != {k:v for k,v in params.items() if k != 'client_learning_rate'} or previous['initial_weights_sha256'] != initial:
            raise ValueError('Architecture/initialization differ from the existing CNN control.')
    names = list(baseline.data_manifest(c)['datasets'])
    datasets = [baseline.records(baseline.run_root(c)/'prepared',[n],'binary',
        params['batch_size'],c['local_epochs'],c['seed']) for n in names]
    history = [];seconds = 0.
    for current in range(1,c['rounds']+1):
        start = time.monotonic();state = process.next(state,datasets).state
        seconds += time.monotonic()-start
        process.get_model_weights(state).assign_weights_to(model)
        if not all(np.isfinite(x).all() for x in model.get_weights()):
            raise FloatingPointError('Nonfinite FedAdam model parameters.')
        history.append({'round':current,'train_seconds':seconds})
        dump(folder/'history.json',history)
        print(f'CNN FedAdam {mode} trial {number}: round {current}/{c["rounds"]}',flush=True)
    model.save_weights(str(folder/'model.weights.h5'))
    # Validation trials never read test partitions or save test predictions.
    split = 'validation' if mode == 'validation' else 'test'
    evaluation = baseline.evaluate(model,c,'binary',split,folder if mode=='test' else None)
    result = {'identity':identity,'baseline_study_identity':chosen['study_identity'],
        'task':'binary','model':'cnn','scenario':'fedadam_weighted','mode':mode,'trial':number,
        'params':params,'client_optimizer':'adam','server_optimizer':'adam',
        'server_learning_rate':server_rate,'server_beta_1':options['server_beta_1'],
        'server_beta_2':options['server_beta_2'],'server_epsilon':options['server_epsilon'],
        'client_weighting':'number_of_examples','client_weights':baseline.client_weights(c,'number_of_examples'),
        'initial_weights_sha256':initial,'train_seconds':seconds,'completed_steps':len(history),
        'rounds':c['rounds'],'local_epochs':c['local_epochs'],'checkpoint_selection':'final_round',
        'selected_epoch':None,'hardware':hardware,'evaluation':evaluation,
        'evaluation_split':split,'test_evaluated':mode=='test','test_used_for_selection':False,
        'new_hyperparameter_tuning':True,'tuned_parameters':['server_learning_rate'],
        'client_learning_rate_fixed':True,
        'retrained_from_scratch':True}
    if mode == 'validation':
        result['validation_score'] = validation_score(evaluation)
    else:
        result['selection'] = selection
        validation_run = json.loads((search_folder(c)/f'trial_{number:03d}/result.json').read_text())
        if validation_run['initial_weights_sha256'] != initial:
            raise ValueError('Final training initialization differs from validation.')
        result['validation_weights_match_final'] = (digest_file(folder/'model.weights.h5') ==
            digest_file(search_folder(c)/f'trial_{number:03d}/model.weights.h5'))
    dump(marker,result)
    if mode == 'test':
        baseline.report(c['_config_path'])
    return result


def _execute_worker(options,mode,folder,number=None):
    from process_runner import execute
    command = [sys.executable,'-u','-B',str(Path(__file__).resolve()),'--config',options['_config_path'],
               '--mode',mode]
    if number is not None:
        command += ['--trial',str(number)]
    folder.mkdir(parents=True,exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        execute(command,folder/'train.log')


def run(config_path=None,show_progress=True):
    import fcntl
    options = settings(config_path);c,chosen = source(options)
    folder = search_folder(c);folder.mkdir(parents=True,exist_ok=True)
    identity = search_identity(options,c,chosen)
    protocol_path = folder/'protocol.json'
    with (folder/'execution.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This FedAdam LR search is already running.') from None
        if protocol_path.exists() and json.loads(protocol_path.read_text())['identity'] != identity:
            raise ValueError('Search code/config/data changed; use a new search directory.')
        dump(protocol_path,{'identity':identity,'options':{k:v for k,v in options.items() if not k.startswith('_')},
            'baseline_selection':chosen,'rounds':c['rounds'],'local_epochs':c['local_epochs'],
            'candidates':[{'trial':i,'server_learning_rate':rate}
                          for i,rate in enumerate(options['server_learning_rates'],1)],
            'fixed_client_learning_rate':chosen['params']['client_learning_rate'],'test_used_for_selection':False})
        done = {r['trial'] for r in completed_trials(options,c,chosen)}
        for number,server_rate in enumerate(options['server_learning_rates'],1):
            if number not in done:
                if show_progress:
                    print(f'FedAdam CNN: server LR trial {number}/20, LR={server_rate:g}; client LR fixed',flush=True)
                dump(folder/'active_trial.json',{'trial':number,'client_learning_rate':chosen['params']['client_learning_rate'],
                    'server_learning_rate':server_rate,'status':'running_validation','test_evaluated':False})
                _execute_worker(options,'validation',folder/f'trial_{number:03d}',number)
            best = write_selection(options,c,chosen)
            if show_progress:
                print(f'Completed {best["completed_trials"]}/20; best mean validation F1={best["validation_score"]["mean_validation_f1"]:.5f}',flush=True)
        selected = write_selection(options,c,chosen)
        if not selected['ready_for_final']:
            raise RuntimeError('Search is incomplete; test is reserved.')
        dump(folder/'active_trial.json',{'trial':selected['trial'],'status':'final_training_after_validation_selection',
            'client_learning_rate':selected['params']['client_learning_rate'],
            'server_learning_rate':selected['server_learning_rate'],'completed_trials':20})
        if show_progress:
            print(f'Selected trial {selected["trial"]}: fixed client LR={selected["params"]["client_learning_rate"]:g}, '
                  f'server LR={selected["server_learning_rate"]:g}; retrain from scratch, then test once.',flush=True)
        _execute_worker(options,'test',destination(c))
        result = json.loads((destination(c)/'result.json').read_text(encoding='utf-8'))
        if result['identity'] != identity:
            raise ValueError('Completed final experiment identity differs.')
        dump(folder/'active_trial.json',{'trial':selected['trial'],'status':'complete','completed_trials':20,
            'final_test_evaluated':True,'test_used_for_selection':False})
        return result


def tuning_table(config_path=None):
    options = settings(config_path);c,chosen = source(options)
    if completed_trials(options,c,chosen):
        return pd.read_csv(search_folder(c)/'trials.csv')
    return pd.DataFrame()


def summary(result):
    values = {**result['evaluation']['datasets'],'pooled':result['evaluation']['pooled']}
    return pd.DataFrame([{'test_client':n,**{k:m[k] for k in
        ('samples','accuracy','precision','recall','f1','far','auc','pr_auc')}} for n,m in values.items()])


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(DEFAULT_CONFIG))
    parser.add_argument('--mode',choices=['validation','test','search'],default='search')
    parser.add_argument('--trial',type=int)
    args = parser.parse_args()
    if args.mode == 'search':
        run(args.config);return
    options = settings(args.config);c,chosen = source(options)
    folder = baseline.run_root(c);folder.mkdir(parents=True,exist_ok=True)
    shared = ROOT.parent/'runs/execution.lock';shared.parent.mkdir(parents=True,exist_ok=True)
    with (folder/'execution.lock').open('a+b') as lock,shared.open('a+b') as gpu_lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another experiment is using this GPU.') from None
        _train(options,c,chosen,args.mode,args.trial)


if __name__ == '__main__':
    main()
