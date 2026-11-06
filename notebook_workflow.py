"""Tune and evaluate Local, Centralized and sample-weighted FedAvg models."""
import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT, SPLITS, dump, digest_file, protocol_digest
from models import MODEL_KINDS, MODEL_BACKENDS, make_model

DEFAULT_CONFIG = ROOT/'configs/tuning.json'
ACTIVE_TASKS = ('binary',)


def settings(path=None):
    path = Path(path or DEFAULT_CONFIG).resolve()
    c = json.loads(path.read_text(encoding='utf-8-sig'))
    if c['client_optimizer'] != 'adam': raise ValueError('The requested client optimizer is Adam.')
    if (c['server_optimizer'], c['server_learning_rate'], c['server_momentum']) != ('sgd', 1.0, 0.0):
        raise ValueError('FedAvg requires direct server averaging: SGD lr=1, momentum=0; server Adam is postponed.')
    if c['tuning_weighting'] != 'number_of_examples' or c['final_weightings'] != ['number_of_examples']:
        raise ValueError('Use sample-count weighting for FedAvg tuning and final evaluation.')
    if c['selection'] != 'mean_cross_client_validation_f1' or c['checkpoint_selection'] != 'final_round':
        raise ValueError('Use the original mean of 12 cross-client validation F1 values.')
    if c['threshold'] != .5: raise ValueError('Binary threshold is fixed at 0.5.')
    if c['tasks'] != list(ACTIVE_TASKS): raise ValueError('Only binary classification is enabled.')
    if c.get('tuning_task') not in c['tasks']: raise ValueError('Set one tuning_task from the enabled tasks.')
    if not c['models'] or set(c['models'])-set(MODEL_KINDS): raise ValueError('Unknown model.')
    if not 1 <= c['startup_trials'] <= c['trials_per_model']: raise ValueError('Invalid trial budget.')
    if min(c[k] for k in ('rounds', 'local_epochs', 'baseline_epochs')) < 1: raise ValueError('Invalid epoch budget.')
    for kind in c['models']:
        plan = c['model_tuning'][kind]
        expected = {'architecture', 'dropout', 'batch_size', 'client_learning_rate'}
        if kind == 'cnn': expected.add('kernel_size')
        if set(plan['anchor']) != expected or set(plan['search_space']) != expected:
            raise ValueError('Search space includes unsupported or missing parameters; no server LR tuning.')
        if any(v not in plan['search_space'][k] for k, v in plan['anchor'].items()):
            raise ValueError('Anchor outside search space.')
    c['_config_path'] = str(path)
    return c


def run_root(c): return ROOT/c['run_directory']


def data_manifest(c):
    from prepare import settings_base, data_identity
    path = run_root(c)/'prepared/manifest.json'
    if not path.exists(): raise FileNotFoundError('Run prepare first.')
    m = json.loads(path.read_text(encoding='utf-8'))
    if m['identity'] != data_identity(c, settings_base(c)):
        raise ValueError('Prepared data/config/code changed; create a separate study.')
    return m


def study_identity(c):
    return protocol_digest({'settings': {k:v for k,v in c.items() if not k.startswith('_')},
        'data': data_manifest(c)['identity'], 'code': {f:digest_file(ROOT/f) for f in
            ('notebook_workflow.py', 'models.py', 'metrics.py', 'runtime.py')}})


def client_weights(c, weighting):
    m = data_manifest(c)
    sizes = {n: v['fit_rows'] for n,v in m['datasets'].items()}
    if weighting == 'number_of_examples': return {n:v/sum(sizes.values()) for n,v in sizes.items()}
    raise ValueError(weighting)


def budget(config_path=None):
    c = settings(config_path)
    result = {k:c[k] for k in ('tasks','tuning_task','models','trials_per_model','rounds','local_epochs',
        'baseline_epochs','client_optimizer','server_optimizer','server_learning_rate','tuning_weighting',
        'final_weightings','selection','checkpoint_selection')}
    result['total_tuning_trials'] = len(c['models'])*c['trials_per_model']
    result['fits_per_trial'] = '3 Local + 1 Centralized + 1 sample-weighted FedAvg'
    result['objective'] = 'mean of 12 validation F1: 6 cross-client Local + 3 Centralized + 3 FL'
    result['trial_1'] = {kind:c['model_tuning'][kind]['anchor'] for kind in c['models']}
    return result


def tuning_tasks(c, requested='binary'):
    """Return the enabled binary tuning task."""
    if requested not in c['tasks']: raise ValueError('Disabled task.')
    return [c['tuning_task']]


def tuning_scenarios(names):
    return ['local_'+n for n in names]+['centralized','fedavg_weighted']


def score_tuning_results(results, names):
    """Original selection: omit Local self-evaluation, average the remaining 12 F1."""
    if len(results) != len(names)+2 or {r['scenario'] for r in results} != set(tuning_scenarios(names)):
        raise ValueError('Tuning requires three Local models, Centralized and sample-weighted FedAvg.')
    scored = []
    for result in results:
        if result['test_evaluated']: raise ValueError('Test cannot be used for tuning.')
        for target, metrics in result['evaluation']['datasets'].items():
            if target not in names: raise ValueError('Unknown validation client.')
            if result['scenario'] == 'local_'+target: continue
            scored.append({'scenario':result['scenario'],'dataset':target,
                **{k:metrics[k] for k in ('accuracy','precision','recall','f1','far','auc')}})
    if len(scored) != len(names)*(len(names)+1): raise ValueError('Incomplete validation comparisons.')
    rank = [float(np.mean([r['f1'] for r in scored])),
        float(min(r['f1'] for r in scored if r['scenario'] == 'fedavg_weighted')),
        float(np.mean([r['recall'] for r in scored]))]
    return scored,rank


def prepare_validation(config_path=None):
    from prepare import prepare
    return prepare(config_path)


def split_table(config_path=None):
    m = data_manifest(settings(config_path))
    return pd.DataFrame([{'client':n,'split':s,**v['splits'][s]} for n,v in m['datasets'].items() for s in SPLITS])


def make_process(c, kind, params, task):
    from runtime import tf, tff
    from models import adam
    def model_fn():
        spec = tf.TensorSpec([None,1],tf.float32)
        loss = tf.keras.losses.BinaryCrossentropy()
        return tff.learning.models.from_keras_model(make_model(kind,params,task,c['seed']),
            input_spec=(tf.TensorSpec([None,39],tf.float32),spec),loss=loss,metrics=[])
    return tff.learning.algorithms.build_weighted_fed_avg(model_fn=model_fn,
        client_optimizer_fn=adam(params['client_learning_rate'],c),
        server_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=1.0),
        client_weighting=tff.learning.ClientWeighting.NUM_EXAMPLES)


def records(folder, names, task, batch, epochs=1, seed=42):
    from runtime import tf
    xs = [str(Path(folder)/n/'fit_x.bin') for n in names]
    ys = [str(Path(folder)/n/f'fit_y_{task}.bin') for n in names]
    count = sum(Path(p).stat().st_size//156 for p in xs)
    features = tf.data.FixedLengthRecordDataset(xs,record_bytes=156,buffer_size=8*1024*1024)
    targets = tf.data.FixedLengthRecordDataset(ys,record_bytes=4,buffer_size=1024*1024)
    ds = tf.data.Dataset.zip((features, targets))
    def decode(x,y):
        x = tf.ensure_shape(tf.io.decode_raw(x,tf.float32),[None,39])
        y = tf.reshape(tf.io.decode_raw(y,tf.int32),[-1])
        y = tf.cast(tf.reshape(y,[-1,1]),tf.float32)
        return x,y
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    opts.threading.private_threadpool_size = 2
    opts.threading.max_intra_op_parallelism = 1
    return (ds.shuffle(min(count,100000),seed=seed,reshuffle_each_iteration=True).batch(batch)
        .map(decode,num_parallel_calls=2,deterministic=True).repeat(epochs).prefetch(2).with_options(opts))


def predictions(model, x, task):
    values = np.concatenate([model(x[i:i+2048],training=False).numpy() for i in range(0,len(x),2048)])
    return values.reshape(-1)


def evaluate(model, c, task, split, destination=None):
    from prepare import load_partition
    from metrics import classification_metrics
    m = data_manifest(c)
    folder = run_root(c)/'prepared'
    per_client, all_y, all_scores, attacks = {}, [], [], []
    for name in m['datasets']:
        x,y = load_partition(folder,name,split,task)
        scores = predictions(model,x,task)
        per_client[name] = classification_metrics(y,scores,task)
        all_y.append(np.asarray(y)); all_scores.append(scores)
        pred = (scores >= .5).astype(int)
        with np.load(folder/name/f'{split}_meta.npz') as z:
            if destination is not None:
                np.savez_compressed(Path(destination)/f'{name}_{split}_predictions.npz',
                    y=y,scores=scores,row_id=z['row_id'])
            for code in np.unique(z['attack_code'][z['y_binary'] == 1]):
                mask = z['attack_code'] == code
                unambiguous = mask & ~z['attack_metadata_ambiguous']
                attacks.append({'client':name,'attack':m['source']['attack_vocabulary'][int(code)],
                    'support':int(mask.sum()),'detection_recall':float(np.mean(pred[mask] != 0)),
                    'ambiguous_source_attack_inputs':int((mask & z['attack_metadata_ambiguous']).sum()),
                    'unambiguous_support':int(unambiguous.sum()),
                    'unambiguous_detection_recall':float(np.mean(pred[unambiguous] != 0)) if unambiguous.any() else None})
    pooled = classification_metrics(np.concatenate(all_y),np.concatenate(all_scores),task)
    return {'datasets':per_client,'pooled':pooled,'attack_detection_rates':attacks}


def validation_loss(model,c,task,names):
    from prepare import load_partition
    total = count = 0
    for name in names:
        x,y = load_partition(run_root(c)/'prepared',name,'validation',task)
        p = np.clip(predictions(model,x,task).astype(float),1e-7,1-1e-7)
        losses = -y*np.log(p)-(1-y)*np.log1p(-p)
        total += float(losses.sum()); count += len(y)
    return total/count


def training_step(model, optimizer, task):
    from runtime import tf
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    @tf.function
    def step(x,y):
        with tf.GradientTape() as tape: loss = loss_fn(y,model(x,training=True))
        tf.debugging.assert_all_finite(loss,'Nonfinite training loss')
        optimizer.apply_gradients(zip(tape.gradient(loss,model.trainable_variables),model.trainable_variables))
        return loss
    return step


def _fit(c,kind,params,task,scenario,mode,folder,identity):
    if task not in c['tasks']: raise ValueError('Disabled task.')
    names = list(data_manifest(c)['datasets'])
    if scenario not in tuning_scenarios(names): raise ValueError(f'Unknown scenario: {scenario}')
    destination = Path(folder)/scenario
    destination.mkdir(parents=True,exist_ok=True)
    marker = destination/'result.json'
    if marker.exists():
        result = json.loads(marker.read_text(encoding='utf-8'))
        if result['identity'] != identity: raise ValueError('Completed run identity differs.')
        print(f'REUSE {task}/{kind}/{scenario}',flush=True)
        return result
    from runtime import tf,tff
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(c['seed'])
    history, seconds, selected_epoch = [], 0., None
    weighting = 'number_of_examples' if scenario == 'fedavg_weighted' else None
    if weighting:
        process = make_process(c,kind,params,task)
        state = process.initialize()
        model = make_model(kind,params,task,c['seed'])
        state = process.set_model_weights(state,tff.learning.models.ModelWeights.from_model(model))
    else:
        model = make_model(kind,params,task,c['seed'])
    initial_hash = hashlib.sha256(b''.join(v.tobytes() for v in model.get_weights())).hexdigest()
    data = run_root(c)/'prepared'
    if weighting:
        datasets = [records(data,[n],task,params['batch_size'],c['local_epochs'],c['seed']) for n in names]
        for number in range(1,c['rounds']+1):
            start = time.monotonic()
            state = process.next(state,datasets).state
            seconds += time.monotonic()-start
            process.get_model_weights(state).assign_weights_to(model)
            record = {'round':number}
            if mode == 'validation':
                record['validation'] = evaluate(model,c,task,'validation')
                message = f'validation Macro-F1={record["validation"]["pooled"]["macro_f1"]:.5f}'
            else: message = 'test reserved for final round'
            history.append(record); dump(destination/'history.json',history)
            print(f'{task}/{kind}/{scenario}: round {number}/{c["rounds"]}; {message}',flush=True)
    else:
        local_names = names if scenario == 'centralized' else [scenario.removeprefix('local_')]
        optimizer = tf.keras.optimizers.Adam(params['client_learning_rate'],beta_1=c['adam_beta1'],
            beta_2=c['adam_beta2'],epsilon=c['adam_epsilon'])
        step = training_step(model,optimizer,task)
        ds = records(data,local_names,task,params['batch_size'],seed=c['seed'])
        best_loss, best_weights, wait = float('inf'), None, 0
        for epoch in range(1,c['baseline_epochs']+1):
            start = time.monotonic(); total = count = 0
            for x,y in ds:
                loss = float(step(x,y)); size = int(x.shape[0])
                total += loss*size; count += size
            seconds += time.monotonic()-start
            val_loss = validation_loss(model,c,task,local_names)
            history.append({'epoch':epoch,'loss':total/count,'val_loss':val_loss})
            dump(destination/'history.json',history)
            if val_loss < best_loss-c['early_stopping']['min_delta']:
                best_loss,best_weights,selected_epoch,wait = val_loss,model.get_weights(),epoch,0
            else: wait += 1
            print(f'{task}/{kind}/{scenario}: epoch {epoch}, val_loss={val_loss:.6f}',flush=True)
            if wait >= c['early_stopping']['patience']: break
        model.set_weights(best_weights)
    model.save_weights(str(destination/'model.weights.h5'))
    result = {'identity':identity,'task':task,'model':kind,'scenario':scenario,'mode':mode,'params':params,
        'client_optimizer':'adam','server_optimizer':'sgd_lr1_no_momentum' if weighting else None,
        'client_weighting':weighting,'client_weights':client_weights(c,weighting) if weighting else None,
        'initial_weights_sha256':initial_hash,'train_seconds':seconds,'completed_steps':len(history),
        'selected_epoch':selected_epoch,'checkpoint_selection':'final_round' if weighting else 'minimum_validation_loss',
        'evaluation':evaluate(model,c,task,'validation' if mode == 'validation' else 'test',destination),
        'test_evaluated':mode == 'test','test_used_for_selection':False}
    dump(marker,result)
    return result


def _execute(args,log=None,own_group=True):
    from process_runner import execute
    return execute([sys.executable,'-u','-B',str(Path(__file__).resolve()),*map(str,args)],log,own_group)


def _job(c,kind,task,params,mode,folder,scenarios):
    folder.mkdir(parents=True,exist_ok=True)
    identity = protocol_digest({'study':study_identity(c),'model':kind,'task':task,'params':params,'mode':mode})
    job = {'identity':identity,'study_identity':study_identity(c),'model':kind,'task':task,'params':params,
        'mode':mode,'destination':str(folder),'config_path':c['_config_path'],'scenarios':scenarios}
    dump(folder/'job.json',job)
    return job


def suite(job_path):
    from runtime import configure
    job = json.loads(Path(job_path).read_text(encoding='utf-8'))
    c = settings(job['config_path'])
    if job['study_identity'] != study_identity(c): raise ValueError('Job identity changed.')
    hardware = configure()
    results = [_fit(c,job['model'],job['params'],job['task'],s,job['mode'],job['destination'],job['identity'])
               for s in job['scenarios']]
    output = {'identity':job['identity'],'task':job['task'],'model':job['model'],'params':job['params'],
        'mode':job['mode'],'hardware':hardware,'test_evaluated':job['mode'] == 'test'}
    if job['mode'] == 'validation':
        names = list(data_manifest(c)['datasets'])
        output['scored_rows'],output['rank'] = score_tuning_results(results,names)
    dump(Path(job['destination'])/'result.json',output)


def _write_best(c,kind,task,study):
    import optuna
    folder = run_root(c)/'tuning'/task/kind
    results = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            result = json.loads((folder/f'trial_{trial.number:03d}/result.json').read_text(encoding='utf-8'))
            results.append((trial.number,result))
    pd.DataFrame([{'trial':n,'score':r['rank'][0],'min_fl_f1':r['rank'][1],**r['params']}
        for n,r in results]).to_csv(folder/'trials.csv',index=False)
    if not results: return {'completed_trials':0,'required_trials':c['trials_per_model']}
    number,result = max(results,key=lambda pair:tuple(pair[1]['rank']))
    ready = len(results) >= c['trials_per_model']
    chosen = {'study_identity':study_identity(c),'task':task,'model':kind,'trial':number,
        'params':result['params'],'rank':result['rank'],'completed_trials':len(results),
        'required_trials':c['trials_per_model'],'ready_for_final':ready,'test_used_for_selection':False,
        'tuning_scenario':'fedavg_weighted','tuning_scenarios':tuning_scenarios(list(data_manifest(c)['datasets'])),
        'objective':c['selection']}
    dump(folder/('best_params.json' if ready else 'current_best.json'),chosen)
    return chosen


def _tune(c,kind,task,max_new):
    import optuna
    if kind not in c['models'] or task not in c['tasks']: raise ValueError('Disabled model/task.')
    if task != c['tuning_task']: raise ValueError('Only the configured tuning_task has its own study.')
    folder = run_root(c)/'tuning'/task/kind
    folder.mkdir(parents=True,exist_ok=True)
    marker = folder/'protocol.json'
    identity = study_identity(c)
    if marker.exists() and json.loads(marker.read_text(encoding='utf-8'))['identity'] != identity:
        raise ValueError('Study changed; choose a new run_directory.')
    dump(marker,{'identity':identity,'budget':budget(c['_config_path'])})
    sampler_path = folder/'sampler.pkl'
    if sampler_path.exists():
        with sampler_path.open('rb') as stream: sampler = pickle.load(stream)
    else: sampler = optuna.samplers.TPESampler(seed=c['seed'],n_startup_trials=c['startup_trials'])
    study = optuna.create_study(study_name=f'{task}_{kind}',direction='maximize',sampler=sampler,
        storage='sqlite:///'+str((folder/'study.sqlite3').resolve()),load_if_exists=True)
    plan = c['model_tuning'][kind]
    if not study.trials: study.enqueue_trial(plan['anchor'])
    count = 0
    while sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials) < c['trials_per_model']:
        if max_new is not None and count >= max_new: break
        pending = next((t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING),None)
        if pending is not None:
            if set(pending.params) != set(plan['search_space']):
                study.tell(pending.number,state=optuna.trial.TrialState.FAIL); continue
            number,params = pending.number,pending.params
        else:
            trial = study.ask()
            number = trial.number
            params = {k:trial.suggest_categorical(k,v) for k,v in plan['search_space'].items()}
            with sampler_path.open('wb') as stream: pickle.dump(study.sampler,stream)
            if any(t.state == optuna.trial.TrialState.COMPLETE and t.params == params for t in study.trials):
                study.tell(number,state=optuna.trial.TrialState.PRUNED); continue
        target = folder/f'trial_{number:03d}'
        _job(c,kind,task,params,'validation',target,tuning_scenarios(list(data_manifest(c)['datasets'])))
        print(f'TUNE {task}/{kind} trial {number}: {params}',flush=True)
        _execute(['suite','--job',target/'job.json'],target/'train.log',own_group=False)
        result = json.loads((target/'result.json').read_text(encoding='utf-8'))
        if result['test_evaluated'] or len(result['scored_rows']) != 12:
            raise ValueError('Tuning must use 12 validation comparisons and no test data.')
        study.tell(number,result['rank'][0]); count += 1
        _write_best(c,kind,task,study)
    return _write_best(c,kind,task,study)


def tune(model,config_path=None,max_new_evaluations=None,task='binary'):
    c = settings(config_path)
    if task not in c['tasks']: raise ValueError('Disabled task.')
    args = ['tune','--config',c['_config_path'],'--model',model,'--task',task]
    if max_new_evaluations is not None:
        if max_new_evaluations < 1: raise ValueError('max_new_evaluations must be positive.')
        args += ['--max-new',max_new_evaluations]
    _execute(args)
    folder = run_root(c)/'tuning'/task/model
    path = folder/'best_params.json'
    if not path.exists(): path = folder/'current_best.json'
    chosen = json.loads(path.read_text(encoding='utf-8'))
    return chosen


def selected_params(c,kind,task='binary'):
    if task not in c['tasks']: raise ValueError('Disabled task.')
    path = run_root(c)/'tuning'/task/kind/'best_params.json'
    if not path.exists(): raise RuntimeError(f'Complete tuning for {task}/{kind} first.')
    chosen = json.loads(path.read_text(encoding='utf-8'))
    if (chosen['study_identity'] != study_identity(c) or not chosen['ready_for_final']
            or chosen['test_used_for_selection'] or chosen['task'] != task or chosen['model'] != kind):
        raise ValueError('Chosen parameters no longer match this study.')
    return chosen


def _final(c,kind,task,scenarios):
    chosen = selected_params(c,kind,task)
    names = list(data_manifest(c)['datasets'])
    allowed = tuning_scenarios(names)
    scenarios = allowed if scenarios is None else scenarios
    if not scenarios or set(scenarios)-set(allowed): raise ValueError('Unknown scenario.')
    folder = run_root(c)/'results'/task/kind
    _job(c,kind,task,chosen['params'],'test',folder,scenarios)
    dump(folder/'selection.json',chosen)
    _execute(['suite','--job',folder/'job.json'],folder/'train.log',own_group=False)


def run_experiments(model,config_path=None,scenarios=None,task='binary'):
    c = settings(config_path)
    selected_params(c,model,task)
    args = ['final','--config',c['_config_path'],'--model',model,'--task',task]
    if scenarios is not None: args += ['--scenarios',','.join(scenarios)]
    _execute(args)
    return report(config_path)


def tuning_table(model,config_path=None,task='binary'):
    c = settings(config_path)
    if task not in c['tasks']: raise ValueError('Disabled task.')
    path = run_root(c)/'tuning'/c['tuning_task']/model/'trials.csv'
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def report(config_path=None):
    from prepare import settings_base
    c = settings(config_path)
    scenarios = {*tuning_scenarios(settings_base(c)['clients']), 'fedadam_weighted'}
    directory = run_root(c)/'results'
    rows, per_class, attacks, confusion = [], [], [], []
    for path in sorted(directory.glob('*/*/*/result.json')):
        result = json.loads(path.read_text(encoding='utf-8'))
        if result['mode'] != 'test' or result['task'] not in c['tasks'] or result['scenario'] not in scenarios:
            continue
        prefix = {k:result[k] for k in ('task','model','scenario','client_weighting')}
        for target,m in {**result['evaluation']['datasets'],'pooled':result['evaluation']['pooled']}.items():
            rows.append({**prefix,'test_client':target,**{k:v for k,v in m.items() if not isinstance(v,(dict,list))}})
            per_class.extend({**prefix,'test_client':target,'class':label,**v} for label,v in m['per_class'].items())
            confusion.append({**prefix,'test_client':target,'class_names':m['class_names'],'matrix':m['confusion_matrix']})
        attacks.extend({**prefix,**r} for r in result['evaluation']['attack_detection_rates'])
    frame = pd.DataFrame(rows)
    if not frame.empty:
        directory.mkdir(parents=True,exist_ok=True)
        frame.to_csv(directory/'all_evaluations.csv',index=False)
        pd.DataFrame(per_class).to_csv(directory/'per_class_metrics.csv',index=False)
        pd.DataFrame(attacks).to_csv(directory/'attack_detection_rates.csv',index=False)
        dump(directory/'confusion_matrices.json',confusion)
    return frame


def main():
    import fcntl
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['tune','suite','final'])
    parser.add_argument('--config',default=str(DEFAULT_CONFIG))
    parser.add_argument('--model',choices=list(MODEL_KINDS))
    parser.add_argument('--task',choices=list(ACTIVE_TASKS),default='binary')
    parser.add_argument('--max-new',type=int)
    parser.add_argument('--job'); parser.add_argument('--scenarios')
    args = parser.parse_args()
    if args.action == 'suite': return suite(args.job)
    c = settings(args.config)
    folder = run_root(c); folder.mkdir(parents=True,exist_ok=True)
    # Shared with the original checkout: avoid concurrently occupying its GPU.
    gpu_lock_path = ROOT.parent/'runs/execution.lock'
    gpu_lock_path.parent.mkdir(parents=True,exist_ok=True)
    with (folder/'execution.lock').open('a+b') as lock,gpu_lock_path.open('a+b') as gpu_lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another experiment is using this GPU.') from None
        if args.action == 'tune': print(json.dumps(_tune(c,args.model,args.task,args.max_new),indent=2),flush=True)
        else: _final(c,args.model,args.task,args.scenarios.split(',') if args.scenarios else None)


if __name__ == '__main__': main()
