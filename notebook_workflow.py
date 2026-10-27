"""FL-IDS tuning, training and evaluation notebook workflow.

Notebook calls stream a fresh subprocess so TF/TFF graphs and GPU memory are
released between trials. Completed fits/trials are reused. Interrupted fits
restart from their initial seed rather than pretending RNG was checkpointed.
"""
import argparse
import fcntl
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, config as paper_config, dump, digest_file, protocol_digest
from models import MODEL_KINDS, MODEL_BACKENDS, make_model

DEFAULT_CONFIG = ROOT / 'configs/tuning.json'
NAMES = tuple(paper_config()['datasets'])
DEFAULT_SCENARIOS = ['local_' + n for n in NAMES] + ['centralized', 'federated']
METRICS = ('accuracy', 'precision', 'recall', 'f1', 'far', 'auc')
from feature_preprocessing import POLICY as VALIDATION_POLICY


def settings(path=None):
    path = Path(path or DEFAULT_CONFIG).resolve()
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    paper = paper_config()
    expected_anchor = {'architecture':'-'.join(map(str,paper['hidden_units'])),
        **{k:paper[k] for k in ('dropout','batch_size','client_learning_rate','server_learning_rate')}}
    if value['paper_anchor'] != expected_anchor:
        raise ValueError('paper_anchor must remain the published paper configuration.')
    if value['selection'] != 'mean_cross_client_validation_f1':
        raise ValueError('Select equal-weight cross-client Local/Centralized/FL validation F1.')
    if value.get('client_weighting') != 'uniform':
        raise ValueError('FL in this project requires uniform client aggregation weights.')
    if value['threshold'] != .5 or value['seed'] != 42:
        raise ValueError('This workflow fixes threshold 0.5 and seed 42; use a separate study for changes.')
    if value['lstm_backend'] != 'cudnn': raise ValueError('Notebook LSTM requires the GPU-only cuDNN backend.')
    if value['early_stopping'] != {'monitor':'val_loss','patience':5,'min_delta':0.0,'restore_best_weights':True}:
        raise ValueError('Local/Centralized restore minimum-val_loss weights with patience 5; no refit.')
    if not 1 <= value['startup_trials'] <= value['trials_per_model']:
        raise ValueError('Invalid trial budget.')
    if set(value.get('model_tuning', {}))-set(value['models']):
        raise ValueError('Unknown model in model_tuning.')
    if not value['models'] or set(value['models'])-set(MODEL_KINDS):
        raise ValueError('Unknown model in models.')
    for kind in value['models']:
        anchor, space = tuning_plan(value, kind)
        keys = set(expected_anchor) | ({'kernel_size'} if kind == 'cnn' else set())
        if set(anchor)!=keys or set(space)!=keys:
            raise ValueError(f'Incomplete tuning parameters for {kind}.')
        for key, item in anchor.items():
            if item not in space[key]: raise ValueError(f'Anchor outside search: {kind}/{key}')
        if kind == 'cnn' and not set(space['kernel_size']).issubset({3,5}):
            raise ValueError('CNN kernel_size must be 3 or 5.')
    value['_config_path'] = str(path)
    return value


def tuning_plan(cfg, kind):
    override = cfg.get('model_tuning', {}).get(kind, {})
    anchor = dict(override.get('anchor', cfg['paper_anchor']))
    space = {**cfg['search_space'], **override.get('search_space', {})}
    return anchor, space


def run_root(cfg):
    return (ROOT / cfg['run_directory']).resolve()


def data_manifest(cfg):
    from prepare import protocol
    path = run_root(cfg)/'prepared/manifest.json'
    if not path.exists():
        raise FileNotFoundError('Run notebook 01 to prepare Log+Z-score data first.')
    result = json.loads(path.read_text(encoding='utf-8'))
    _, expected_folder, identity = protocol()
    if path.parent.resolve() != expected_folder.resolve() or result['identity'] != identity:
        raise ValueError('Prepared Log+Z-score data differs from the current protocol.')
    return result


def study_identity(cfg):
    clean = {k:v for k,v in cfg.items() if not k.startswith('_')}
    return protocol_digest({'settings':clean, 'data':data_manifest(cfg)['identity'],
                            'workflow_sha256':digest_file(Path(__file__)),
                            'runtime_sha256':digest_file(ROOT/'runtime.py'),
                            'models_sha256':digest_file(ROOT/'models.py'),
                            'metrics_sha256':digest_file(ROOT/'metrics.py')})


def budget(config_path=None):
    c = settings(config_path)
    return {'models':c['models'], 'trials_per_model':c['trials_per_model'],
            'trial_1':{kind:tuning_plan(c,kind)[0] for kind in c['models']},
            'startup':'anchor + 4 random trials with default budget',
            'remaining':'5 TPE trials with default budget',
            'fits_per_trial':f'{len(NAMES)} Local + 1 Centralized + 1 FL',
            'rounds':c['rounds'], 'local_epochs':c['local_epochs'],
            'client_weighting':c['client_weighting'],
            'client_weights':{name:1.0/len(NAMES) for name in NAMES},
            'baseline_epochs':c['baseline_epochs'], 'threshold':c['threshold'],
            'early_stopping':c['early_stopping'],
            'objective':'mean of 12 F1: cross-client Local (6), Centralized (3), FL (3)',
            'tie_break':'higher minimum FL client F1, then mean recall of same twelve rows',
            'same_params_across_scenarios':True,
            'run_root':str(run_root(c)),
            'search_space':{kind:tuning_plan(c,kind)[1] for kind in c['models']},
            'cnn':'39 features -> Conv1D/MaxPool -> Conv1D/Flatten -> Dense -> sigmoid; architecture=filters1-filters2-dense',
            'lstm':'GPU-only CuDNNLSTM, tanh/sigmoid; differs from paper ReLU',
            'final_training':'same fit/validation split as tuning; restore minimum-val_loss weights, no refit'}


def prepare_validation(config_path=None):
    from prepare import prepare
    return prepare(config_path)


def split_table(config_path=None):
    manifest = data_manifest(settings(config_path))
    rows = []
    for name in NAMES:
        info = manifest['datasets'][name]
        for split in ('fit', 'validation', 'test'):
            counts = info[f'{split}_class_counts']
            rows.append({'dataset': name, 'split': split, 'rows': info[f'{split}_rows'],
                         'benign': counts[0], 'attack': counts[1]})
    return pd.DataFrame(rows)


def make_process(kind,params):
    from runtime import tf,tff
    from models import adam
    def model_fn():
        return tff.learning.models.from_keras_model(make_model(kind,params),
            input_spec=(tf.TensorSpec([None,39],tf.float32),tf.TensorSpec([None,1],tf.float32)),
            loss=tf.keras.losses.BinaryCrossentropy(),metrics=[])
    return tff.learning.algorithms.build_weighted_fed_avg(model_fn=model_fn,
            client_optimizer_fn=adam(params['client_learning_rate']),
            server_optimizer_fn=adam(params['server_learning_rate']),
            # One vote per client update, independent of the number of fit rows.
            # Server Adam consumes the mean model delta; it does not just copy
            # the arithmetic mean of the clients' final model weights.
            client_weighting=tff.learning.ClientWeighting.UNIFORM)


def records(folder,names,rows,batch,epochs=1):
    from runtime import tf
    paths=[str((folder/n/'train.bin').resolve()) for n in names]
    # Cache raw records before shuffle, so cached data never freezes epoch order.
    dataset=tf.data.FixedLengthRecordDataset(paths,record_bytes=160).cache()
    def decode_batch(records):
        values=tf.ensure_shape(tf.io.decode_raw(records,tf.float32,little_endian=True),[None,40])
        return values[:,:39],values[:,39:40]
    options=tf.data.Options(); options.experimental_deterministic=True
    options.threading.private_threadpool_size=2; options.threading.max_intra_op_parallelism=1
    return (dataset.shuffle(sum(rows[n] for n in names),seed=42,reshuffle_each_iteration=True)
            .batch(batch).map(decode_batch,num_parallel_calls=2,deterministic=True)
            .repeat(epochs).prefetch(2).with_options(options))


def training_step(model,optimizer):
    """Accumulate sample-weighted loss on GPU; read it only at epoch end."""
    from runtime import tf
    loss_fn=tf.keras.losses.BinaryCrossentropy()
    with tf.device('/GPU:0'):
        total=tf.Variable(0.,dtype=tf.float64,trainable=False)
        count=tf.Variable(0,dtype=tf.int64,trainable=False)
    @tf.function
    def step(x,y):
        with tf.device('/GPU:0'),tf.GradientTape() as tape:
            loss=loss_fn(y,model(x,training=True))
        optimizer.apply_gradients(zip(tape.gradient(loss,model.trainable_variables),model.trainable_variables))
        tf.debugging.assert_all_finite(loss,'Nonfinite loss')
        with tf.device('/GPU:0'):
            size=tf.cast(tf.shape(y)[0],tf.int64)
            total.assign_add(tf.cast(loss,tf.float64)*tf.cast(size,tf.float64))
            count.assign_add(size)
        return loss
    return step,total,count


def evaluate(model,folder,split,destination=None):
    from metrics import binary_metrics
    values={}; attacks=[]
    for name in NAMES:
        with np.load(folder/name/f'{split}.npz') as data:
            y=data['y'].reshape(-1)
            scores=np.concatenate([model(data['x'][i:i+2048],training=False).numpy().reshape(-1)
                                   for i in range(0,len(y),2048)])
            values[name]=binary_metrics(y,scores)
            if destination is not None:
                np.savez(destination/f'{name}_{split}_predictions.npz',y=y,scores=scores,row_id=data['row_id'])
            for category in sorted(np.unique(data['attack'][y==1])):
                mask=(y==1)&(data['attack']==category)
                attacks.append({'dataset':name,'attack':str(category),'support':int(mask.sum()),
                                'recall':float((scores[mask]>=.5).mean())})
    return {'datasets':values,'attack_detection_rates':attacks}


def validation_loss(model,folder,names):
    """BCE on local validation for Local, pooled validation for Centralized."""
    total=0.; count=0
    for name in names:
        with np.load(folder/name/'validation.npz') as data:
            y=data['y'].reshape(-1).astype(np.float64)
            score=np.concatenate([model(data['x'][i:i+2048],training=False).numpy().reshape(-1)
                                  for i in range(0,len(y),2048)]).astype(np.float64)
            score=np.clip(score,1e-7,1-1e-7)
            total+=float((-y*np.log(score)-(1-y)*np.log1p(-score)).sum()); count+=len(y)
    value=total/count
    if not np.isfinite(value): raise FloatingPointError('Nonfinite validation loss')
    return value


def _fit(c,kind,params,scenario,mode,folder,identity):
    from runtime import tf,tff
    destination=folder/scenario; destination.mkdir(parents=True,exist_ok=True)
    result_path=destination/'result.json'
    if result_path.exists():
        result=json.loads(result_path.read_text())
        if result['identity']!=identity: raise ValueError('Completed fit identity differs.')
        print(f'REUSE {kind}/{mode}/{scenario}',flush=True); return result
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(c['seed'])
    # Keep validation held out in every scenario, including final training.
    # There is no full-train refit or separately selected epoch count.
    data=run_root(c)/'prepared'
    rows={n:data_manifest(c)['datasets'][n]['fit_rows'] for n in NAMES}
    if mode=='test' and any(not (data/n/'test.npz').exists() for n in NAMES):
        raise FileNotFoundError('Run notebook 01 to prepare test with the fit-only Log+Z-score scaler.')
    history=[]; seconds=0.; selected_epoch=None; split='validation' if mode=='validation' else 'test'
    if (destination/'history.json').exists(): print('Incomplete fit restarts from initial seed; complete fits are reused.',flush=True)
    if scenario=='federated':
        process=make_process(kind,params); state=process.initialize()
        # TFF builds Keras models inside TF graphs; create the eager evaluator
        # afterwards to avoid invalidating it during graph construction.
        model=make_model(kind,params)
        state=process.set_model_weights(state,tff.learning.models.ModelWeights.from_model(model))
        datasets=[records(data,[n],rows,params['batch_size'],c['local_epochs']) for n in NAMES]
        for number in range(1,c['rounds']+1):
            start=time.monotonic(); state=process.next(state,datasets).state; seconds+=time.monotonic()-start
            process.get_model_weights(state).assign_weights_to(model)
            record={'round':number}
            if mode=='validation':
                values=evaluate(model,data,'validation')['datasets']; record['validation']=values
                message=' | '.join(f'{n}: F1={v["f1"]:.4f}, recall={v["recall"]:.4f}, FAR={v["far"]:.4f}' for n,v in values.items())
            else: message='test reserved for final round'
            history.append(record); dump(destination/'history.json',history)
            print(f'{kind}/{scenario}/{mode}: round {number}/{c["rounds"]} | {message}',flush=True)
    else:
        model=make_model(kind,params)
        names=list(NAMES) if scenario=='centralized' else [scenario.removeprefix('local_')]
        optimizer=tf.keras.optimizers.Adam(params['client_learning_rate'])
        stopping=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=c['early_stopping']['patience'],
                   min_delta=c['early_stopping']['min_delta'],mode='min',
                   restore_best_weights=c['early_stopping']['restore_best_weights'])
        stopping.set_model(model); stopping.on_train_begin()
        step,total,count=training_step(model,optimizer)
        dataset=records(data,names,rows,params['batch_size'])
        for epoch in range(1,c['baseline_epochs']+1):
            start=time.monotonic(); total.assign(0.); count.assign(0)
            for x,y in dataset:
                loss=step(x,y)
                if 'GPU:0' not in loss.device: raise RuntimeError('Training loss not on GPU')
            epoch_loss=float(total/tf.cast(count,tf.float64))
            seconds+=time.monotonic()-start
            monitored=validation_loss(model,data,names)
            record={'epoch':epoch,'loss':epoch_loss,'val_loss':monitored}
            message=''
            if mode=='validation':
                record['validation']=evaluate(model,data,'validation')['datasets']
                message=' | '+' | '.join(f'{n}: F1={v["f1"]:.4f}, recall={v["recall"]:.4f}, FAR={v["far"]:.4f}'
                                          for n,v in record['validation'].items())
            history.append(record); dump(destination/'history.json',history)
            stopping.on_epoch_end(epoch-1,{'val_loss':monitored})
            print(f'{kind}/{scenario}/{mode}: epoch {epoch}/{c["baseline_epochs"]}, loss={epoch_loss:.6f}, '
                  f'val_loss={monitored:.6f}, wait={stopping.wait}/{stopping.patience}{message}',flush=True)
            if model.stop_training:
                print(f'EARLY STOP {kind}/{scenario}: 5 epochs without val_loss improvement; no refit.',flush=True)
                break
        stopping.on_train_end()
        # TF/Keras 2.14 may only restore automatically when patience is reached.
        # Also restore the best weights when the maximum epoch budget is reached.
        model.set_weights(stopping.best_weights)
        selected_epoch=min(history,key=lambda record:record['val_loss'])['epoch']
        print(f'RESTORE {kind}/{scenario}: epoch {selected_epoch} with minimum val_loss; no refit.',flush=True)
    model.save_weights(str(destination/'model.weights.h5'))
    result={'identity':identity,'model':kind,'scenario':scenario,'mode':mode,'params':params,
            'client_weighting':c['client_weighting'] if scenario=='federated' else None,
            'client_weights':{name:1.0/len(NAMES) for name in NAMES} if scenario=='federated' else None,
            'train_seconds':seconds,'evaluation':evaluate(model,data,split,destination),
            'test_evaluated':mode=='test','checkpoint_selection':'minimum val_loss epoch for Local/Centralized, final round for FL',
            'selected_epoch':selected_epoch,
            'completed_steps':len(history),'training_partition':'same held-out-validation fit rows in all scenarios',
            'early_stopping':None if scenario=='federated' else c['early_stopping']}
    dump(result_path,result); return result


def suite(job_path):
    from runtime import configure
    job=json.loads(Path(job_path).read_text()); c=settings(job['config_path'])
    if job['study_identity']!=study_identity(c): raise ValueError('Job config/code/data changed.')
    hardware={**configure(),'model_backend':MODEL_BACKENDS[job['model']]}
    folder=Path(job['destination']); folder.mkdir(parents=True,exist_ok=True)
    scenarios=job.get('scenarios',DEFAULT_SCENARIOS)
    values=[_fit(c,job['model'],job['params'],scenario,job['mode'],folder,job['identity']) for scenario in scenarios]
    selected=[]
    for result in values:
        for target,m in result['evaluation']['datasets'].items():
            if result['scenario'].startswith('local_') and result['scenario']=='local_'+target: continue
            selected.append({'scenario':result['scenario'],'dataset':target,**m})
    result={'identity':job['identity'],'model':job['model'],'params':job['params'],
            'mode':job['mode'],'hardware':hardware,'scored_rows':selected,
            'test_evaluated':job['mode']=='test'}
    if set(scenarios)==set(DEFAULT_SCENARIOS):
        assert len(selected)==len(NAMES)*(len(NAMES)+1)
        result['rank']=[float(np.mean([r['f1'] for r in selected])),
            float(min(r['f1'] for r in selected if r['scenario']=='federated')),
            float(np.mean([r['recall'] for r in selected]))]
    dump(folder/'result.json',result)


def _execute(args,log=None,own_group=True):
    """Stream subprocess output without waiting for inherited pipe EOF."""
    from process_runner import execute
    command=[sys.executable,'-u','-B',str(Path(__file__).resolve()),*map(str,args)]
    return execute(command,log=log,own_group=own_group)


def _job(c,kind,params,mode,folder):
    identity=protocol_digest({'study':study_identity(c),'model':kind,'params':params,'mode':mode})
    folder.mkdir(parents=True,exist_ok=True)
    value={'identity':identity,'study_identity':study_identity(c),'model':kind,'params':params,
           'mode':mode,'destination':str(folder),'config_path':c['_config_path']}
    dump(folder/'job.json',value)
    return value


def _tune(c,kind,max_new):
    import optuna
    if kind not in c['models']: raise ValueError(kind)
    anchor, space = tuning_plan(c,kind)
    folder=run_root(c)/'tuning'/kind; folder.mkdir(parents=True,exist_ok=True)
    identity=study_identity(c); marker=folder/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())['identity']!=identity:
        raise ValueError('Study config/code/data changed. Choose a new run_directory; do not mix trials.')
    dump(marker,{'identity':identity,'budget':budget(c['_config_path']),
                 'paper_anchor':c['paper_anchor'],'tuning_anchor':anchor,'search_space':space})
    sampler_path=folder/'sampler.pkl'
    if sampler_path.exists():
        with sampler_path.open('rb') as stream: sampler=pickle.load(stream)
    else: sampler=optuna.samplers.TPESampler(seed=c['seed'],n_startup_trials=c['startup_trials'])
    storage='sqlite:///'+str((folder/'study.sqlite3').resolve())
    study=optuna.create_study(study_name=kind,direction='maximize',sampler=sampler,storage=storage,load_if_exists=True)
    if not study.trials: study.enqueue_trial(anchor)
    count=0
    while len([t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE])<c['trials_per_model']:
        if max_new is not None and count>=max_new: break
        pending=next((t for t in study.trials if t.state==optuna.trial.TrialState.RUNNING),None)
        if pending is not None:
            if set(pending.params)!=set(space):
                study.tell(pending.number,state=optuna.trial.TrialState.FAIL); continue
            number=pending.number; params=pending.params
        else:
            trial=study.ask(); number=trial.number
            params={name:trial.suggest_categorical(name,choices) for name,choices in space.items()}
            with sampler_path.open('wb') as stream: pickle.dump(study.sampler,stream)
            previous=[t.params for t in study.trials if t.number!=number and t.state==optuna.trial.TrialState.COMPLETE]
            if params in previous:
                study.tell(number,state=optuna.trial.TrialState.PRUNED)
                print('Duplicate proposal skipped without training.',flush=True); continue
        completed=len([t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE])
        print(f'\n{kind.upper()} evaluation {completed+1}/{c["trials_per_model"]}, Optuna trial {number}, params={params}',flush=True)
        target=folder/f'trial_{number:03d}'; _job(c,kind,params,'validation',target)
        # Keep RUNNING on interruption so the same proposal resumes, not replaced.
        _execute(['suite','--job',target/'job.json'],target/'train.log',own_group=False)
        result=json.loads((target/'result.json').read_text())
        assert not result['test_evaluated'] and len(result['scored_rows'])==len(NAMES)*(len(NAMES)+1)
        study.tell(number,result['rank'][0]); count+=1
        print(f'COMPLETE score mean F1={result["rank"][0]:.6f}, min FL F1={result["rank"][1]:.6f}',flush=True)
        _write_best(c,kind,study)
    return _write_best(c,kind,study)


def _write_best(c,kind,study):
    import optuna
    folder=run_root(c)/'tuning'/kind
    complete=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
    rows=[]; results=[]
    for trial in complete:
        result=json.loads((folder/f'trial_{trial.number:03d}/result.json').read_text()); results.append((trial.number,result))
        rows.append({'trial':trial.number,'score':result['rank'][0],'min_fl_f1':result['rank'][1],**trial.params})
    pd.DataFrame(rows).to_csv(folder/'trials.csv',index=False)
    if not results: return {'completed_trials':0,'required_trials':c['trials_per_model']}
    number,result=max(results,key=lambda pair:tuple(pair[1]['rank']))
    ready=len(complete)>=c['trials_per_model']
    selected={'study_identity':study_identity(c),'model':kind,'trial':number,'params':result['params'],
              'rank':result['rank'],'completed_trials':len(complete),'required_trials':c['trials_per_model'],
              'ready_for_final':ready,'test_used_for_selection':False,
              'objective':'mean of 12 three-client comparison F1 values; Local/Centralized at minimum val_loss epoch, FL at final round'}
    dump(folder/('best_params.json' if ready else 'current_best.json'),selected)
    return selected


def tune(model,config_path=None,max_new_evaluations=None):
    c=settings(config_path); data_manifest(c)
    if max_new_evaluations is not None and max_new_evaluations<1: raise ValueError('max_new_evaluations must be positive or None')
    args=['tune','--config',c['_config_path'],'--model',model]
    if max_new_evaluations is not None: args+=['--max-new',max_new_evaluations]
    _execute(args)
    folder=run_root(c)/'tuning'/model
    path=folder/'best_params.json'
    if not path.exists(): path=folder/'current_best.json'
    return json.loads(path.read_text())


def selected_params(c,kind):
    path=run_root(c)/'tuning'/kind/'best_params.json'
    if not path.exists(): raise RuntimeError(f'{kind}: complete all {c["trials_per_model"]} trials in notebook 02 first.')
    chosen=json.loads(path.read_text())
    if chosen['study_identity']!=study_identity(c) or not chosen['ready_for_final'] or chosen['test_used_for_selection']:
        raise ValueError('Selected configuration no longer matches this study.')
    return chosen


def _final(c,kind,scenarios):
    chosen=selected_params(c,kind)
    folder=run_root(c)/'results'/kind
    _job(c,kind,chosen['params'],'test',folder)
    dump(folder/'selection.json',chosen)
    job=json.loads((folder/'job.json').read_text()); job['scenarios']=scenarios; dump(folder/'job.json',job)
    _execute(['suite','--job',folder/'job.json'],folder/'train.log',own_group=False)


def run_experiments(model,config_path=None,scenarios=None):
    c=settings(config_path); selected_params(c,model)
    args=['final','--config',c['_config_path'],'--model',model]
    if scenarios is not None: args+=['--scenarios',','.join(scenarios)]
    _execute(args)
    return report(config_path)


def tuning_table(model,config_path=None):
    c=settings(config_path); path=run_root(c)/'tuning'/model/'trials.csv'
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def tuning_details(model,config_path=None):
    c=settings(config_path); rows=[]
    for path in sorted((run_root(c)/'tuning'/model).glob('trial_*/result.json')):
        result=json.loads(path.read_text())
        rows.extend({'trial':int(path.parent.name.split('_')[1]),**r} for r in result['scored_rows'])
    return pd.DataFrame(rows)


def report(config_path=None):
    c=settings(config_path); directory=run_root(c)/'results'; rows=[]; attacks=[]
    for path in sorted(directory.glob('*/*/result.json')):
        result=json.loads(path.read_text()); scenario=result['scenario']
        if result['mode']!='test': continue
        for target,metrics in result['evaluation']['datasets'].items():
            source=scenario.removeprefix('local_') if scenario.startswith('local_') else 'all_clients'
            rows.append({'model':result['model'],'scenario':'local' if scenario.startswith('local_') else scenario,
                         'source':source,'test_dataset':target,**metrics})
        attacks.extend({'model':result['model'],'scenario':scenario,**r} for r in result['evaluation']['attack_detection_rates'])
    frame=pd.DataFrame(rows)
    if frame.empty: return frame
    directory.mkdir(parents=True,exist_ok=True)
    frame.to_csv(directory/'all_evaluations.csv',index=False)
    pd.DataFrame(attacks).to_csv(directory/'attack_detection_rates.csv',index=False)
    return frame


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['tune','suite','final'])
    parser.add_argument('--config',default=str(DEFAULT_CONFIG)); parser.add_argument('--model',choices=list(MODEL_KINDS))
    parser.add_argument('--max-new',type=int); parser.add_argument('--job'); parser.add_argument('--scenarios')
    args=parser.parse_args()
    if args.action=='suite': return suite(args.job)
    c=settings(args.config); folder=run_root(c); folder.mkdir(parents=True,exist_ok=True)
    (ROOT/'runs').mkdir(parents=True,exist_ok=True)
    with (folder/'execution.lock').open('a+b') as lock,(ROOT/'runs/execution.lock').open('a+b') as gpu_lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another notebook/experiment is using this GPU run.') from None
        if args.action=='tune': print(json.dumps(_tune(c,args.model,args.max_new),indent=2),flush=True)
        else: _final(c,args.model,args.scenarios.split(',') if args.scenarios else DEFAULT_SCENARIOS)


if __name__=='__main__': main()
