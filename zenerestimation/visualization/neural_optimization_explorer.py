"""Read-only neural optimization evidence explorer; never trains models."""
from __future__ import annotations
import json
import math
from pathlib import Path

MODELS = ('LSTM', 'GRU', 'LinearTrendLSTM', 'KalmanLSTM')


def _load_json(path: Path) -> dict:
    def no_duplicates(pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                raise ValueError(f'Duplicate JSON key: {k}')
            d[k] = v
        return d
    def no_constant(v):
        raise ValueError(f'Nonfinite JSON token: {v}')
    result = json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=no_duplicates, parse_constant=no_constant)
    if not isinstance(result, dict):
        raise ValueError('Expected JSON object')
    return result


def load_neural_run(folder: Path) -> dict:
    """Validate paired artifacts; do not infer unavailable data."""
    found = {}
    for model in MODELS:
        ap, rp = folder / f'{model}_neural_training.json', folder / f'{model}_learning_report.json'
        if not ap.exists() and not rp.exists():
            continue
        if not ap.is_file() or not rp.is_file():
            raise ValueError(f'Incomplete artifact pair: {model}')
        a, r = _load_json(ap), _load_json(rp)
        if (a.get('schema_version') != 'neural-training-artifact-v1' or
            a.get('scope') != 'internal_temporal_neural_training_only' or
            r.get('schema_version') != 'neural-candidate-report-v1' or
            a.get('model') != model or r.get('model') != model):
            raise ValueError(f'Artifact identity/schema mismatch: {model}')
        if a.get('best_params') != r.get('best_params') or not math.isclose(float(a['best_score']),float(r['best_score']),abs_tol=1e-10):
            raise ValueError(f'Selection mismatch: {model}')
        ac, rc = a.get('candidates'), r.get('candidates')
        if not isinstance(ac,list) or not isinstance(rc,list) or len(ac)!=len(rc):
            raise ValueError(f'Candidate count mismatch: {model}')
        for x,y in zip(ac,rc):
            if x.get('candidate_index') != y.get('candidate_index') or x.get('params') != y.get('params') or not math.isclose(float(x['score']),float(y['score']),abs_tol=1e-10):
                raise ValueError(f'Candidate mismatch: {model}')
        found[model] = (a,r)
    return found


def render_neural_optimization_explorer(repository_root: Path) -> None:
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    st.header('Neural Optimization Explorer')
    st.caption('Development-only temporal validation. Read-only evidence; NOT final holdout accuracy or frozen benchmark ranking.')
    base = Path(repository_root) / 'results' / '732B-5610410'
    folders = [p for p in (base/'neural_demo_16_4_a',base/'neural_expanded_16_4') if p.is_dir()]
    if not folders:
        st.info('No neural optimization artifacts found. Run the demonstration or expanded optimization first.')
        return
    folder = st.selectbox('Optimization run',folders,format_func=lambda p:p.name,key='neural_run')
    try:
        found = load_neural_run(folder)
    except (ValueError,KeyError,TypeError,OSError) as exc:
        st.error(f'Neural artifact validation failed: {exc}')
        return
    if not found:
        st.info('No complete neural model artifact pairs in selected run.')
        return
    model = st.selectbox('Neural model',list(found),key='neural_model')
    a,r = found[model]
    st.write('Selected parameters:',r['best_params'])
    st.metric('Best internal validation RMSE (µV)',f"{float(r['best_score']):.6f}")
    rows=[]
    for m,(_,report) in found.items():
        for c in report['candidates']:
            rows.append({'model':m,'candidate':f"C{c['candidate_index']}",'window':c['params'].get('window'),'RMSE (µV)':float(c['metrics']['rmse'])})
    st.subheader('Candidate RMSE comparison')
    df=pd.DataFrame(rows)
    st.plotly_chart(px.bar(df,x='candidate',y='RMSE (µV)',color='model',barmode='group',hover_data=['window']),width="stretch")
    st.dataframe(df,hide_index=True)
    candidate=st.selectbox('Candidate',a['candidates'],format_func=lambda c:f"C{c['candidate_index']} · {c['params']}",key='neural_candidate')
    folds=candidate.get('fold_training',[])
    if not folds:
        st.warning('Training histories unavailable for this candidate.')
        return
    indices=[f.get('fold_index') for f in folds]
    fold_id=st.selectbox('Validation fold (1-based)',indices,key='neural_fold')
    fold=next(f for f in folds if f.get('fold_index')==fold_id)
    history=fold.get('history') or {}
    epochs=history.get('epochs') or []
    loss=[{'epoch':e['epoch'],'training loss':e['metrics']['loss']} for e in epochs if e.get('metrics',{}).get('loss') is not None]
    if loss:
        st.subheader('Training loss by epoch')
        st.plotly_chart(px.line(pd.DataFrame(loss),x='epoch',y='training loss',markers=True),width="stretch")
    else:
        st.warning('Training loss unavailable.')
    diagnostics=fold.get('diagnostics') or {}
    st.write('Estimated learning onset epoch:',diagnostics.get('estimated_loss_improvement_onset_epoch','Unavailable'))
    st.write('Best training-loss epoch:',diagnostics.get('best_training_loss_epoch','Unavailable'))
    st.write('Training plateau at end:',diagnostics.get('training_plateau_at_end','Unavailable'))
    if diagnostics.get('best_validation_loss') is None:
        st.info('Validation loss unavailable: validation RMSE is recorded separately; do not interpret it as validation loss.')
    else:
        st.write('Best validation loss:',diagnostics['best_validation_loss'])
    if history.get('training_started_at') is None:
        st.info('Training start time unavailable: learning onset is an epoch estimate, not a wall-clock timestamp.')
    else:
        st.write('Recorded training start:',history['training_started_at'])
    if history.get('duration_seconds') is None:
        st.info('Training duration/timing unavailable: event recorder was not enabled for this run.')
    else:
        st.write('Recorded duration (seconds):',history['duration_seconds'])
    st.subheader('Full historical and future model curves')
    st.info('Unavailable in training artifacts. Requires separately generated, date-aligned fitted values and forecasts; no curves are inferred or fabricated.')
    st.caption('This explorer never accesses external measured holdout targets or retrains models.')
