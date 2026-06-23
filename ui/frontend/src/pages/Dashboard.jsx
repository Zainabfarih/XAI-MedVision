import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { api, DEMO_METRICS } from '../api/client'
import './Dashboard.css'

const METRIC_KEYS   = ['accuracy','auc_roc','precision','recall','f1']
const METRIC_LABELS = { accuracy:'Accuracy', auc_roc:'AUC-ROC',
  precision:'Precision (Nodule)', recall:'Recall (Nodule)', f1:'F1 (Nodule)' }

function pct(v)  { return v!=null ? `${(v*100).toFixed(2)}%` : '—' }
function fmt4(v) { return v!=null ? v.toFixed(4) : '—' }

function MiniBar({ value, color='var(--primary)' }) {
  return (
    <div style={{display:'flex',alignItems:'center',gap:8}}>
      <div style={{flex:1,height:7,background:'var(--g200)',borderRadius:4,overflow:'hidden'}}>
        <div style={{width:`${Math.min((value||0)*100,100)}%`,height:'100%',background:color,borderRadius:4,transition:'width .8s ease'}}/>
      </div>
      <span style={{fontFamily:'var(--mono)',fontSize:12,fontWeight:700,color,minWidth:46,textAlign:'right'}}>{pct(value)}</span>
    </div>
  )
}

function ModelCard({ name, m, isTop }) {
  const [open,setOpen] = useState(false)
  const cm = m.confusion_matrix || {}
  return (
    <div className={`mcard${isTop?' mcard-top':''}`}>
      <div className="mcard-hd" onClick={()=>setOpen(o=>!o)}>
        <div style={{display:'flex',alignItems:'center',gap:10,flex:1,minWidth:0}}>
          <div>
            <div style={{fontWeight:700,fontSize:14,color:'var(--dark)'}}>{name}</div>
            <div style={{fontSize:11,color:'var(--g400)',marginTop:2}}>{m.architecture}</div>
          </div>
        </div>
        <div style={{display:'flex',alignItems:'center',gap:14,flexShrink:0}}>
          <div style={{textAlign:'right'}}>
            <div style={{fontSize:11,color:'var(--g400)'}}>AUC-ROC</div>
            <b style={{fontFamily:'var(--mono)',fontSize:16,color:isTop?'var(--primary)':'var(--g700)'}}>{pct(m.auc_roc)}</b>
          </div>
          <div style={{textAlign:'right'}}>
            <div style={{fontSize:11,color:'var(--g400)'}}>F1</div>
            <b style={{fontFamily:'var(--mono)',fontSize:16,color:isTop?'var(--primary)':'var(--g700)'}}>{pct(m.f1)}</b>
          </div>
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"
            style={{color:'var(--g300)',transform:open?'rotate(180deg)':'none',transition:'transform .2s',flexShrink:0}}>
            <polyline points="6 9 12 15 18 9"/>
          </svg>
        </div>
      </div>

      <div className="mcard-metrics">
        {METRIC_KEYS.map(k=>(
          <div key={k} className="metric-row">
            <span className="metric-key">{METRIC_LABELS[k]}</span>
            <MiniBar value={m[k]} color={isTop?'var(--primary)':'var(--g400)'}/>
          </div>
        ))}
      </div>

      {open && (
        <div className="mcard-expand">
          <div className="expand-section">
            <p className="expand-title">Per-Class Metrics</p>
            <div className="per-class-grid">
              <div className="pc-col">
                <div className="pc-label nodule-label">Nodule class</div>
                {[['Precision',m.precision],['Recall',m.recall],['F1 Score',m.f1],['Specificity',m.specificity]].map(([l,v])=>(
                  <div key={l} className="pc-row"><span>{l}</span><b style={{fontFamily:'var(--mono)',color:'var(--danger)'}}>{pct(v)}</b></div>
                ))}
              </div>
              <div className="pc-col">
                <div className="pc-label healthy-label">No Nodule class</div>
                {[['Precision',m.precision_no_nodule],['Recall',m.recall_no_nodule],['F1 Score',m.f1_no_nodule],['MCC',m.mcc]].map(([l,v])=>(
                  <div key={l} className="pc-row"><span>{l}</span><b style={{fontFamily:'var(--mono)',color:'var(--success)'}}>{fmt4(v)}</b></div>
                ))}
              </div>
            </div>
          </div>

          {cm.TP!=null && (
            <div className="expand-section">
              <p className="expand-title">Confusion Matrix (test set — 3,065 samples)</p>
              <div className="cm-wrap">
                <table className="cm-table">
                  <thead><tr><th/><th>Pred: Nodule</th><th>Pred: Healthy</th></tr></thead>
                  <tbody>
                    <tr><th>Actual: Nodule</th><td className="cm-tp">TP = {cm.TP}</td><td className="cm-fn">FN = {cm.FN}</td></tr>
                    <tr><th>Actual: Healthy</th><td className="cm-fp">FP = {cm.FP}</td><td className="cm-tn">TN = {cm.TN}</td></tr>
                  </tbody>
                </table>
                <div className="cm-legend">
                  <div><span className="cm-dot" style={{background:'#dcfce7'}}/>TP/TN correct</div>
                  <div><span className="cm-dot" style={{background:'#fee2e2'}}/>FP/FN errors</div>
                </div>
              </div>
            </div>
          )}

          <div className="expand-section">
            <p className="expand-title">Training Details</p>
            <div className="train-rows">
              {[
                ['Strategy',         m.training_strategy],
                ['Epochs',           m.training_epochs+(m.probe_epochs?` pretrain + ${m.probe_epochs} probe`:'')],
                ['Best Val AUC',     fmt4(m.val_auc_best_epoch)],
                m.pretrain_loss_final!=null&&['Pretrain loss (final)', m.pretrain_loss_final],
                ['XAI Deletion AUC', fmt4(m.faithfulness_deletion_auc)],
              ].filter(Boolean).map(([k,v])=>(
                <div key={k} className="train-row"><span>{k}</span><span style={{fontFamily:'var(--mono)',fontWeight:600}}>{v}</span></div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default function Dashboard() {
  const [data,   setData]   = useState(DEMO_METRICS)
  const [loading,setLoading]= useState(true)
  const [sec,    setSec]    = useState('models')

  useEffect(()=>{
    api.metrics().then(d=>setData(d||DEMO_METRICS)).catch(()=>setData(DEMO_METRICS)).finally(()=>setLoading(false))
  },[])

  const models    = data.models           || {}
  const faith     = data.xai_faithfulness || {}
  const ds        = data.dataset          || {}
  const cfg       = data.training_config  || {}
  const isDemo    = data.demo_mode
  const modelList = Object.entries(models)
  const ijepa     = modelList[0]?.[1] || {}

  return (
    <main className="pg-dash">
      <div className="container dash-wrap">

        <div className="dash-hd">
          <div>
            <h1 className="headline">Dashboard</h1>
            <p className="subtext" style={{fontSize:15,marginTop:6}}>Detailed performance metrics and XAI faithfulness for the I-JEPA model.</p>
          </div>
          {isDemo && (
            <span className="badge badge-amber">
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
              </svg>
              Representative metrics
            </span>
          )}
        </div>

        {/* KPIs */}
        <div className="kpi-row">
          {[
            {label:'AUC-ROC',  value:pct(ijepa.auc_roc),  color:'var(--primary)', note:'I-JEPA probe'},
            {label:'Accuracy', value:pct(ijepa.accuracy), color:'var(--success)', note:'Test set'},
            {label:'F1',       value:pct(ijepa.f1),       color:'var(--cyan)',    note:'Nodule class'},
            {label:'MCC',      value:fmt4(ijepa.mcc),     color:'var(--warning)', note:'Matthews Corr. Coef.'},
          ].map(k=>(
            <div key={k.label} className="card kpi-card">
              <div className="kpi-top"><span className="label">{k.label}</span><div className="kpi-dot" style={{background:k.color}}/></div>
              <div className="kpi-val" style={{color:k.color}}>{k.value}</div>
              <div className="kpi-note">{k.note}</div>
            </div>
          ))}
        </div>

        {/* Section nav */}
        <div className="sec-nav">
          {[['models','Model'],['xai','XAI Faithfulness'],['dataset','Dataset'],['training','Training Config']].map(([s,l])=>(
            <button key={s} className={`sec-btn${sec===s?' active':''}`} onClick={()=>setSec(s)}>{l}</button>
          ))}
        </div>

        {/* ── Models ── */}
        {sec==='models' && (
          <div className="dash-section">
            <div className="section-intro">
              <h2 className="section-title">Model Performance</h2>
              <p className="section-desc">Click the model card to expand per-class metrics, confusion matrix and training details.</p>
            </div>
            <div className="card" style={{padding:'1.25rem',marginBottom:'1.25rem'}}>
              <p className="label" style={{marginBottom:'1rem'}}>Performance summary</p>
              <div className="tbl-wrap">
                <table className="dtable">
                  <thead>
                    <tr><th>Model</th><th>Accuracy</th><th>AUC-ROC</th><th>Precision</th><th>Recall</th><th>F1</th><th>Specificity</th><th>MCC</th></tr>
                  </thead>
                  <tbody>
                    {modelList.map(([name,m],i)=>{
                      const top=i===0
                      return (
                        <tr key={name} className={top?'tr-top':''}>
                          <td><div style={{display:'flex',alignItems:'center',gap:6}}>
                            <span style={{fontWeight:top?700:400,fontSize:12}}>{name}</span>
                          </div></td>
                          {['accuracy','auc_roc','precision','recall','f1','specificity'].map(k=>(
                            <td key={k} className={top?'td-best':'td-norm'}>{pct(m[k])}</td>
                          ))}
                          <td className={top?'td-best':'td-norm'}>{fmt4(m.mcc)}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
            <div style={{display:'flex',flexDirection:'column',gap:'1rem'}}>
              {modelList.map(([name,m],i)=><ModelCard key={name} name={name} m={m} isTop={i===0}/>)}
            </div>
          </div>
        )}

        {/* ── XAI ── */}
        {sec==='xai' && (
          <div className="dash-section">
            <div className="section-intro">
              <h2 className="section-title">XAI Faithfulness — Deletion Curve</h2>
              <p className="section-desc">
                Lower Deletion AUC = more faithful (removing the most important regions
                degrades the prediction faster).
              </p>
            </div>
            <div className="faith-cards">
              {Object.entries(faith).map(([method,v])=>(
                <div key={method} className="card faith-big-card">
                  <h3 style={{fontSize:15,fontWeight:700,marginBottom:'.9rem'}}>{method}</h3>
                  <div style={{display:'flex',flexDirection:'column',gap:14}}>
                    <div>
                      <div style={{display:'flex',justifyContent:'space-between',marginBottom:5}}>
                        <span style={{fontSize:13,fontWeight:600,color:'var(--danger)'}}>Deletion AUC ↓ <span style={{fontSize:11,fontWeight:400,color:'var(--g400)'}}>lower is better</span></span>
                        <span style={{fontFamily:'var(--mono)',fontWeight:700,color:'var(--danger)'}}>{v.deletion_auc.toFixed(4)} ± {v.deletion_std?.toFixed(4)||'—'}</span>
                      </div>
                      <div className="faith-track"><div className="faith-fill fill-red" style={{width:`${v.deletion_auc*100}%`}}/></div>
                    </div>
                    {v.method_detail && <p style={{fontSize:12,color:'var(--g400)',borderTop:'1px solid var(--g100)',paddingTop:10}}>{v.method_detail}</p>}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── Dataset ── */}
        {sec==='dataset' && (
          <div className="dash-section">
            <div className="section-intro">
              <h2 className="section-title">Dataset — {ds.name}</h2>
              <p className="section-desc">{ds.task}</p>
            </div>
            <div className="dash-2col">
              <div className="card" style={{padding:'1.25rem'}}>
                <p className="label" style={{marginBottom:'1rem'}}>Train / Val / Test split</p>
                {[{label:'Train',val:ds.train||14302,pct:70,color:'var(--primary)'},{label:'Val',val:ds.val||3065,pct:15,color:'var(--cyan)'},{label:'Test',val:ds.test||3065,pct:15,color:'var(--success)'}].map(s=>(
                  <div key={s.label} style={{marginBottom:14}}>
                    <div style={{display:'flex',justifyContent:'space-between',fontSize:13,marginBottom:5}}>
                      <span style={{fontWeight:600,color:'var(--g700)'}}>{s.label}</span>
                      <span style={{fontFamily:'var(--mono)',fontWeight:700,color:s.color}}>{s.val.toLocaleString()} &nbsp;·&nbsp; {s.pct}%</span>
                    </div>
                    <div style={{height:9,background:'var(--g200)',borderRadius:5,overflow:'hidden'}}>
                      <div style={{width:`${s.pct}%`,height:'100%',background:s.color,borderRadius:5,transition:'width .8s ease'}}/>
                    </div>
                  </div>
                ))}
                <div style={{marginTop:'1rem',paddingTop:'1rem',borderTop:'1px solid var(--g200)',display:'flex',justifyContent:'space-between',alignItems:'center'}}>
                  <span style={{fontSize:13,color:'var(--g500)'}}>Total</span>
                  <b style={{fontFamily:'var(--mono)',fontSize:22,color:'var(--dark)'}}>{(ds.total_images||20432).toLocaleString()}</b>
                </div>
              </div>
              <div className="card" style={{padding:'1.25rem'}}>
                <p className="label" style={{marginBottom:'1rem'}}>Class distribution — train</p>
                {ds.class_train && Object.entries(ds.class_train).map(([cls,val])=>{
                  const total=Object.values(ds.class_train).reduce((a,b)=>a+b,0)
                  const color=cls==='nodule'?'var(--danger)':'var(--success)'
                  return (
                    <div key={cls} style={{marginBottom:12}}>
                      <div style={{display:'flex',justifyContent:'space-between',fontSize:13,marginBottom:4}}>
                        <span style={{fontWeight:500,textTransform:'capitalize'}}>{cls}</span>
                        <span style={{fontFamily:'var(--mono)',fontWeight:700,color}}>{val.toLocaleString()} ({((val/total)*100).toFixed(0)}%)</span>
                      </div>
                      <div style={{height:7,background:'var(--g200)',borderRadius:3,overflow:'hidden'}}>
                        <div style={{width:`${(val/total)*100}%`,height:'100%',background:color,borderRadius:3}}/>
                      </div>
                    </div>
                  )
                })}
                <div style={{marginTop:'1rem'}}>
                  <p className="label" style={{marginBottom:'1rem'}}>Technical details</p>
                  {[['Image size',ds.image_size],['Normalization',ds.normalization],['Augmentation',ds.augmentation]].map(([k,v])=>v&&(
                    <div key={k} style={{padding:'7px 0',borderBottom:'1px solid var(--g100)',fontSize:12}}>
                      <span style={{color:'var(--g400)',display:'block',marginBottom:2}}>{k}</span>
                      <span style={{color:'var(--g700)',fontWeight:500}}>{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── Training ── */}
        {sec==='training' && (
          <div className="dash-section">
            <div className="section-intro">
              <h2 className="section-title">Training Configuration</h2>
              <p className="section-desc">Hyperparameters for I-JEPA pre-training and linear probe.</p>
            </div>
            <div className="dash-2col">
              {cfg.ijepa_pretrain && (
                <div className="card" style={{padding:'1.25rem'}}>
                  <p className="label" style={{marginBottom:'1rem',color:'var(--primary)'}}>I-JEPA Pre-training</p>
                  {[['Epochs',cfg.ijepa_pretrain.epochs],['Batch size',cfg.ijepa_pretrain.batch_size],['Optimizer',cfg.ijepa_pretrain.optimizer],['Learning rate',cfg.ijepa_pretrain.lr],['Weight decay',cfg.ijepa_pretrain.weight_decay],['Warmup epochs',cfg.ijepa_pretrain.warmup_epochs],['Scheduler',cfg.ijepa_pretrain.scheduler],['EMA momentum',cfg.ijepa_pretrain.ema_momentum],['Target mask ratio',cfg.ijepa_pretrain.target_mask_ratio?`${(cfg.ijepa_pretrain.target_mask_ratio*100).toFixed(0)}%`:null],['Final pretrain loss',cfg.ijepa_pretrain.final_pretrain_loss]].filter(([,v])=>v!=null).map(([k,v])=>(
                    <div key={k} className="cfg-row"><span className="cfg-key">{k}</span><span className="cfg-val">{String(v)}</span></div>
                  ))}
                </div>
              )}
              {cfg.linear_probe && (
                <div className="card" style={{padding:'1.25rem'}}>
                  <p className="label" style={{marginBottom:'1rem',color:'var(--success)'}}>Linear Probe</p>
                  {[['Epochs',cfg.linear_probe.epochs],['Batch size',cfg.linear_probe.batch_size],['Optimizer',cfg.linear_probe.optimizer],['Learning rate',cfg.linear_probe.lr],['Weight decay',cfg.linear_probe.weight_decay],['Dropout',cfg.linear_probe.dropout],['Encoder',cfg.linear_probe.encoder],['Scheduler',cfg.linear_probe.scheduler]].filter(([,v])=>v!=null).map(([k,v])=>(
                    <div key={k} className="cfg-row"><span className="cfg-key">{k}</span><span className="cfg-val">{String(v)}</span></div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {isDemo && (
          <div className="demo-note">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
              <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>
            Representative metrics. Start the backend with your trained checkpoints to display live results.
          </div>
        )}
      </div>
    </main>
  )
}
