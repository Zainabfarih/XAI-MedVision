import { useState, useRef, useCallback } from 'react'
import { api } from '../api/client'
import './Analyze.css'

const TABS = ['Results','Heatmap','Boundaries','Attribution','Compare']
const XAI_KEYS = { Heatmap:'gradcam', Boundaries:'lime', Attribution:'shap' }
const XAI_COLORS = { Heatmap:'#f59e0b', Boundaries:'#10b981', Attribution:'#3b82f6' }
const XAI_DESCS = {
  Heatmap:     'Gradient-weighted activation map — warm colors show regions that most influenced the decision.',
  Boundaries:  'Superpixel regions — green areas contributed positively, red areas negatively to the prediction.',
  Attribution: 'Shapley pixel attributions — blue pixels support the result, red pixels oppose it.',
}

export default function Analyze() {
  const [file,    setFile]    = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result,  setResult]  = useState(null)
  const [xai,     setXai]     = useState({})
  const [xLoad,   setXLoad]   = useState({})
  const [tab,     setTab]     = useState('Results')
  const [drag,    setDrag]    = useState(false)
  const [error,   setError]   = useState(null)
  const inputRef = useRef()

  const pickFile = useCallback(f => {
    if (!f) return
    const ok = f.type.startsWith('image/')
    if (!ok) { setError('Please upload an image file (PNG, JPG, TIFF).'); return }
    setFile(f); setPreview(URL.createObjectURL(f))
    setResult(null); setXai({}); setError(null); setTab('Results')
  }, [])

  const onDrop = e => {
    e.preventDefault(); setDrag(false); pickFile(e.dataTransfer.files[0])
  }

  const runAnalysis = async () => {
    if (!file) return
    setLoading(true); setError(null)
    try {
      const data = await api.predict(file)
      setResult(data); setTab('Results')
    } catch {
      setError('Analysis failed. Check the console for details.')
    } finally { setLoading(false) }
  }

  const runXai = async method => {
    const key = XAI_KEYS[method]
    if (!key || xai[key]) return
    setXLoad(p => ({ ...p, [key]: true }))
    try {
      const du = result?.original_image || preview
      let data
      if (key === 'gradcam')   data = await api.gradcam(file, du)
      else if (key === 'lime') data = await api.lime(file, 500, du)
      else                     data = await api.shap(file, du)
      setXai(p => ({ ...p, [key]: data }))
    } finally { setXLoad(p => ({ ...p, [key]: false })) }
  }

  const onTab = t => {
    setTab(t)
    if (result) {
      if (XAI_KEYS[t]) runXai(t)
      if (t === 'Compare') Object.keys(XAI_KEYS).forEach(m => runXai(m))
    }
  }

  const isNodule = result?.label === 1
  const prob = result?.probabilities

  return (
    <main className="pg-analyze">
      <div className="container" style={{paddingTop:'calc(var(--nav-h) + 2.5rem)',paddingBottom:'4rem'}}>

        <div className="az-header">
          <h1 className="headline">Scan Analysis</h1>
          <p className="subtext" style={{fontSize:15,marginTop:6}}>
            Upload a pulmonary CT scan to receive an AI assessment with visual explanations.
          </p>
        </div>

        <div className="az-layout">
          {/* ── Sidebar ── */}
          <aside className="az-side">

            {/* Upload */}
            <div className="card" style={{padding:'1.1rem'}}>
              <p className="label" style={{marginBottom:'1rem'}}>CT Scan Image</p>
              <div
                className={`upload-box${drag?' drag':''}${preview?' filled':''}`}
                onDragOver={e=>{e.preventDefault();setDrag(true)}}
                onDragLeave={()=>setDrag(false)}
                onDrop={onDrop}
                onClick={()=>!preview && inputRef.current?.click()}
              >
                <input ref={inputRef} type="file" accept="image/*"
                  style={{display:'none'}} onChange={e=>pickFile(e.target.files[0])}/>
                {preview ? (
                  <>
                    <img src={preview} alt="scan" className="upload-img"/>
                    <div className="upload-hover">
                      <button className="btn btn-sm" style={{background:'#fff',color:'var(--g800)'}}
                        onClick={e=>{e.stopPropagation();inputRef.current?.click()}}>
                        Change
                      </button>
                    </div>
                  </>
                ) : (
                  <div className="upload-empty">
                    <div className="upload-icon-box">
                      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                        <polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
                      </svg>
                    </div>
                    <p style={{fontWeight:600,fontSize:14,marginTop:10}}>Drop scan here</p>
                    <p style={{fontSize:12,color:'var(--g400)',marginTop:4}}>
                      or <span style={{color:'var(--primary)',fontWeight:600}}>browse files</span>
                    </p>
                    <p style={{fontSize:11,color:'var(--g300)',marginTop:10}}>PNG · JPG · TIFF</p>
                  </div>
                )}
              </div>

              {file && (
                <div className="file-row">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                    <polyline points="14 2 14 8 20 8"/>
                  </svg>
                  <span className="file-name">{file.name}</span>
                  <span style={{fontSize:11,color:'var(--g400)',flexShrink:0}}>{(file.size/1024).toFixed(0)} KB</span>
                  <button className="file-del"
                    onClick={()=>{setFile(null);setPreview(null);setResult(null);setXai({})}}>
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                  </button>
                </div>
              )}

              {error && <div className="az-error">{error}</div>}

              <button className="btn btn-primary" disabled={!file||loading}
                style={{width:'100%',justifyContent:'center',marginTop:12}}
                onClick={runAnalysis}>
                {loading
                  ? <><div className="spinner"/><span>Analyzing…</span></>
                  : <><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg><span>Analyze Scan</span></>
                }
              </button>
            </div>

            {/* Result summary */}
            {result && (
              <div className={`card result-sum${isNodule?' sum-red':' sum-green'}`}>
                <div style={{display:'flex',alignItems:'center',gap:8,flexWrap:'wrap'}}>
                  <span className={`badge ${isNodule?'badge-red':'badge-green'}`} style={{fontSize:13,padding:'5px 13px'}}>
                    {isNodule ? '⚠ Nodule Detected' : '✓ No Nodule Found'}
                  </span>
                  {result.demo_mode && <span className="badge badge-amber" style={{fontSize:11}}>Demo</span>}
                </div>

                <div style={{marginTop:14,display:'flex',flexDirection:'column',gap:9}}>
                  {[
                    {label:'Nodule',     val:prob.nodule,    cls:'fill-red'},
                    {label:'No Nodule',  val:prob.no_nodule, cls:'fill-green'},
                  ].map(({label,val,cls})=>(
                    <div key={label}>
                      <div style={{display:'flex',justifyContent:'space-between',fontSize:12,color:'var(--g500)',marginBottom:4}}>
                        <span>{label}</span>
                        <b style={{fontFamily:'var(--mono)'}}>{(val*100).toFixed(1)}%</b>
                      </div>
                      <div className="prob-track">
                        <div className={`prob-fill ${cls}`} style={{width:`${val*100}%`}}/>
                      </div>
                    </div>
                  ))}
                </div>

                <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginTop:14,paddingTop:12,borderTop:'1px solid var(--border)'}}>
                  <span style={{fontSize:13,color:'var(--g500)'}}>Confidence</span>
                  <b style={{fontFamily:'var(--mono)',fontSize:22,color:isNodule?'var(--danger)':'var(--success)'}}>
                    {(result.confidence*100).toFixed(1)}%
                  </b>
                </div>
                <div style={{fontSize:11,color:'var(--g400)',marginTop:8,display:'flex',justifyContent:'space-between'}}>
                  <span>I-JEPA + Linear Probe</span>
                  <span>{result.inference_time_ms} ms</span>
                </div>
              </div>
            )}
          </aside>

          {/* ── Main panel ── */}
          <section className="az-main">

            {/* Tabs */}
            {result && (
              <div className="az-tabs">
                {TABS.map(t=>(
                  <button key={t} className={`az-tab${tab===t?' active':''}`} onClick={()=>onTab(t)}>
                    {t}
                    {XAI_KEYS[t] && xai[XAI_KEYS[t]] && <span className="tab-done"/>}
                  </button>
                ))}
              </div>
            )}

            {/* Empty */}
            {!result && !loading && (
              <div className="az-empty">
                <div className="empty-icon">
                  <svg width="38" height="38" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="3" y="3" width="18" height="18" rx="3"/>
                    <circle cx="12" cy="10" r="3"/>
                    <path d="M7 20a5 5 0 0 1 10 0"/>
                  </svg>
                </div>
                <h3>No scan loaded</h3>
                <p>Upload a CT image and click <b>Analyze Scan</b> to get started.</p>
              </div>
            )}

            {/* Loading */}
            {loading && (
              <div className="az-empty">
                <div className="spinner spinner-blue" style={{width:36,height:36,borderWidth:3}}/>
                <h3 style={{marginTop:14}}>Processing scan…</h3>
                <p>The AI model is analyzing your image.</p>
              </div>
            )}

            {/* Results tab */}
            {result && tab==='Results' && (
              <div className="tab-body">
                <div className="res-grid">
                  <div>
                    <p className="label" style={{marginBottom:10}}>Uploaded scan</p>
                    <img src={result.original_image||preview} alt="CT scan" className="scan-disp"/>
                  </div>
                  <div>
                    <p className="label" style={{marginBottom:10}}>Assessment summary</p>
                    <div className="detail-list">
                      {[
                        ['Result',       isNodule?'⚠ Nodule Detected':'✓ No Nodule Found'],
                        ['Confidence',   `${(result.confidence*100).toFixed(1)}%`],
                        ['P(Nodule)',    `${(prob.nodule*100).toFixed(2)}%`],
                        ['P(No Nodule)',`${(prob.no_nodule*100).toFixed(2)}%`],
                        ['Processing',  `${result.inference_time_ms} ms`],
                        ['Mode',        result.demo_mode?'Demo (simulated)':'Live model'],
                      ].map(([k,v])=>(
                        <div key={k} className="detail-row">
                          <span className="dk">{k}</span>
                          <span className="dv">{v}</span>
                        </div>
                      ))}
                    </div>
                    <div className="notice">
                      This analysis is a research aid only and does not constitute medical advice. Consult a qualified radiologist for clinical decisions.
                    </div>
                    <button className="btn btn-outline" style={{width:'100%',justifyContent:'center',marginTop:12}}
                      onClick={()=>onTab('Compare')}>
                      View all explanations →
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* XAI individual tab */}
            {result && XAI_KEYS[tab] && (
              <div className="tab-body">
                <div className="xai-info-bar" style={{borderLeft:`3px solid ${XAI_COLORS[tab]}`}}>
                  <p style={{fontSize:13,color:'var(--g600)',lineHeight:1.65}}>{XAI_DESCS[tab]}</p>
                </div>
                <div className="xai-duo">
                  <div>
                    <p className="label" style={{marginBottom:10}}>Original scan</p>
                    <div className="xai-img-box dark">
                      <img src={result.original_image||preview} alt="original"/>
                    </div>
                  </div>
                  <div>
                    <p className="label" style={{marginBottom:10}}>{tab} view</p>
                    <div className="xai-img-box dark">
                      {xLoad[XAI_KEYS[tab]] ? (
                        <div className="xai-center">
                          <div className="spinner" style={{borderTopColor:XAI_COLORS[tab],borderColor:XAI_COLORS[tab]+'33'}}/>
                          <span style={{fontSize:12,color:'var(--g400)',marginTop:10}}>Computing…</span>
                        </div>
                      ) : xai[XAI_KEYS[tab]] ? (
                        <img
                          src={xai[XAI_KEYS[tab]].image}
                          alt={tab}
                          style={xai[XAI_KEYS[tab]].demo_mode
                            ? {filter: xai[XAI_KEYS[tab]].demoFilter, mixBlendMode:'screen'}
                            : {}}
                        />
                      ) : (
                        <div className="xai-center">
                          <span style={{fontSize:12,color:'var(--g400)'}}>Click the tab to generate</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Compare tab */}
            {result && tab==='Compare' && (
              <div className="tab-body">
                <div className="cmp-grid">
                  {Object.entries(XAI_KEYS).map(([method, key])=>(
                    <div key={key} className="cmp-card" style={{borderTop:`3px solid ${XAI_COLORS[method]}`}}>
                      <div className="cmp-head">
                        <b style={{fontSize:13,color:'var(--g700)'}}>{method}</b>
                      </div>
                      <div className="xai-img-box dark" style={{borderRadius:0,aspectRatio:'1',border:'none',borderBottom:'1px solid var(--border)'}}>
                        {xLoad[key] ? (
                          <div className="xai-center">
                            <div className="spinner spinner-blue"/>
                          </div>
                        ) : xai[key] ? (
                          <img
                            src={xai[key].image}
                            alt={method}
                            style={xai[key].demo_mode
                              ? {filter:xai[key].demoFilter, mixBlendMode:'screen'}
                              : {}}
                          />
                        ) : (
                          <div className="xai-center">
                            <button className="btn btn-sm btn-outline" onClick={()=>runXai(method)}>Generate</button>
                          </div>
                        )}
                      </div>
                      <p className="cmp-desc">{XAI_DESCS[method]}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>
        </div>
      </div>
    </main>
  )
}
