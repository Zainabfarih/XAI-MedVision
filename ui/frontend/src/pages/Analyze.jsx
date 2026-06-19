import { useState, useRef, useCallback } from 'react'
import { api, isVolumeFile, dataUrlToFile } from '../api/client'
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
  const [isVol,   setIsVol]   = useState(false)
  const [xaiFile, setXaiFile] = useState(null)
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
    const vol = isVolumeFile(f)
    const ok = vol || f.type.startsWith('image/')
    if (!ok) { setError('Upload a 2D image (PNG, JPG, TIFF) or a 3D volume (NIfTI, DICOM, NPY).'); return }
    setFile(f); setIsVol(vol)
    setPreview(vol ? null : URL.createObjectURL(f))
    setResult(null); setXai({}); setXaiFile(null); setError(null); setTab('Results')
  }, [])

  const onDrop = e => {
    e.preventDefault(); setDrag(false); pickFile(e.dataTransfer.files[0])
  }

  const runAnalysis = async () => {
    if (!file) return
    setLoading(true); setError(null)
    try {
      if (isVol) {
        const v = await api.predictVolume(file)
        const top = v.slice_results.reduce((a, b) => (b.nodule > a.nodule ? b : a), v.slice_results[0])
        const pred = v.max_slice?.prediction
        if (pred) {
          setXaiFile(dataUrlToFile(pred.original_image, 'slice.png'))
          setResult({ ...pred, is_volume: true, volume: v })
          setXai(v.max_slice.gradcam
            ? { gradcam: { method: 'gradcam', image: v.max_slice.gradcam,
                original: pred.original_image, demo_mode: false,
                description: 'Gradient-weighted activation map for the most suspicious slice.' } }
            : {})
        } else {
          setXaiFile(null); setXai({})
          setResult({
            is_volume: true, demo_mode: true, volume: v,
            label: v.volume_label, label_name: v.volume_label_name,
            probabilities: { nodule: top.nodule, no_nodule: +(1 - top.nodule).toFixed(4) },
            confidence: +Math.max(top.nodule, 1 - top.nodule).toFixed(4),
            inference_time_ms: v.inference_time_ms, original_image: null,
          })
        }
        setTab('Volume')
      } else {
        const data = await api.predict(file)
        setXaiFile(file)
        setResult(data); setTab('Results')
      }
    } catch {
      setError('Analysis failed. Check the console for details.')
    } finally { setLoading(false) }
  }

  const runXai = async method => {
    const key = XAI_KEYS[method]
    if (!key || xai[key]) return
    const src = xaiFile || file
    if (!src) return
    setXLoad(p => ({ ...p, [key]: true }))
    try {
      const du = result?.original_image || preview
      let data
      if (key === 'gradcam')   data = await api.gradcam(src, du)
      else if (key === 'lime') data = await api.lime(src, 500, du)
      else                     data = await api.shap(src, du)
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
                <input ref={inputRef} type="file" accept="image/*,.nii,.nii.gz,.dcm,.dicom,.npy,.npz,.tif,.tiff"
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
                ) : (file && isVol) ? (
                  <div className="upload-empty">
                    <div className="upload-icon-box">
                      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polygon points="12 2 2 7 12 12 22 7 12 2"/>
                        <polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>
                      </svg>
                    </div>
                    <p style={{fontWeight:600,fontSize:14,marginTop:10}}>3D volume ready</p>
                    <p style={{fontSize:12,color:'var(--g400)',marginTop:4}}>
                      Sliced into 2D views on analysis
                    </p>
                    <button className="btn btn-sm" style={{marginTop:10,background:'var(--g100)',color:'var(--g700)'}}
                      onClick={e=>{e.stopPropagation();inputRef.current?.click()}}>
                      Change
                    </button>
                  </div>
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
                    <p style={{fontSize:11,color:'var(--g300)',marginTop:10}}>2D: PNG · JPG · TIFF &nbsp;|&nbsp; 3D: NIfTI · DICOM · NPY</p>
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
                    onClick={()=>{setFile(null);setPreview(null);setIsVol(false);setXaiFile(null);setResult(null);setXai({})}}>
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
                  : <><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg><span>{isVol?'Analyze Volume':'Analyze Scan'}</span></>
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
                  <span>{result.is_volume ? '3D volume · per-slice' : 'I-JEPA + Linear Probe'}</span>
                  <span>{result.inference_time_ms} ms</span>
                </div>
                {result.is_volume && (
                  <div style={{fontSize:11,color:'var(--g400)',marginTop:4,display:'flex',justifyContent:'space-between'}}>
                    <span>{result.volume.num_slices_processed} slices analyzed</span>
                    <span>{result.volume.num_positive_slices} positive</span>
                  </div>
                )}
              </div>
            )}
          </aside>

          {/* ── Main panel ── */}
          <section className="az-main">

            {/* Tabs */}
            {result && (
              <div className="az-tabs">
                {(result.is_volume
                  ? (result.original_image ? ['Volume', ...TABS.slice(1)] : ['Volume'])
                  : TABS).map(t=>(
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

            {/* Volume tab — verdict + per-slice detail + most-suspicious slice XAI */}
            {result && result.is_volume && tab==='Volume' && (() => {
              const vol = result.volume
              const slices = vol.slice_results || []
              const maxNodule = slices.reduce((m, s) => Math.max(m, s.nodule), 0)
              const topIdx = vol.max_slice?.index
              return (
              <div className="tab-body">
                <div className={`vol-verdict${isNodule?' v-red':' v-green'}`}>
                  <div className="vol-verdict-icon">{isNodule?'⚠':'✓'}</div>
                  <h2 className="vol-verdict-title">
                    {isNodule ? 'Nodule Detected' : 'No Nodule Found'}
                  </h2>
                  <p className="vol-verdict-sub">
                    {isNodule
                      ? `Nodule found in ${vol.num_positive_slices} of ${vol.num_slices_processed} analyzed slices.`
                      : `No nodule across ${vol.num_slices_processed} analyzed slices.`}
                  </p>
                  <div className="vol-verdict-meta">
                    {result.demo_mode && <span className="badge badge-amber" style={{fontSize:11}}>Demo</span>}
                  </div>
                </div>

                {/* Volume statistics */}
                <p className="label" style={{margin:'22px 0 10px'}}>Volume breakdown</p>
                <div className="detail-list">
                  {[
                    ['Total slices',        vol.num_slices_total],
                    ['Slices analyzed',     vol.num_slices_processed],
                    ['Positive slices',     `${vol.num_positive_slices} (${((vol.num_positive_slices/vol.num_slices_processed)*100).toFixed(1)}%)`],
                    ['Most suspicious slice', topIdx != null ? `#${topIdx}` : '—'],
                    ['Peak nodule score',   `${(maxNodule*100).toFixed(2)}%`],
                    ['Processing',          `${vol.inference_time_ms} ms`],
                    ['Mode',                result.demo_mode ? 'Demo (simulated)' : 'Live model'],
                  ].map(([k,v])=>(
                    <div key={k} className="detail-row">
                      <span className="dk">{k}</span>
                      <span className="dv">{v}</span>
                    </div>
                  ))}
                </div>

                {/* Per-slice score chart */}
                <p className="label" style={{margin:'22px 0 10px'}}>Per-slice nodule score</p>
                <div className="slice-chart">
                  {slices.map(s=>(
                    <div
                      key={s.index}
                      className={`slice-bar${s.label===1?' pos':''}${s.index===topIdx?' top':''}`}
                      style={{height:`${Math.max(2, s.nodule*100)}%`}}
                      title={`Slice #${s.index} — ${(s.nodule*100).toFixed(1)}%`}
                    />
                  ))}
                </div>
                <div className="slice-legend">
                  <span><i className="dot dot-pos"/> Above threshold</span>
                  <span><i className="dot dot-neg"/> Below threshold</span>
                  <span><i className="dot dot-top"/> Most suspicious</span>
                </div>

                {/* Most-suspicious-slice explainability */}
                {result.original_image && (
                  <>
                    <p className="label" style={{margin:'22px 0 10px'}}>
                      Why this decision? — Grad-CAM on slice #{topIdx}
                    </p>
                    <div className="xai-duo">
                      <div>
                        <p style={{fontSize:12,color:'var(--g400)',marginBottom:8}}>Suspicious slice</p>
                        <div className="xai-img-box dark"><img src={result.original_image} alt="slice"/></div>
                      </div>
                      <div>
                        <p style={{fontSize:12,color:'var(--g400)',marginBottom:8}}>Heatmap</p>
                        <div className="xai-img-box dark">
                          {xai.gradcam
                            ? <img src={xai.gradcam.image} alt="gradcam"/>
                            : <div className="xai-center"><span style={{fontSize:12,color:'var(--g400)'}}>Computing…</span></div>}
                        </div>
                      </div>
                    </div>
                    <button className="btn btn-outline" style={{width:'100%',justifyContent:'center',marginTop:12}}
                      onClick={()=>onTab('Compare')}>
                      View all explanations (Heatmap · Boundaries · Attribution) →
                    </button>
                  </>
                )}

                <div className="notice" style={{marginTop:16}}>
                  The 2D model scores every slice of the volume independently. The volume is reported as positive if any single slice crosses the detection threshold. The explainability views above apply to the most suspicious slice. This analysis is a research aid only and does not constitute medical advice.
                </div>
              </div>
            )})()}

            {/* Results tab */}
            {result && tab==='Results' && (
              <div className="tab-body">
                <div className="res-grid">
                  <div>
                    <p className="label" style={{marginBottom:10}}>Uploaded scan</p>
                    {(result.original_image||preview)
                      ? <img src={result.original_image||preview} alt="CT scan" className="scan-disp"/>
                      : <div className="xai-img-box dark" style={{aspectRatio:'1'}}><div className="xai-center"><span style={{fontSize:12,color:'var(--g400)'}}>3D volume — no 2D preview</span></div></div>}
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
