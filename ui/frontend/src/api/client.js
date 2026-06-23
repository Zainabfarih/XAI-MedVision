let _alive = null

async function ping() {
  if (_alive !== null) return _alive
  try { const r = await fetch('/api/status', { signal: AbortSignal.timeout(1800) }); _alive = r.ok }
  catch { _alive = false }
  return _alive
}

export const DEMO_METRICS = {
  demo_mode: true,
  models: {
    "I-JEPA + Linear Probe": {
      training_strategy: "Self-supervised (I-JEPA) + frozen encoder + linear head",
      architecture:      "ViT-Small/16  —  384-dim embeddings  —  196 patches",
      accuracy: 0.8914, auc_roc: 0.9582, precision: 0.9195, recall: 0.8560, f1: 0.8866,
      precision_no_nodule: 0.8672, recall_no_nodule: 0.9262, f1_no_nodule: 0.8957,
      specificity: 0.9262, mcc: 0.7844,
      faithfulness_deletion_auc: 0.2249,
      confusion_matrix: { TP:1302, TN:1430, FP:114, FN:219 },
      val_auc_best_epoch: 0.9619, training_epochs: 100, probe_epochs: 30,
      pretrain_loss_final: 0.0212,
    },
  },
  xai_faithfulness: {
    "Heatmap (Grad-CAM)": { deletion_auc:0.2249, deletion_std:0.1142 },
    "Boundaries (LIME)":  { deletion_auc:0.2253, deletion_std:0.1621 },
    "Attribution (SHAP)": { deletion_auc:0.2105, deletion_std:0.1605 },
  },
  dataset: {
    name:"Lung CT Classification", source:"Kaggle — Healthy / Lung_Nodule",
    task:"Binary classification: Nodule vs Healthy",
    total_images:20432, train:14302, val:3065, test:3065,
    class_train:{ nodule:7098, healthy:7204 },
    class_test:{ nodule:1521, healthy:1544 },
    image_size:"224 × 224 px",
    normalization:"Per-dataset mean/std (train split)",
    augmentation:"HorizontalFlip, Rotate(15°), BrightnessContrast, GaussNoise, CLAHE",
  },
  training_config: {
    ijepa_pretrain: {
      epochs:100, batch_size:32, optimizer:"AdamW", lr:0.001,
      weight_decay:0.04, warmup_epochs:10, scheduler:"Cosine annealing",
      ema_momentum:0.996, target_mask_ratio:0.75, final_pretrain_loss:0.0212,
    },
    linear_probe: {
      epochs:30, batch_size:64, optimizer:"AdamW", lr:0.001,
      weight_decay:0.0001, dropout:0.1, encoder:"frozen", scheduler:"CosineAnnealing",
    },
  },
}

function seedFrom(name='') { return name.split('').reduce((a,c)=>(a*31+c.charCodeAt(0))&0xffff,0) }

function demoPrediction(fileName) {
  const s=seedFrom(fileName); const raw=((s*9301+49297)%233280)/233280
  const prob=parseFloat((0.25+raw*0.55).toFixed(4)); const label=prob>0.5?1:0
  return { label, label_name:label?'Nodule Detected':'No Nodule Found',
           probabilities:{nodule:prob,no_nodule:parseFloat((1-prob).toFixed(4))},
           confidence:parseFloat(Math.max(prob,1-prob).toFixed(4)),
           inference_time_ms:18+(s%35), demo_mode:true }
}

const FILTERS = {
  gradcam: 'sepia(.8) saturate(4) hue-rotate(330deg) brightness(1.1)',
  lime:    'sepia(.6) saturate(3) hue-rotate(80deg) brightness(1.05)',
  shap:    'sepia(.6) saturate(3) hue-rotate(200deg) brightness(1.1)',
}
const DESCS = {
  gradcam: "Gradient-weighted activation map — warm regions most influenced the decision.",
  lime:    "Superpixel regions contributing positively (green) or negatively (red) to the prediction.",
  shap:    "Shapley pixel attributions — blue pixels support the prediction, red pixels oppose it.",
}

function demoXai(method,dataUrl){ return {method,image:dataUrl,demoFilter:FILTERS[method],original:dataUrl,description:DESCS[method],demo_mode:true} }
function readAsDataUrl(file){ return new Promise((r,j)=>{const fr=new FileReader();fr.onload=()=>r(fr.result);fr.onerror=j;fr.readAsDataURL(file)}) }
async function post(url,file,extra={}){ const fd=new FormData();fd.append('file',file);for(const[k,v] of Object.entries(extra))fd.append(k,v);const r=await fetch(url,{method:'POST',body:fd});if(!r.ok)throw new Error(r.status);return r.json() }

// Convert a base64 data URL (e.g. an extracted slice) back into a File so the
// existing 2D XAI endpoints can be reused on it.
export function dataUrlToFile(dataUrl, name='slice.png') {
  const [head, b64] = dataUrl.split(',')
  const mime = (head.match(/:(.*?);/) || [])[1] || 'image/png'
  const bin = atob(b64); const arr = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i)
  return new File([arr], name, { type: mime })
}

const VOLUME_RE = /\.(nii|nii\.gz|dcm|dicom|npy|npz|tif|tiff)$/i
export function isVolumeFile(file) { return !!file && VOLUME_RE.test(file.name || '') }

function demoVolume(fileName) {
  const base = seedFrom(fileName); const n = 24 + (base % 40)
  const peak = base % n; const spread = 3 + (base % 4)
  const slice_results = Array.from({ length: n }, (_, i) => {
    const d = Math.abs(i - peak)
    const prob = parseFloat(Math.max(0.04, 0.92 * Math.exp(-(d * d) / (2 * spread * spread))
      + (((base + i) % 17) / 17 - 0.5) * 0.08).toFixed(4))
    return { index: i, nodule: prob, label: prob > 0.5 ? 1 : 0 }
  })
  const n_pos = slice_results.filter(s => s.label === 1).length
  const top = slice_results.reduce((a, b) => (b.nodule > a.nodule ? b : a))
  return {
    type: 'volume', num_slices_total: n, num_slices_processed: n,
    num_positive_slices: n_pos, volume_label: n_pos > 0 ? 1 : 0,
    volume_label_name: n_pos > 0 ? 'Nodule Detected' : 'No Nodule Found',
    slice_results,
    max_slice: { index: top.index, prediction: null, gradcam: null },
    demo_mode: true, inference_time_ms: 40 + (base % 120),
  }
}

export const api = {
  status: async()=>{const a=await ping();if(a)return fetch('/api/status').then(r=>r.json());return{status:'demo',demo_mode:true}},
  metrics: async()=>{if(await ping()){try{return await fetch('/api/metrics').then(r=>r.json())}catch{}}return DEMO_METRICS},
  predict: async(file)=>{const du=await readAsDataUrl(file);if(await ping()){try{return await post('/api/predict',file)}catch{}}return{...demoPrediction(file.name),original_image:du}},
  predictVolume: async(file)=>{if(await ping()){try{return await post('/api/predict/volume',file)}catch{}}return demoVolume(file.name)},
  gradcam: async(file,du)=>{const d=du||await readAsDataUrl(file);if(await ping()){try{return await post('/api/explain/gradcam',file)}catch{}}return demoXai('gradcam',d)},
  lime:    async(file,n=500,du)=>{const d=du||await readAsDataUrl(file);if(await ping()){try{return await post('/api/explain/lime',file,{num_samples:n})}catch{}}return demoXai('lime',d)},
  shap:    async(file,du)=>{const d=du||await readAsDataUrl(file);if(await ping()){try{return await post('/api/explain/shap',file)}catch{}}return demoXai('shap',d)},
}
