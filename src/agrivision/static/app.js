const isSimulation=document.body.dataset.simulation==='true';
function healthClass(status){return (status||'neutral').toLowerCase().replaceAll(' ','-')}
function renderPredictions(rows){
  const box=document.getElementById('predictions');
  if(!rows||!rows.length){box.innerHTML='<span class="muted">No scan yet.</span>';return}
  box.innerHTML=rows.map(p=>`<div class="pred-row"><b>${isSimulation?'SIMULATION: ':''}${escapeHtml(p.label)}</b><b>${isSimulation?'SIM ':''}${Number(p.confidence).toFixed(1)}%</b></div>`).join('');
}
function escapeHtml(text){const d=document.createElement('div');d.textContent=text??'';return d.innerHTML}
async function refresh(){
  try{
    const r=await fetch('/api/status',{cache:'no-store'}); const s=await r.json();
    const plant=document.getElementById('plant'); plant.textContent=s.plant;
    plant.className='plant-status '+healthClass(s.plant);
    const prefix=isSimulation?'SIM ':'';
    rawLabel.textContent=s.raw_label==='-'?'-':prefix+s.raw_label; confidence.textContent=prefix+Number(s.confidence).toFixed(1)+'%';
    soil.textContent=prefix+Number(s.soil).toFixed(1)+'%'; temp.textContent=prefix+Number(s.temperature).toFixed(1)+' °C';
    hum.textContent=prefix+Number(s.humidity).toFixed(1)+'%'; pump.textContent=prefix+s.pump; last.textContent=s.last_scan;
    scanMessage.textContent=s.message; renderPredictions(s.top_predictions);
  }catch(e){scanMessage.textContent='Dashboard connection error: '+e}
}
async function scanLeaf(){
  scanBtn.disabled=true; scanMessage.textContent=isSimulation?'Capturing and analysing with simulation only...':'Capturing and analysing on Edge TPU...';
  try{
    const r=await fetch('/api/scan',{method:'POST'}); const payload=await r.json();
    if(!r.ok) throw new Error(payload.error||'scan failed');
    leaf.style.opacity=1; leaf.src='/latest.jpg?t='+Date.now(); await refresh();
  }catch(e){scanMessage.textContent='Scan failed: '+e.message}
  finally{scanBtn.disabled=false}
}
setInterval(refresh,1500); refresh();
