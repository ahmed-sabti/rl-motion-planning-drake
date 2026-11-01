'use strict';
const $=id=>document.getElementById(id), palette=['#287d6c','#ca8144','#7273b5','#c45c85'];
const appBase=new URL('./',location.href), storagePrefix=appBase.pathname==='/'?'':appBase.pathname;
const store={getItem:key=>localStorage.getItem(storagePrefix+key),setItem:(key,value)=>localStorage.setItem(storagePrefix+key,value),removeItem:key=>localStorage.removeItem(storagePrefix+key)};

const shortNames={rrt:'RRT',rrt_star:'RRT*',rrt_connect:'RRT-Connect',policy:'Policy'};
const goalPose=key=>cfg?.goal_poses?.[key]||cfg?.pose?.q||[0,0,0];
const goalKeyNow=()=>{const keys=goalKeys().filter(k=>k!=='manual');return keys.length?keys[cycleIndex%keys.length]:Object.keys(cfg?.goals||{})[0];};
const armName=arm=>arm.name||cfg?.modes?.[arm.mode]||arm.mode;
const armShort=arm=>shortNames[arm.mode]||armName(arm);

const SETTINGS={margin_m:.01,planning_s:3,deadline_s:20};
const outcomes={solved_and_held:'Goal reached',not_held:'Goal not reached',plan_failed:'No path found · unknown',trajectory_rejected:'Trajectory rejected',deadline_exceeded:'Too slow for the deadline',joint_limits:'Stopped at joint range',collision:'Stopped at collision',geometry_unknown:'Stopped: clearance unavailable',geometry_uncertain:'Stopped: clearance uncertain',reached_not_held:'Arrived, not held'};

function outcomeLabel(outcome,reason){reason=String(reason||'');if(reason.startsWith('rejected:kto:'))return 'Curve step failed (KTO)';return (outcome==='not_held'&&outcomes[reason])||outcomes[outcome]||outcome;}
let cfg, currentPose, q=[], arms=[], result=null, busy=false, playing=false;
let cursor=0, lastTick=0, replayFrame=null, poseRequest=0, jobId=null, cycleIndex=0, autoTimer;
let records=[], seed=0, goalQ=[], runGeneration=0;
try{records=JSON.parse(store.getItem('physicsResults')||'[]');if(!Array.isArray(records))records=[];}catch{records=[];}
let offModes=new Set(['rrt','rrt_star']);try{const v=JSON.parse(store.getItem('physicsOff'));if(Array.isArray(v))offModes=new Set(v);}catch{}
const isOff=mode=>offModes.has(mode);
const available=mode=>mode!=='policy'||(policies.length>0&&!!policyKey);
const runModes=()=>arms.filter(a=>!isOff(a.mode)&&available(a.mode)).map(a=>a.mode);
const armSet=()=>runModes().join(',');
const policyLabel=key=>{const p=policies.find(x=>x.key===key);return p?`${p.label} · ${p.folder}`:key||'none';};
let source='preset-nominal',policyKey=null,policies=[];
try{source=store.getItem('physicsSource')||source;policyKey=store.getItem('physicsPolicy');}catch{}
function saveSource(){try{store.setItem('physicsSource',source);if(policyKey)store.setItem('physicsPolicy',policyKey);}catch{}}
const SRC=()=>cfg.source+'|'+(policyKey||'none')+'|'+source+'|'+JSON.stringify(goalKeys().map(k=>cfg.goals[k]));
const totalKey=mode=>SRC()+'|'+armSet()+'|'+mode;
function saveArms(){try{store.setItem('physicsOff',JSON.stringify([...offModes]));}catch{}shapeKey='';armsKey='';render();}
function removeArm(mode){if(busy||runModes().length<=1&&runModes().includes(mode))return;offModes.add(mode);saveArms();}
function addArm(mode){if(busy||!arms.some(a=>a.mode===mode))return;offModes.delete(mode);saveArms();}
let armsKey='';
function renderArms(){const key=JSON.stringify([busy,[...offModes],policies.length,policyKey]);if(key===armsKey)return;armsKey=key;
 const active=arms.map((a,i)=>({...a,i})).filter(a=>!isOff(a.mode)),removable=runModes().length>1;
 $('armChips').innerHTML=active.map(a=>{const ok=available(a.mode),name=escapeHTML(armShort(a));return `<span class="chip${ok?'':' unavailable'}" title="${escapeHTML(armName(a))}${ok?'':': no checkpoint, train one or add a checkpoint folder'}"><i class="swatch" style="background:${palette[a.i]}"></i>${name}${ok?'':' · no checkpoint'}${removable||!ok?`<button type="button" data-remove="${escapeHTML(a.mode)}" aria-label="Remove ${name}"${busy?' disabled':''}>×</button>`:''}</span>`;}).join('');
 const rest=arms.filter(a=>isOff(a.mode));$('addArm').hidden=!rest.length;
 $('addArm').innerHTML='<option value="">+ Add arm</option>'+rest.map(a=>`<option value="${escapeHTML(a.mode)}">${escapeHTML(armShort(a))}${available(a.mode)?'':' (no checkpoint)'}</option>`).join('');$('addArm').value='';$('addArm').disabled=busy;}
let totals={};try{totals=JSON.parse(store.getItem('physicsTotals')||'{}')||{};if(typeof totals!=='object'||Array.isArray(totals))totals={};
 if(store.getItem('physicsTotalsVersion')!=='2'){
  for(const t of Object.values(totals)){if(!t||typeof t!=='object')continue;const kto=Number(t.kto)||0;t.n=(Number(t.n)||0)+kto;t.fails=t.fails||{};if(kto)t.fails['Curve step failed (KTO)']=(t.fails['Curve step failed (KTO)']||0)+kto;}
  store.setItem('physicsTotals',JSON.stringify(totals));store.setItem('physicsTotalsVersion','2');
 }
}catch{totals={};}
const fmt=(x,n=2)=>typeof x==='number'&&Number.isFinite(x)?x.toFixed(n).replace('.',','):'—';
const escapeHTML=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function status(text,isError=false){const e=$('errorNotice');if(!e)return;e.hidden=!isError;if(isError)e.textContent=String(text||'Error');const l=$('statusLine');if(l)l.textContent=isError?'':String(text||'');}
async function api(path,data){
 const url=new URL(path.replace(/^\//,''),appBase);
 const response=await fetch(url,data?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}:{});
 let value=null;try{value=await response.json();}catch{}
 if(!response.ok)throw Error(value?.error||`The server answered ${response.status}.`);if(value===null)throw Error('The server sent no readable answer.');return value;}
function confirmAction(text){const dialog=$('confirmDialog');if(dialog.open)return Promise.resolve(false);$('confirmText').textContent=text;dialog.returnValue='cancel';return new Promise(resolve=>{dialog.addEventListener('close',()=>resolve(dialog.returnValue==='confirm'),{once:true});dialog.showModal();});}
function goalKeys(){return cfg.scene_goals?.[sceneName()]||Object.keys(cfg.goals);}
function cycleLength(){return goalKeys().length;}
function sceneName(){return source==='preset-free'?'free':'nominal';}
function syncGoal(){setGoalQ(goalPose(goalKeyNow()));}
function setGoalQ(values){goalQ=values.slice();}
function payload(){return basePayload();}
function basePayload(){return{q:q.slice(),goal_q:goalQ.slice(),goal:cfg?.goals?.[goalKeyNow()]||cfg?.goals?.[Object.keys(cfg?.goals||{upper:[.6,.6]})[0]],scene:sceneName(),self_collision:true,seed:seed,deadline_s:SETTINGS.deadline_s,physics_dt_s:cfg?.physics_dt_s??.002,margin_m:cfg?.margin_m??SETTINGS.margin_m,planning_s:SETTINGS.planning_s};}
function advanceSeed(){seed=(seed+1)%2147483648;}

function maxTime(){return Math.max(0,...(result?.lanes||[]).filter(l=>l.state==='done').map(l=>l.metrics.physical_s));}
function viewFrame(lane){if(lane.state!=='done')return null;const physical=Math.max(0,cursor);let low=0,high=lane.frames.length-1;while(low<high){const mid=Math.ceil((low+high)/2);if(lane.frames[mid].t<=physical+1e-9)low=mid;else high=mid-1;}return{frame:lane.frames[low],index:low,physical:Math.min(physical,lane.metrics.physical_s)};}
function resetReplay(){playing=false;cancelAnimationFrame(replayFrame);clearTimeout(autoTimer);result=null;cursor=0;jobId=null;$('timeline').value=0;render();}
let lastBusy=null;
function controls(){

 if(lastBusy!==busy){lastBusy=busy;
  document.querySelectorAll('.bar input,.bar select,.bar button').forEach(e=>{if(e.id!=='autoNext'&&e.id!=='stop')e.disabled=busy;});
 }
 $('trainStart').disabled=trainSubmitting||busy||runElsewhere||trainElsewhere||trainState==='running'||trainState==='stopping'||!trainBank;$('drawNext').disabled=busy||loading;
 const training=trainState==='running'||trainState==='stopping'||trainElsewhere;
 $('run').disabled=busy||loading||training||!runModes().length||!currentPose?.valid||!currentPose?.planning_valid;
 $('run').title=training?'A training is running; timed runs are paused so it cannot distort them.':!runModes().length?'Add an available arm first.':'Run the shown request with the chosen arms.';
 $('play').disabled=busy||!result||!maxTime();$('play').textContent=playing?'Pause':'Play';$('timeline').disabled=busy||!result||!maxTime();$('stop').disabled=!busy&&!autoPending;
}
function setQ(values){q=values.slice();}
async function updatePose(){if(!cfg)return;const id=++poseRequest;resetReplay();currentPose=null;controls();try{const data=await api('/api/pose',payload());if(id!==poseRequest)return;currentPose=data;status(!data.valid?'Starting pose is not collision-free.':!data.planning_valid?'Starting pose does not meet the planning clearance margin.':'Ready.',!data.valid||!data.planning_valid);render();}catch(e){if(id===poseRequest){status(e.message,true);controls();}}}
async function randomize(){const id=++poseRequest;
 const data=await api('/api/random',{...payload(),random_goal:false});
 if(id!==poseRequest)return false;setQ(data.q);currentPose=data;showSource(data);return true;}
function refText(c){if(!c)return '';const parts=Object.entries(c).filter(([,v])=>Number.isFinite(v?.total_s)).map(([k,v])=>`${shortNames[k]||k} ${fmt(v.total_s,2)} s`);return parts.length?`reference: ${parts.join(' · ')}`:'';}
function showSource(data){const s=data?.source;let text;
 if(s)text=s.label+(s.classical?` · ${refText(s.classical)}`:'');
 else text=`Fixed scene · goal ${goalKeyNow()} · x ${fmt(data.goal[0])} m · y ${fmt(data.goal[1])} m`;
 $('requestLabel').textContent=text;}
let loading=false;
async function loadRequest(){loading=true;currentPose=null;$('requestLabel').textContent='Loading the request…';status('Loading the request…');controls();try{if(await randomize())await updatePose();}catch(e){status(e.message,true);}finally{loading=false;controls();}}
async function startRun(){
 if(busy)return;
 const generation=++runGeneration;
 poseRequest++;resetReplay();busy=true;controls();
 try{
  if(!currentPose?.valid){busy=false;render();return;}
  const data=payload();render();

  while(pendingView||viewSending){await flushView();if(generation!==runGeneration)return;}
  const modes=runModes(),usesPolicy=modes.includes('policy');
  const job=await api('/api/run',{...data,policy:usesPolicy?policyKey:null,modes});
  if(generation!==runGeneration){api('/api/cancel',{id:job.id}).catch(()=>{});return;}
  jobId=job.id;result={id:job.id,query:currentPose||data,label:$('requestLabel').textContent,policy:usesPolicy?policyKey:null,lanes:arms.map(arm=>({...arm,state:modes.includes(arm.mode)?'queued':'off',name:armName(arm),query:data}))};
  cursor=0;render();
  status('Running the arms one after another, as in the evaluation…');
  await pollJob(job.id);
 }catch(e){if(generation!==runGeneration)return;busy=false;jobId=null;result=null;render();status(e.message,true);}
}
async function nextRequest(){cycleIndex++;if(cycleIndex%cycleLength()===0)advanceSeed();
 syncGoal();resetReplay();await loadRequest();}
let autoPending=false,autoGen=0;
async function continueRuns(){const gen=++autoGen;autoPending=true;controls();
 for(let skipped=0;skipped<=20;skipped++){await nextRequest();if(gen!==autoGen||!$('autoNext').checked)break;
  if(currentPose?.valid&&currentPose?.planning_valid){autoPending=false;startRun();return;}
  status(`Request skipped: ${$('errorNotice').hidden?'starting pose not usable':$('errorNotice').textContent}`);}
 autoPending=false;controls();}
async function pollJob(id){let polls=0,misses=0;try{while(jobId===id){let data;try{data=await api('/api/jobs/'+id);misses=0;}catch(e){if(++misses>3||/expired/.test(e.message))throw e;await new Promise(r=>setTimeout(r,1500));continue;}if(jobId!==id)return;if(data.cancelled){busy=false;setContinuous(false);resetReplay();status('Run stopped. Its partial result was discarded.');return;}result={...data,label:result?.label,policy:result?.policy};render();if(data.done){busy=false;saveRecords();$('timeline').max=maxTime();status(data.lanes.some(l=>l.state==='error')?'One or more arms could not complete. See the outcome.':maxTime()>0?'Calculation complete. Replaying recorded physics with motion starts aligned.':'Calculation complete. No arm moved, so there is nothing to replay.',data.lanes.some(l=>l.state==='error'));if(maxTime()>0)play();else{render();queueNext();}return;}if(++polls%20===0)status('The server is still planning or validating. Each arm keeps its own physics state.');await new Promise(r=>setTimeout(r,600));}}catch(e){if(jobId!==id)return;api('/api/cancel',{id}).catch(()=>{});busy=false;status(e.message+' Start a new run to retry.',true);controls();}}
function play(){clearTimeout(autoTimer);if(!result||!maxTime())return;if(cursor>=maxTime()-.001)cursor=0;playing=true;lastTick=performance.now();controls();replayStatus();cancelAnimationFrame(replayFrame);replayFrame=requestAnimationFrame(tick);}
function tick(now){replayFrame=null;if(!playing)return;cursor=Math.min(maxTime(),cursor+Math.max(0,now-lastTick)/1000);lastTick=now;render();if(cursor>=maxTime()){playing=false;controls();replayStatus();queueNext();}else replayFrame=requestAnimationFrame(tick);}
function queueNext(){clearTimeout(autoTimer);if($('autoNext').checked&&result){autoPending=true;controls();autoTimer=setTimeout(()=>{autoPending=false;if($('autoNext').checked)continueRuns();else controls();},4000);}}
function setContinuous(on){$('autoNext').checked=on;try{store.setItem('physicsContinuous',on?'1':'0');}catch{}}
function stopRun(){clearTimeout(autoTimer);autoGen++;autoPending=false;setContinuous(false);
 if(busy){if(jobId)api('/api/cancel',{id:jobId}).catch(()=>{});runGeneration++;poseRequest++;jobId=null;busy=false;resetReplay();status('Run stopped. Its partial result was discarded.');}
 else{controls();status('Continuous runs stopped. The last result stays visible.');}}
function pause(){playing=false;cancelAnimationFrame(replayFrame);clearTimeout(autoTimer);controls();if(result)replayStatus();}
function replayStatus(){const error=result.lanes.some(l=>l.state==='error');const active='Replaying recorded physics with motion starts aligned.';status((playing?active:cursor>=maxTime()?'Replay complete. Inspect an arm or scrub to review.':'Replay paused. Press Play to continue.')+(error?' One or more arms could not complete. See the outcome.':''),error);draw();}
function frameStatus(lane){if(lane.state==='off')return 'Not in this run';const outcome=lane.outcome==='plan_failed'&&lane.feasibility==='certified_infeasible'?'Geometrically infeasible':outcomeLabel(lane.outcome,lane.reason);if(lane.state==='done'&&!lane.metrics.physical_s)return 'No motion · '+outcome;if(lane.state==='error')return lane.reason;if(lane.state==='queued')return'Queued';if(lane.state!=='done')return'Calculating…';if(cursor<lane.metrics.physical_s-.001)return 'Executing';return outcome;}
function render(){if(!cfg)return;controls();renderArms();draw();$('clock').textContent=`${fmt(cursor)} / ${fmt(maxTime())} s`;$('timeline').value=cursor;renderCompare(result?.lanes||arms.map(arm=>({...arm,state:'setup'})));}
const sumOf=a=>Array.isArray(a)?a.reduce((x,y)=>x+y,0):null;
const COMPARE=[
 {label:'Total completion · s',get:l=>l.mission_s,d:2,best:'min',group:true},

 {label:'Planning · s',get:l=>l.planning_s,d:3,best:'min'},
 {label:'Post-processing · s',get:l=>l.preparation_s,d:3,best:'min'},
 {label:'Motion · s',get:l=>l.metrics?.physical_s,d:2,best:'min'},
 {label:'Tool path · m',get:l=>l.metrics?.distance_m,d:3,best:'min',group:true,more:true},
 {label:'Joint traverse · rad',more:true,get:l=>sumOf(l.metrics?.joint_traverse_rad),d:2,best:'min'},
 {label:'Average speed · m/s',more:true,get:l=>l.metrics?.mean_speed_m_s,d:3,best:'max'},
 {label:'Peak torque · % of limit',more:true,get:l=>{const t=l.metrics?.peak_torque_nm,c=cfg?.robot?.max_joint_torque_nm;return Array.isArray(t)&&Array.isArray(c)?100*Math.max(...t.map((v,i)=>v/c[i])):null;},d:0,best:null,help:'Highest torque of any joint relative to its limit. Informational: more torque can mean a faster or a more wasteful motion, so no winner is marked.'},
 {label:'Joint jerk RMS · rad/s³',more:true,get:l=>l.metrics?.jerk_rms_rad_s3,d:0,best:'min'},

 {label:'Joint reversals',more:true,help:'Direction changes of each joint on the executed reference curve (10 ms samples, 0.05 rad/s deadband). Cell: total over all joints; below: per joint.',get:l=>sumOf(l.joint_reversals),detail:l=>Array.isArray(l.joint_reversals)?l.joint_reversals.join(' · '):'',d:0,md:1,best:'min'},
 {label:'Raw path waypoints',more:true,get:l=>l.raw_waypoint_count,d:0,best:null,group:true}];
function sessionRows(lane){
 const set=armSet();return records.filter(r=>r.source===SRC()&&r.set===set&&r.mode===lane.mode&&r.success);}
function sessionMean(read,rows){
 const values=rows.map(read).filter(Number.isFinite);
 return values.length>1?{mean:values.reduce((a,b)=>a+b,0)/values.length,n:values.length}:null;}
const LIVE={'Motion · s':f=>f.t,'Tool path · m':f=>f.distance,'Joint traverse · rad':f=>sumOf(f.joint_arc),
 'Average speed · m/s':f=>f.mean_speed,
 'Joint jerk RMS · rad/s³':f=>f.jerk_rms};
let shapeKey='',liveCells=[];
function replaying(lane){const v=viewFrame(lane);return v&&v.physical<lane.metrics.physical_s-1e-9?v.frame:null;}

function cellHTML(text,sub,win){
 return `<td class="${win?'win':''}">${text===null?'&mdash;':text}<small>${sub||'&nbsp;'}</small></td>`;
}
function buildCompare(all){
 const lanes=all.map((l,i)=>({...l,_i:i})).filter(l=>!isOff(l.mode)&&(l.state==='setup'?available(l.mode):l.state!=='off'));
 const table=$('compare'),head=table.querySelector('thead'),body=table.querySelector('tbody');table.style.setProperty('--compare-lanes',lanes.length);
 head.innerHTML='<tr><th scope="col"></th>'+lanes.map(lane=>`<th scope="col" title="${escapeHTML(armName(lane))}${lane.mode==='policy'&&result?.policy?' · '+escapeHTML(policyLabel(result.policy)):''}"><span class="arm-head"><i class="swatch" style="background:${palette[lane._i]}"></i>${escapeHTML(armShort(lane))}</span><small>${lane.state==='setup'?(busy?'Preparing…':'Ready'):escapeHTML(frameStatus(lane))+(lane.queue_s>.5?` · queued ${fmt(lane.queue_s,1)} s`:'')}</small></th>`).join('')+'</tr>';
 const reachedIds=lanes.map(l=>new Set(sessionRows(l).map(r=>r.id)));
 const common=new Set(reachedIds.length?[...reachedIds[0]].filter(id=>reachedIds.every(ids=>ids.has(id))):[]);
 const laneRows=lanes.map(lane=>sessionRows(lane).filter(r=>common.has(r.id)));
 const rows=['<tr class="group"><th scope="row">Outcome</th>'+lanes.map(lane=>cellHTML(
   lane.state==='done'?escapeHTML(outcomeLabel(lane.outcome,lane.reason)):lane.state==='error'?'Error':null,lane.state==='done'&&!lane.success&&lane.reason?.startsWith('rejected:kto:')?escapeHTML(lane.reason.slice(13)):'',false)).join('')+'</tr>',
  `<tr class="group"><th scope="row" title="Means (⌀) use only the ${common.size} runs whose goal every shown arm reached, so all arms are compared on the same requests.">Session</th>`+lanes.map(lane=>{
   const t=totals[totalKey(lane.mode)];
   const off='';
   if(!t||!(t.n||t.kto))return `<td${off}>&mdash;<small>&nbsp;</small></td>`;
   const fails=Object.entries(t.fails||{}).sort((a,b)=>b[1]-a[1]).map(([f,c])=>`${c} × ${f}`);
   const parts=[t.n?`${fmt(100*t.reached/t.n,0)} % reached`:'nothing counted yet'];
   parts.push(fails.length?fails.join(' · '):'no failures');
   const note=parts.join(' · ');
   return `<td${off} title="${escapeHTML(note)}">${t.reached} / ${t.n} reached<small>${escapeHTML(note)}</small></td>`;}).join('')+'</tr>'];
 const live=[];
 for(const metric of COMPARE){
  const values=lanes.map(lane=>{const v=lane.state==='done'?metric.get(lane):null;return Number.isFinite(v)?v:null;});

  const avgs=lanes.map((lane,i)=>sessionMean(metric.get,laneRows[i]));
  const means=avgs.map(a=>a&&Number.isFinite(a.mean)?a.mean:null);
  const md=metric.md??metric.d;
  const meanShown=means.map(v=>v===null?null:fmt(v,md));
  const meanPool=means.filter(v=>v!==null);
  let target=null;
  if(metric.best&&meanPool.length>1){
   const extremum=metric.best==='min'?Math.min(...meanPool):Math.max(...meanPool);
   const label=fmt(extremum,md);
   if(meanShown.filter(s=>s===label).length===1)target=label;
  }
  const at=LIVE[metric.label]?live.push(metric)-1:-1;
  rows.push(`<tr class="${metric.group?'group':''}${metric.more?' more':''}"${at<0?'':` data-live="${at}"`}><th scope="row" title="${escapeHTML(metric.help||'')}">${metric.label}</th>`+lanes.map((lane,i)=>{
   const win=target!==null&&meanShown[i]===target;

   const avg=avgs[i];
   const detail=values[i]!==null&&metric.detail?metric.detail(lane):'';
   const mean=avg?`⌀ ${fmt(avg.mean,md)}`:'';
   return cellHTML(values[i]===null?null:fmt(values[i],metric.d),
                   detail&&mean?`${detail} (${mean})`:detail||mean,win);}).join('')+'</tr>');}
 body.innerHTML=rows.join('');
 liveCells=[];
 live.forEach((metric,k)=>{const tr=body.querySelector('tr[data-live="'+k+'"]');if(!tr)return;
  tr.querySelectorAll('td').forEach((td,i)=>liveCells.push({td,lane:lanes[i]._i,get:LIVE[metric.label],d:metric.d,rest:td.firstChild.nodeValue}));});
}

function updateLive(lanes){
 if(!liveCells.length)return;
 const frames=lanes.map(l=>l.state==='done'?replaying(l):null);
 for(const c of liveCells){
  const frame=frames[c.lane];
  if(!frame){if(c.td.classList.contains('live')){c.td.classList.remove('live');c.td.firstChild.nodeValue=c.rest;}continue;}
  const now=c.get(frame);if(!Number.isFinite(now))continue;
  const text=fmt(now,c.d);
  if(c.td.firstChild.nodeValue!==text)c.td.firstChild.nodeValue=text;
  c.td.classList.add('live');}}
function renderResultInfo(){$('resultInfo').textContent=result?`Result: ${result.label||'request'}${result.policy?' · checkpoint '+policyLabel(result.policy):''}`:`No result yet${runModes().includes('policy')?' · checkpoint '+policyLabel(policyKey):''}`;}
function renderCompare(lanes){
 renderResultInfo();
 const shape=JSON.stringify([SRC(),result?.id,busy,records.length,policies.length,
  lanes.map(l=>[l.mode,l.state,l.success,l.outcome,frameStatus(l)]),[...offModes]]);
 if(shape!==shapeKey){shapeKey=shape;buildCompare(lanes);}
 updateLive(lanes);}
const viewId=globalThis.crypto?.randomUUID?.()||Date.now().toString(36)+'-'+Math.random().toString(36).slice(2);
let pendingView=null,viewSending=null,viewTimer=null,mapTimer=null,lastViewSent=0,viewError=null;
function poseMapDescription(m,twoLink){
 if(m.status==='queued')return ['High-resolution heat map queued for computation.',''];
 if(m.status==='saving')return ['Saving the completed heat map…',''];
 if(m.status==='computing')return [Number.isInteger(m.completed_cells)&&m.grid_points>0?
  `Computing the heat map… ${fmt(100*m.completed_cells/m.grid_points,1)}% of grid cells checked.`:
  'Computing the heat map for this scene…',''];
 if(m.status==='waiting')return ['Heat map follows after the run.',''];
 if(m.status!=='ready')return [`Map unavailable: ${m.reason||'unknown error'}`,''];
 const number=n=>n.toLocaleString('en-US');
 const grid=Number.isInteger(m.grid_points)?`${number(m.grid_points)} grid points · `:'';
 const checked=Number.isInteger(m.checked_postures)?`${number(m.checked_postures)} postures checked · `:'';
 const sampling=twoLink?'':`${m.candidates} candidates per grid point · `;
 const info=`${grid}${checked}${sampling}${fmt(m.cell_m*1000,1)} mm grid · ${m.source} map`;
 const explain=twoLink?
  'Every exact regular grid point is checked for both possible elbow postures. With two links, the endpoint fixes the tool orientation, so more orientation samples cannot add distinct solutions.':
  `Share of the ${m.candidates} candidate postures per grid point that are free: orientations of the last link, each with both elbow branches. Free means inside the joint limits and at the safety margin from the wall.`;
 return [info,explain];
}
async function flushView(){
 clearTimeout(viewTimer);viewTimer=null;if(viewSending)return viewSending;if(!pendingView)return;
 const value=pendingView;pendingView=null;lastViewSent=performance.now();
 viewSending=(async()=>{
  try{const data=await api('/api/view',value);const url=new URL('physics_dashboard_meshcat.html',location.href);url.searchParams.set('port',new URL(data.meshcat_url).port);if(data.session)url.searchParams.set('session',data.session);
   if($('arena').src!==url.href)$('arena').src=url.href;
   clearTimeout(mapTimer);mapTimer=null;
   if($('poseMapInfo')&&data.pose_map){const m=data.pose_map;if(value.reach&&['computing','waiting','queued','saving'].includes(m.status))mapTimer=setTimeout(()=>{mapTimer=null;draw();},1500);
    const [info,explain]=poseMapDescription(m,cfg.variant==='2r');
    $('poseMapInfo').textContent=info;$('poseMapExplain').textContent=explain;}

   if(viewError){if($('errorNotice')?.textContent===viewError)status('',false);viewError=null;}
  }catch(e){

   viewError=`Viewer update failed: ${e?.message||e}. Reload the page if the scene stays frozen.`;status(viewError,true);}
  finally{viewSending=null;if(pendingView)viewTimer=setTimeout(flushView,Math.max(0,50-(performance.now()-lastViewSent)));}
 })();
 return viewSending;
}
function draw(){
 if(!cfg)return;
 clearTimeout(mapTimer);mapTimer=null;
 if($('poseLegend'))$('poseLegend').hidden=!!episodeView||!$('reachLayer').checked;
 if(episodeView){pendingView={lanes:1,view_id:viewId,training_episode:episodeView.id,cursor_s:episodeView.cursor,reference:true,executed:true,route:true,found:true,ghost:false,search:false,reach:false};
  if(!viewSending&&!viewTimer)viewTimer=setTimeout(flushView,Math.max(0,50-(performance.now()-lastViewSent)));return;}
 const setup=result?.query||payload();
 const sf=setup.source&&typeof setup.source==='object'?{source:setup.source.kind,set:setup.source.set,index:setup.source.index}:typeof setup.source==='string'?{source:setup.source,set:setup.set,index:setup.index}:{};
 pendingView={...sf,lanes:arms.length,view_id:viewId,id:result?.id||null,cursor_s:cursor,q:setup.q,scene:setup.scene,world:setup.scene==='drawn'?setup.world:null,goal:currentPose?.goal||setup.goal,
  off:arms.map((a,i)=>isOff(a.mode)||!available(a.mode)?i:-1).filter(i=>i>=0),reference:$('reference').checked,executed:$('reference').checked,found:$('foundPath').checked,route:$('route').checked,ghost:$('ghost').checked,search:$('searchTree').checked,reach:$('reachLayer').checked,goal_q:result?null:goalQ.slice()};
 if(!viewSending&&!viewTimer)viewTimer=setTimeout(flushView,Math.max(0,50-(performance.now()-lastViewSent)));
}
function saveRecords(){
 if(records.some(r=>r.id===result.id))return;
 const lanes=result.lanes.filter(l=>l.state!=='off'),set=lanes.map(l=>l.mode).join(',');
 if(set!==armSet())return;
 for(const lane of lanes){
  const k=SRC()+'|'+set+'|'+lane.mode,t=totals[k]||(totals[k]={n:0,reached:0,kto:0,fails:{}});
  t.n++;
  if(lane.success===true)t.reached++;
  else{const f=lane.state==='error'?'Error':outcomeLabel(lane.outcome||'error',lane.reason);t.fails[f]=(t.fails[f]||0)+1;
   if(lane.outcome==='trajectory_rejected'&&String(lane.reason||'').startsWith('rejected:kto:'))t.kto=(t.kto||0)+1;}
  records.push({id:result.id,source:SRC(),set,mode:lane.mode,success:lane.success===true,
   joint_reversals:lane.joint_reversals,mission_s:lane.mission_s,planning_s:lane.planning_s,
   preparation_s:lane.preparation_s,raw_waypoint_count:lane.raw_waypoint_count,metrics:lane.metrics});
 }
 const keep=new Set([...new Set(records.map(r=>r.id))].slice(-100));records=records.filter(r=>keep.has(r.id));
 try{store.setItem('physicsTotals',JSON.stringify(totals));store.setItem('physicsTotalsVersion','2');store.setItem('physicsResults',JSON.stringify(records));}catch{}
 render();
}
$('run').onclick=startRun;
$('play').onclick=()=>playing?pause():play();$('timeline').step='any';$('timeline').oninput=()=>{pause();cursor=Number($('timeline').value);replayStatus();render();};
['reference','route','ghost','searchTree','foundPath','reachLayer'].forEach(id=>$(id).onchange=draw);
$('requestSource').onchange=()=>{source=$('requestSource').value;saveSource();cycleIndex=0;syncGoal();resetReplay();controls();loadRequest();};
$('policySelect').onchange=()=>{policyKey=$('policySelect').value;saveSource();if(result?.policy&&result.policy!==policyKey&&!busy){resetReplay();status('Checkpoint changed. Run again to see its result.');}shapeKey='';armsKey='';render();};
async function loadPolicies(){try{const r=await api('/api/policies');policies=r.policies;if(policyKey&&!policies.some(p=>p.key===policyKey)){if(policies.length)status('The chosen checkpoint is gone; using the default.');policyKey=r.default;}else if(!policyKey)policyKey=r.default;
 $('policySelect').innerHTML=policies.length?policies.map(p=>`<option value="${escapeHTML(p.key)}" title="${escapeHTML(p.path||p.key)}">${escapeHTML(p.label)} · ${escapeHTML(p.folder)}</option>`).join(''):'<option value="">No checkpoint found</option>';
 if(policyKey)$('policySelect').value=policyKey;saveSource();armsKey='';shapeKey='';render();}catch(e){status(e.message,true);}}
$('stop').onclick=stopRun;
$('autoNext').onchange=()=>{setContinuous($('autoNext').checked);if(!$('autoNext').checked){clearTimeout(autoTimer);autoGen++;autoPending=false;controls();}else if(result&&!busy&&!playing)queueNext();};
$('drawNext').onclick=()=>nextRequest();
$('armChips').onclick=e=>{const b=e.target.closest('button[data-remove]');if(b)removeArm(b.dataset.remove);};
$('addArm').onchange=()=>{if($('addArm').value)addArm($('addArm').value);};
$('moreMetrics').onclick=()=>{const on=!$('compare').classList.contains('show-more');$('compare').classList.toggle('show-more',on);$('moreMetrics').setAttribute('aria-pressed',String(on));$('moreMetrics').textContent=on?'Fewer metrics':'More metrics';try{store.setItem('physicsMore',on?'1':'0');}catch{}};
$('clearHistory').onclick=async()=>{if(!await confirmAction('Clear the session statistics of every request source, arm set and checkpoint stored in this browser? This cannot be undone.'))return;records=[];totals={};try{store.removeItem('physicsResults');store.removeItem('physicsTotals');store.removeItem('physicsCells');}catch{}shapeKey='';render();};
window.addEventListener('resize',()=>{draw();});
let trainTimer=null,trainState='idle',trainBank=true,trainPolling=false,episodePolling=false,trainElsewhere=false,runElsewhere=false,trainSubmitting=false;
const duration=s=>!Number.isFinite(s)?'—':s>=3600?`${fmt(s/3600,1)} h`:s>=60?`${fmt(s/60,0)} min`:`${fmt(s,0)} s`;
function trainErr(text){$('trainError').hidden=!text;$('trainError').textContent=text||'';}
let trainFolder=null;
function showTraining(t){
 trainBank=!!t.bank;
 if(t.folder!==trainFolder){trainFolder=t.folder;if(t.settings){$('trainSteps').value=t.settings.steps;$('trainWidth').value=t.settings.width;}$('trainEpisode').textContent='';$('trainRefs').textContent='';endEpisodes();}
 const was=trainState;trainState=t.state;if(was!==trainState){controls();if($('trainingCard').open)listRuns();if(was!=='idle')loadPolicies();}followEpisodes();const running=t.state==='running'||t.state==='stopping';
 trainElsewhere=!running&&!!t.training_elsewhere;runElsewhere=!!t.run_elsewhere;controls();
 $('trainStop').disabled=!running;$('trainStart').title=!t.bank?'Training bank missing':runElsewhere?'A timed run is calculating in the other dashboard.':trainElsewhere?'A training is running in the other dashboard.':'';
 $('trainSteps').disabled=running||trainSubmitting;$('trainWidth').disabled=running||trainSubmitting;
 $('trainStatus').hidden=t.state==='idle';
 if(t.state==='idle')return;
 const share=t.total?t.steps/t.total:0;
 $('trainProgress').style.width=`${Math.min(100,100*share)}%`;
 $('trainLine').textContent=`Width ${t.settings.width} · ${t.state} · ${fmt(100*share,0)} % of ${fmt(t.total,0)} steps`+(running&&t.rate?` · ${duration(t.eta_s)} left`:'');
 const ep=t.episodes||{},n=Object.values(ep).reduce((a,b)=>a+b,0),hit=ep.reached||0,
  kto=Object.entries(ep).filter(([k])=>k.startsWith('rejected:kto')).reduce((a,[,v])=>a+v,0);
 $('trainDetail').textContent=[n?`Recent episodes: ${fmt(100*hit/n,0)} % reached (${hit} of ${n})`+(kto?`, ${kto} curve step failed`:''):'',
  `${t.checkpoints?.length||0} checkpoint${t.checkpoints?.length===1?'':'s'} · ${t.folder}`,t.last_line&&`Log: ${t.last_line}`].filter(Boolean).join('\n');}
let episodeView=null,episodeTimer=null,episodeFrame=null,episodeGeneration=0;
const watching=()=>(trainState==='running'||trainState==='stopping')&&$('trainWatch').checked&&!busy;
function endEpisodes(){episodeGeneration++;clearTimeout(episodeTimer);episodeTimer=null;cancelAnimationFrame(episodeFrame);episodeFrame=null;if(episodeView){episodeView=null;draw();}}
function animateEpisode(now){episodeFrame=null;if(!episodeView)return;episodeView.cursor=Math.min(episodeView.replay_s,(now-episodeView.t0)/1000);draw();
 if(episodeView.cursor<episodeView.replay_s)episodeFrame=requestAnimationFrame(animateEpisode);else{clearTimeout(episodeTimer);episodeTimer=setTimeout(pollEpisode,700);}}
function episodeText(e){
 const walk=`${e.steps} steps, ${e.blocked} blocked, ${e.pruned_count||0} waypoints after shortening`;
 let what;
 if(e.code==='reached'){const s=e.simulation;
  what=`goal reached · KTO/TOPPRA curve ${fmt(e.duration_s,2)} s vs bank reference ${fmt(e.reference_s,2)} s`+(s?` · simulated: ${s.outcome==='held'?'held at the goal':s.outcome}${Number.isFinite(s.peak_torque_share)?`, peak torque ${fmt(100*s.peak_torque_share,0)} %`:''}`:'');}
 else if(e.code==='horizon')what='ran out of steps · no curve, geometric replay of the walk';
 else what=`goal reached, curve step refused (${String(e.code).replace(/^rejected:kto:/,'KTO: ')}) · no bonus, geometric replay`;
 return `Env ${(e.env??0)+1}, episode ${fmt(e.episode,0)} (≈ step ${fmt(e.approx_steps,0)}): ${what}\n${walk}\nwalked path: dashed with small dots · shortened: solid with dots · KTO/TOPPRA curve: faint dashed · executed: solid, grows with the arm`;}
async function pollEpisode(){
 clearTimeout(episodeTimer);episodeTimer=null;if(!watching()){endEpisodes();return;}if(episodePolling)return;episodePolling=true;const generation=episodeGeneration;let fresh=false;
 try{const e=await api('/api/training/episode?kind='+($('trainReached').checked?'reached':'any'));
  if(generation===episodeGeneration&&watching()&&(!episodeView||episodeView.id!==e.id)){fresh=true;cancelAnimationFrame(episodeFrame);episodeView={id:e.id,replay_s:e.replay_s,cursor:0,t0:performance.now()};episodeFrame=requestAnimationFrame(animateEpisode);
   $('trainRefs').textContent=refText(e.classical)?`Same task in the bank, ${refText(e.classical)}`:'';
   $('trainEpisode').textContent=episodeText(e);}}
 catch(err){if(generation===episodeGeneration&&watching())$('trainEpisode').textContent=err.message;}
 episodePolling=false;
 if(!watching())endEpisodes();else if(!fresh&&!episodeFrame)episodeTimer=setTimeout(pollEpisode,1000);}
function followEpisodes(){if(watching()){if(!episodeTimer&&!episodeView)pollEpisode();}else endEpisodes();}
let runsPolling=false;
async function listRuns(){if(runsPolling)return;runsPolling=true;try{const r=(await api('/api/training/runs')).runs;$('trainRuns').hidden=!r.length;
 $('trainRunList').innerHTML=r.map(x=>`<li><span>${escapeHTML(x.name.replace(/^dash-/,''))}<small>${x.running?'running · ':''}${x.checkpoints} checkpoint${x.checkpoints===1?'':'s'} · ${fmt(x.mb,1)} MB</small></span>${x.running?'':`<button class="quiet" data-run="${escapeHTML(x.name)}">Delete</button>`}</li>`).join('');}catch(e){trainErr(e.message);}finally{runsPolling=false;}}
$('trainRunList').onclick=async e=>{const b=e.target.closest('button[data-run]');if(!b)return;if(!await confirmAction(`Delete ${b.dataset.run} with its checkpoints?`))return;
 try{await api('/api/training/delete',{name:b.dataset.run});trainErr('');}catch(err){trainErr(err.message);}listRuns();pollTraining();loadPolicies();};
async function pollTraining(){if(trainPolling)return;trainPolling=true;clearTimeout(trainTimer);trainTimer=null;try{showTraining(await api('/api/training'));if($('trainingCard').open)listRuns();}catch(e){$('trainLine').textContent=e.message;}finally{trainPolling=false;}
 trainTimer=setTimeout(pollTraining,$('trainingCard').open||trainState==='running'||trainState==='stopping'?5000:15000);}
$('trainWatch').onchange=followEpisodes;
$('trainReached').onchange=()=>{endEpisodes();$('trainEpisode').textContent='';$('trainRefs').textContent='';followEpisodes();};
$('trainingCard').ontoggle=()=>{if($('trainingCard').open){pollTraining();listRuns();loadPolicies();}};
$('trainStart').onclick=async()=>{if(trainSubmitting)return;trainSubmitting=true;controls();try{showTraining(await api('/api/training/start',{steps:Number($('trainSteps').value),width:Number($('trainWidth').value)}));trainErr('');}catch(e){trainErr(e.message);}finally{trainSubmitting=false;controls();}pollTraining();};
$('trainStop').onclick=async()=>{try{showTraining(await api('/api/training/stop',{}));trainErr('');}catch(e){trainErr(e.message);}pollTraining();};
async function init(){
 try{
  cfg=await api('/api/config');
  if(cfg.variant==='2r'){
   $('reachLayer').checked=true;$('poseLegend').classList.add('two-link');
   $('poseLegendTitle').textContent='Free poses at each grid point';
   $('poseScale').innerHTML='<span>0 poses</span><span>1 pose</span><span>2 poses</span>';
   $('poseOneInfo').hidden=true;
   $('poseMapCaveat').textContent='White: no free elbow branch. Pink: one. Green: both. Singular boundary poses are excluded. These endpoint checks do not establish a path from the current start.';
  }

  arms=cfg.lanes.map(mode=>({mode}));
  currentPose=null;q=cfg.pose.q.slice();syncGoal();
  const opts=[['preset-nominal','Fixed scene · wall on'],['preset-free','Fixed scene · wall off']];
  $('requestSource').innerHTML=opts.map(([k,l])=>`<option value="${k}">${escapeHTML(l)}</option>`).join('');
  if(!opts.some(([k])=>k===source))source='preset-nominal';$('requestSource').value=source;saveSource();
  $('armLabel').textContent=cfg.arm_label;if(cfg.peer||cfg.peer_port){const peer=new URL(cfg.peer||'/',location.href);if(cfg.peer_port)peer.port=cfg.peer_port;$('peerLink').href=peer.href;$('peerLink').textContent=cfg.peer_label||'Other arm';$('peerLink').hidden=false;}
  await loadPolicies();
  if(!runModes().length){offModes.delete('rrt_connect');saveArms();}
  try{$('autoNext').checked=store.getItem('physicsContinuous')==='1';if(store.getItem('physicsMore')==='1')$('moreMetrics').click();}catch{}
  setGoalQ(goalPose(Object.keys(cfg.goals)[0]));
  render();pollTraining();if(await randomize())await updatePose();
 }catch(e){status(e.message,true);}
}
init();
new ResizeObserver(()=>document.documentElement.style.setProperty('--bar',document.querySelector('.bar').offsetHeight+'px')).observe(document.querySelector('.bar'));
document.addEventListener('click',e=>{const m=$('viewMenu');if(m.open&&!m.contains(e.target))m.open=false;});
