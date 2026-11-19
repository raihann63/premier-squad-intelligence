'use strict';

/**
 * PitchPilot AI frontend — vanilla JavaScript.
 * Change this base URL only when deploying the FastAPI backend to an HTTPS host.
 * Never expose private model files or secrets to the browser.
 */
const API_BASE_URL = window.PITCHPILOT_API_BASE || 'http://127.0.0.1:8000';
const POSITIONS = ['GK', 'DEF', 'MID', 'FWD'];
const FORMATIONS = {
  '4-3-3': { GK: 1, DEF: 4, MID: 3, FWD: 3 },
  '4-4-2': { GK: 1, DEF: 4, MID: 4, FWD: 2 },
  '4-2-3-1': { GK: 1, DEF: 4, MID: 5, FWD: 1 },
  '3-5-2': { GK: 1, DEF: 3, MID: 5, FWD: 2 },
  '3-4-3': { GK: 1, DEF: 3, MID: 4, FWD: 3 },
  '5-3-2': { GK: 1, DEF: 5, MID: 3, FWD: 2 },
};
const dom = (id) => document.getElementById(id);
const RESULT_STORAGE_KEY = 'pitchpilot_result_v2';
const SELECTION_STORAGE_KEY = 'pitchpilot_selection_v2';
const ACTIVE_PLAYER_STORAGE_KEY = 'pitchpilot_player_v2';
function readSession(key) {
  try { return JSON.parse(sessionStorage.getItem(key) || 'null'); } catch { return null; }
}
function writeSession(key, value) {
  try { sessionStorage.setItem(key, JSON.stringify(value)); } catch { /* Session storage may be disabled. */ }
}
function deleteSession(key) {
  try { sessionStorage.removeItem(key); } catch { /* Storage may be disabled. */ }
}
function isSquadPage() { return Boolean(dom('selectionForm')); }
function isAnalyticsPage() { return Boolean(dom('playerTableBody')); }

const state = { teams: [], result: null, players: [], activePlayer: -1, busy: false, requestId: 0, managerSuggestions: [], managerEdited: false };
let toastTimer;
let playersFetchId = 0;
let managerFetchId = 0;
let managerOptionIndex = -1;

function escapeHTML(value) {
  return String(value ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
function numberText(value, digits = 0) {
  const n = Number(value);
  return value === null || value === undefined || !Number.isFinite(n) ? '—' : n.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}
function percent(value) {
  return numberText(value, 1) + (Number.isFinite(Number(value)) && value !== null ? '%' : '');
}
function widthPercent(value) {
  return Math.max(0, Math.min(100, Number(value) || 0));
}
function initials(name) {
  const parts = String(name ?? '').trim().split(/\s+/).filter(Boolean);
  return ((parts[0] || '?')[0] + (parts.length > 1 ? parts[parts.length - 1][0] : '')).toUpperCase();
}
function formatError(payload, fallback = 'Request failed.') {
  const detail = payload?.detail ?? payload?.message ?? fallback;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return detail.map((v) => v.msg || JSON.stringify(v)).join('; ');
  if (detail && typeof detail === 'object') {
    const deficits = detail.positions && Object.entries(detail.positions).map(([role, data]) => `${role}: need ${data.required}, available ${data.available}`);
    return [detail.message || fallback, ...(deficits || [])].join(' · ');
  }
  return fallback;
}
async function api(path, init = {}) {
  let response;
  try {
    response = await fetch(API_BASE_URL + path, {
      ...init,
      headers: { 'Accept': 'application/json', ...(init.body ? { 'Content-Type': 'application/json' } : {}), ...init.headers },
    });
  } catch (error) {
    throw new Error(`Cannot connect to ${API_BASE_URL}. Confirm FastAPI is running and the frontend is served on port 5500. (${error.message})`);
  }
  let body;
  try { body = await response.json(); } catch { body = {}; }
  if (!response.ok) throw new Error(formatError(body, `HTTP ${response.status}`));
  return body;
}
function setConnection(status, label) {
  if (!dom('serverStatus')) return;
  const indicator = dom('serverStatus').querySelector('.status-dot');
  indicator.className = 'status-dot ' + status;
  dom('serverStatusText').textContent = label;
}
function toast(message, isError = false) {
  const element = dom('toast');
  clearTimeout(toastTimer);
  element.textContent = message;
  element.className = 'toast visible' + (isError ? ' error' : '');
  toastTimer = setTimeout(() => element.classList.remove('visible'), 3700);
}
function showFormError(message, retry = false) {
  if (!dom('formError')) { toast(message, true); return; }
  const box = dom('formError');
  box.replaceChildren();
  box.append(document.createTextNode(message));
  if (retry) {
    const retryButton = document.createElement('button');
    retryButton.type = 'button';
    retryButton.textContent = ' Retry connection';
    retryButton.style.cssText = 'display:block;margin-top:9px;border:1px solid #ca8d89;border-radius:5px;background:transparent;color:inherit;padding:7px 10px;font-weight:700';
    retryButton.addEventListener('click', initialize);
    box.append(retryButton);
  }
  box.hidden = false;
}
function clearFormError() { if (dom('formError')) dom('formError').hidden = true; }

async function initialize() {
  if (isAnalyticsPage()) restoreAnalytics();
  if (!isSquadPage()) {
    try {
      const health = await api('/health');
      if (dom('teamsStat')) dom('teamsStat').textContent = numberText(health.teams);
      if (dom('rowsStat')) dom('rowsStat').textContent = numberText(health.data_rows);
      setConnection('online', 'Inference API connected');
    } catch (error) { setConnection('offline', 'API connection failed'); }
    return;
  }
  dom('teamSelect').disabled = true;
  dom('generateButton').disabled = true;
  setConnection('pending', 'Connecting to API');
  clearFormError();
  try {
    const [health, catalog] = await Promise.all([api('/health'), api('/api/teams')]);
    if (!Array.isArray(catalog.teams) || !Array.isArray(catalog.formations)) throw new Error('Unexpected /api/teams response format.');
    state.teams = catalog.teams;
    const saved = readSession(SELECTION_STORAGE_KEY) || {};
    populateTeams(catalog.teams, saved.team);
    populateFormations(catalog.formations);
    if (saved.formation && catalog.formations.includes(saved.formation)) {
      dom('formationSelect').value = saved.formation;
      updateFormationGuide();
    }
    if (saved.bench) {
      POSITIONS.forEach(pos => {
        const count = saved.bench[pos];
        if (Number.isInteger(count)) dom('bench' + pos).value = count;
      });
      updateBenchTotal();
    }
    dom('teamSelect').disabled = false;
    dom('generateButton').disabled = false;
    setConnection('online', 'Inference API connected');
    await Promise.all([updateAvailablePlayers(), updateManagerSuggestions()]);
    if (saved.team === dom('teamSelect').value && saved.manager) {
      dom('managerInput').value = saved.manager;
      dom('managerInput').dataset.autofilled = 'false';
      dom('pitchManager').textContent = saved.manager;
    }
    const cached = readSession(RESULT_STORAGE_KEY);
    if (cached && Array.isArray(cached.starting_xi) && cached.starting_xi.length === 11 &&
        cached.team === dom('teamSelect').value && cached.formation === dom('formationSelect').value &&
        cached.manager === dom('managerInput').value.trim()) {
      state.result = cached;
      state.players = [...cached.starting_xi.map(p => ({ ...p, group: 'Starter' })),
                       ...(cached.substitutes || []).map(p => ({ ...p, group: 'Bench' }))];
      renderResult();
    }
  } catch (error) {
    setConnection('offline', 'API connection failed');
    dom('teamSelect').innerHTML = '<option value="">Backend unavailable</option>';
    showFormError(error.message, true);
  }
}
function restoreAnalytics() {
  const cached = readSession(RESULT_STORAGE_KEY);
  if (!cached || !Array.isArray(cached.starting_xi) || cached.starting_xi.length !== 11 || !Array.isArray(cached.substitutes)) return;
  state.result = cached;
  state.players = [...cached.starting_xi.map(p => ({ ...p, group: 'Starter' })),
                   ...cached.substitutes.map(p => ({ ...p, group: 'Bench' }))];
  if (dom('analyticsNoSquad')) dom('analyticsNoSquad').hidden = true;
  renderTable();
  const previous = Number(readSession(ACTIVE_PLAYER_STORAGE_KEY));
  selectPlayer(Number.isInteger(previous) && previous >= 0 && previous < state.players.length ? previous : 0, false);
}

function populateTeams(teams, savedTeam = null) {
  const dropdown = dom('teamSelect');
  const previous = dropdown.value;
  dropdown.replaceChildren();
  teams.forEach((name) => dropdown.add(new Option(name, name)));
  const defaultTeam = (savedTeam && teams.includes(savedTeam)) ? savedTeam : teams.includes(previous) ? previous : (teams.includes('Arsenal') ? 'Arsenal' : teams[0]);
  if (defaultTeam) dropdown.value = defaultTeam;
  dom('pitchTeam').textContent = defaultTeam || 'Your matchday squad';
}
function populateFormations(formations) {
  const select = dom('formationSelect');
  const current = select.value;
  select.replaceChildren();
  formations.filter((f) => FORMATIONS[f]).forEach((formation) => select.add(new Option(formation, formation)));
  if (formations.includes(current)) select.value = current;
  updateFormationGuide();
}
function updateFormationGuide() {
  const value = dom('formationSelect').value;
  const shape = FORMATIONS[value];
  if (!shape) return;
  dom('formationGuide').textContent = `1 GK  ·  ${shape.DEF} DEF  ·  ${shape.MID} MID  ·  ${shape.FWD} FWD`;
  dom('pitchFormation').textContent = value;
}
async function updateAvailablePlayers() {
  const team = dom('teamSelect').value;
  if (!team) return;
  const runId = ++playersFetchId;
  dom('playersAvailable').textContent = 'Loading available players...';
  try {
    const data = await api('/api/players?team=' + encodeURIComponent(team));
    if (runId === playersFetchId) dom('playersAvailable').textContent = `${numberText(data.count)} unique player names in this club's dataset.`;
  } catch {
    if (runId === playersFetchId) dom('playersAvailable').textContent = 'Player preview unavailable. Try generating a squad.';
  }
}
/** Historical team-specific managers, never hardcoded into the frontend. */
function closeManagerSuggestions() {
  const list = dom('managerSuggestionList');
  list.hidden = true;
  dom('managerInput').setAttribute('aria-expanded', 'false');
  dom('managerInput').removeAttribute('aria-activedescendant');
  managerOptionIndex = -1;
}
function focusManagerOption(index) {
  const options = [...dom('managerSuggestionList').querySelectorAll('.manager-option')];
  if (!options.length) return;
  managerOptionIndex = (index + options.length) % options.length;
  options.forEach((option, i) => {
    option.classList.toggle('is-active', i === managerOptionIndex);
    option.setAttribute('aria-selected', String(i === managerOptionIndex));
  });
  const target = options[managerOptionIndex];
  dom('managerInput').setAttribute('aria-activedescendant', target.id);
  target.scrollIntoView({ block: 'nearest' });
}
function chooseManager(name) {
  const input = dom('managerInput');
  input.value = name;
  state.managerEdited = true;
  input.dataset.autofilled = 'false';
  closeManagerSuggestions();
  invalidateResult();
  clearFormError();
  input.focus();
}
function renderManagerSuggestions() {
  const input = dom('managerInput');
  const list = dom('managerSuggestionList');
  list.replaceChildren();
  managerOptionIndex = -1;
  input.removeAttribute('aria-activedescendant');
  const query = input.dataset.autofilled === 'true' ? '' : input.value.trim().toLocaleLowerCase();
  const matches = state.managerSuggestions.filter(name => name.toLocaleLowerCase().includes(query)).slice(0, 8);
  if (!state.managerSuggestions.length) {
    closeManagerSuggestions();
    return;
  }
  if (!matches.length) {
    const empty = document.createElement('p');
    empty.className = 'manager-no-match';
    empty.textContent = 'No matching historical manager. You can enter another name.';
    list.append(empty);
  } else {
    matches.forEach((name, i) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'manager-option';
      button.id = 'manager-option-' + i;
      button.setAttribute('role', 'option');
      button.setAttribute('aria-selected', 'false');
      const text = document.createElement('span');
      text.textContent = name;
      const detail = document.createElement('small');
      detail.textContent = 'Historical club manager';
      button.append(text, detail);
      button.addEventListener('pointerdown', event => event.preventDefault());
      button.addEventListener('click', () => chooseManager(name));
      list.append(button);
    });
  }
  list.hidden = false;
  input.setAttribute('aria-expanded', 'true');
}
async function updateManagerSuggestions() {
  const team = dom('teamSelect').value;
  const requestId = ++managerFetchId;
  const input = dom('managerInput');
  closeManagerSuggestions();
  state.managerSuggestions = [];
  state.managerEdited = false;
  input.value = '';
  input.dataset.autofilled = 'false';
  input.placeholder = 'Loading historical managers...';
  dom('managerHelp').textContent = 'Finding managers from the selected club’s historical dataset...';
  if (!team) return;
  try {
    const data = await api('/api/managers?team=' + encodeURIComponent(team));
    if (requestId !== managerFetchId || dom('teamSelect').value !== team) return;
    state.managerSuggestions = Array.isArray(data.managers) ? data.managers.filter(name => typeof name === 'string' && name.trim()) : [];
    input.placeholder = 'Search or enter manager name';
    if (!state.managerEdited && state.managerSuggestions.length && data.suggested) {
      input.value = data.suggested;
      input.dataset.autofilled = 'true';
      dom('pitchManager').textContent = data.suggested;
    }
    dom('managerHelp').textContent = state.managerSuggestions.length
      ? `${state.managerSuggestions.length} historical manager${state.managerSuggestions.length === 1 ? '' : 's'} found. Click the field to choose or type to search.`
      : 'No club-specific manager found in the dataset. You can enter a name manually.';
    if (document.activeElement === input) renderManagerSuggestions();
  } catch {
    if (requestId !== managerFetchId || dom('teamSelect').value !== team) return;
    input.placeholder = 'Enter manager name';
    dom('managerHelp').textContent = 'Manager suggestions unavailable. Manual entry still works.';
  }
}
function onManagerKeydown(event) {
  const list = dom('managerSuggestionList');
  if (event.key === 'Escape') {
    closeManagerSuggestions();
    return;
  }
  if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
    if (!state.managerSuggestions.length) return;
    event.preventDefault();
    if (list.hidden) renderManagerSuggestions();
    focusManagerOption(managerOptionIndex + (event.key === 'ArrowDown' ? 1 : -1));
  } else if (event.key === 'Enter' && !list.hidden && managerOptionIndex >= 0) {
    const options = list.querySelectorAll('.manager-option');
    const selected = options[managerOptionIndex];
    if (selected) {
      event.preventDefault();
      chooseManager(selected.querySelector('span').textContent);
    }
  }
}

function benchValues() {
  const result = {};
  for (const position of POSITIONS) {
    const field = dom('bench' + position);
    const value = Number(field.value);
    if (!Number.isInteger(value) || value < 0 || value > (position === 'GK' ? 4 : 9)) throw new Error(`Invalid ${position} substitute count.`);
    result[position] = value;
  }
  return result;
}
function updateBenchTotal() {
  const fields = POSITIONS.map((p) => Number(dom('bench' + p).value));
  const total = fields.reduce((a, b) => a + b, 0);
  const invalid = fields.some((n) => !Number.isInteger(n) || n < 0) || fields[0] > 4 || fields.slice(1).some((n) => n > 9) || total < 1 || total > 9;
  dom('benchTotal').textContent = `${total} / 9 max`;
  dom('benchTotal').style.color = invalid ? '#ff9b95' : '';
}
function invalidateResult() {
  deleteSession(RESULT_STORAGE_KEY);
  deleteSession(ACTIVE_PLAYER_STORAGE_KEY);
  state.requestId++;
  if (state.busy) setBusy(false);
  state.result = null;
  state.players = [];
  state.activePlayer = -1;
  dom('resultMeta').hidden = true;
  dom('squadResults').hidden = true;
  if (dom('viewAnalyticsLink')) dom('viewAnalyticsLink').hidden = true;
  dom('modelWarnings').hidden = true;
  dom('pitchPlaceholder').hidden = false;
  dom('pitchPlaceholder').innerHTML = '<span class="placeholder-symbol"><svg class="icon"><use href="#ic-ball"/></svg></span><h4>Ready when you are.</h4><p>Select a team, choose your formation, and generate a data-driven lineup.</p>';
  dom('pitchPlayers').replaceChildren();
  dom('pitchCounter').textContent = '0 / 11 players';
  dom('resultBadge').textContent = 'AWAITING SELECTION';
  dom('pitchManager').textContent = dom('managerInput').value.trim() || 'Choose a team & manager';
  dom('pitchTeam').textContent = dom('teamSelect').value || 'Your matchday squad';
  if (dom('playerTableBody')) dom('playerTableBody').innerHTML = '<tr><td colspan="7" class="empty-table">Generate a squad to explore player analytics.</td></tr>';
  if (dom('tableCount')) dom('tableCount').textContent = 'No squad loaded';
  if (dom('detailPanel')) dom('detailPanel').innerHTML = '<div class="detail-empty"><span class="detail-empty-icon"><svg class="icon"><use href="#ic-users"/></svg></span><h3>Scouting profile</h3><p>Select a player from the generated lineup or table to inspect their historical stats.</p></div>';
}
function setBusy(busy) {
  state.busy = busy;
  dom('generateButton').disabled = busy || dom('teamSelect').disabled;
  dom('generateText').textContent = busy ? 'Generating lineup...' : 'Generate Starting XI';
}
async function onGenerate(event) {
  event.preventDefault();
  if (state.busy) return;
  clearFormError();
  const team = dom('teamSelect').value;
  const manager = dom('managerInput').value.trim();
  const formation = dom('formationSelect').value;
  if (!team) return showFormError('Please select a football club.');
  if (!manager) return showFormError('Enter a manager name to continue.');
  if (!FORMATIONS[formation]) return showFormError('Choose a supported formation.');
  let bench;
  try {
    bench = benchValues();
    const benchCount = Object.values(bench).reduce((a, b) => a + b, 0);
    if (benchCount < 1 || benchCount > 9) throw new Error('Choose between 1 and 9 substitute slots.');
  } catch (error) { return showFormError(error.message); }
  invalidateResult();
  const runId = ++state.requestId;
  setBusy(true);
  dom('pitchPlaceholder').innerHTML = '<span class="placeholder-symbol"><svg class="icon"><use href="#ic-spark"/></svg></span><h4>Analyzing the squad...</h4><p>Running both saved Stacking Ensemble models against historical player features.</p>';
  dom('resultBadge').textContent = 'GENERATING';
  try {
    const result = await api('/api/predict', { method: 'POST', body: JSON.stringify({ team, manager, formation, bench }) });
    if (runId !== state.requestId) return;
    if (!Array.isArray(result.starting_xi) || result.starting_xi.length !== 11 || !Array.isArray(result.substitutes)) {
      throw new Error('Unexpected prediction format: expected 11 starters and a substitute list.');
    }
    state.result = result;
    writeSession(RESULT_STORAGE_KEY, result);
    writeSession(SELECTION_STORAGE_KEY, {team, manager, formation, bench});
    state.players = [
      ...result.starting_xi.map((player) => ({ ...player, group: 'Starter' })),
      ...result.substitutes.map((player) => ({ ...player, group: 'Bench' })),
    ];
    renderResult();
    toast(`Created ${result.team} squad: ${result.starting_xi.length} starters and ${result.substitutes.length} substitutes.`);
  } catch (error) {
    if (runId !== state.requestId) return;
    dom('pitchPlaceholder').innerHTML = '<span class="placeholder-symbol"><svg class="icon"><use href="#ic-info"/></svg></span><h4>Selection unavailable</h4><p>Check the details in the selection panel and try again.</p>';
    dom('resultBadge').textContent = 'ACTION NEEDED';
    showFormError(error.message);
    toast('Could not generate the squad.', true);
  } finally {
    if (runId === state.requestId) setBusy(false);
  }
}
function renderResult() {
  const data = state.result;
  dom('pitchPlaceholder').hidden = true;
  dom('pitchTeam').textContent = data.team;
  dom('pitchManager').textContent = data.manager;
  dom('pitchFormation').textContent = data.formation;
  dom('resultBadge').textContent = 'MODEL GENERATED';
  dom('pitchCounter').textContent = `${data.starting_xi.length} / 11 players`;
  dom('resultMeta').hidden = false;
  dom('squadResults').hidden = false;
  if (dom('viewAnalyticsLink')) dom('viewAnalyticsLink').hidden = false;
  dom('resultTeam').textContent = data.team;
  dom('resultFormation').textContent = `${data.formation} · ${data.manager}`;
  dom('xiCount').textContent = `${data.starting_xi.length} SELECTED`;
  dom('subCount').textContent = `${data.substitutes.length} SELECTED`;
  const warnings = [...(Array.isArray(data.warnings) ? data.warnings : [])];
  if (data.disclaimer) warnings.push(data.disclaimer);
  dom('modelWarnings').hidden = warnings.length === 0;
  dom('modelWarnings').textContent = warnings.join('  •  ');
  renderPitch();
  renderCards();
  if (isAnalyticsPage()) { renderTable(); selectPlayer(0, false); }
}
function squadRows() {
  const lineup = state.result.starting_xi;
  const bucket = (pos) => lineup.map((p, idx) => ({ ...p, idx })).filter((p) => p.position === pos);
  const mid = bucket('MID');
  const rows = [bucket('FWD')];
  // 4-2-3-1 splits the five generic MID players visually into two lines.
  // This does NOT infer individual CDM/CAM roles from the model.
  if (state.result.formation === '4-2-3-1') rows.push(mid.slice(0, 3), mid.slice(3));
  else rows.push(mid);
  rows.push(bucket('DEF'), bucket('GK'));
  return rows.filter((row) => row.length);
}
function renderPitch() {
  const root = dom('pitchPlayers');
  root.replaceChildren();
  const rows = squadRows();
  rows.forEach((row, rowIndex) => {
    const y = rows.length === 1 ? 50 : 15 + (rowIndex / (rows.length - 1)) * 70;
    row.forEach((player, i) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'pitch-player' + (player.position === 'GK' ? ' gk' : '');
      button.style.left = (100 * (i + 1) / (row.length + 1)) + '%';
      button.style.top = y + '%';
      button.dataset.playerIndex = String(player.idx);
      button.title = `${player.name} · ${player.position} · Model score: ${percent(player.model_score_percent)}`;
      button.setAttribute('aria-label', button.title + '. View scouting details.');
      button.innerHTML = `<span class="pitch-avatar">${escapeHTML(initials(player.name))}</span><span class="pitch-name">${escapeHTML(player.name)}</span>`;
      button.addEventListener('click', () => selectPlayer(player.idx, true));
      root.append(button);
    });
  });
}
function renderCards() {
  const starters = state.result.starting_xi;
  dom('xiCards').innerHTML = starters.map((p, i) => `
    <button type="button" class="lineup-card" data-player-index="${i}">
      <span class="lineup-card-top"><span class="number-avatar">${escapeHTML(initials(p.name))}</span><span class="role-tag">${escapeHTML(p.position)}</span></span>
      <strong title="${escapeHTML(p.name)}">${escapeHTML(p.name)}</strong><small>Model score · ${percent(p.model_score_percent)}</small>
      <span class="score-strip"><span style="width:${widthPercent(p.model_score_percent)}%"></span></span>
    </button>`).join('');
  dom('benchCards').innerHTML = state.result.substitutes.map((p, i) => `
    <button type="button" class="bench-card" data-player-index="${i + starters.length}">
      <span class="number-avatar">${escapeHTML(initials(p.name))}</span>
      <span class="bench-card-text"><strong>${escapeHTML(p.name)}</strong><small>${escapeHTML(p.position)} · historical form</small></span>
      <span class="bench-card-score">${percent(p.model_score_percent)}</span>
    </button>`).join('');
  ['xiCards', 'benchCards'].forEach((id) => dom(id).querySelectorAll('[data-player-index]').forEach((button) => {
    button.addEventListener('click', () => selectPlayer(Number(button.dataset.playerIndex), true));
  }));
}
function renderTable() {
  const search = dom('playerSearch').value.trim().toLowerCase();
  const position = dom('positionFilter').value;
  const filtered = state.players.map((player, i) => ({ ...player, idx: i })).filter((player) => {
    return (!search || player.name.toLowerCase().includes(search)) && (position === 'ALL' || player.position === position);
  });
  const body = dom('playerTableBody');
  if (!filtered.length) {
    body.innerHTML = '<tr><td colspan="7" class="empty-table">No players match the selected filters.</td></tr>';
  } else {
    body.innerHTML = filtered.map((p) => `
      <tr tabindex="0" role="button" data-player-index="${p.idx}" aria-label="View ${escapeHTML(p.name)} profile">
        <td><span class="table-name"><span class="table-avatar">${escapeHTML(initials(p.name))}</span>${escapeHTML(p.name)}</span></td>
        <td>${escapeHTML(p.position)}</td><td><span class="group-chip ${p.group === 'Bench' ? 'bench' : ''}">${escapeHTML(p.group)}</span></td>
        <td class="mini-score">${percent(p.model_score_percent)}</td><td>${numberText(p.goals)}</td><td>${numberText(p.assists)}</td><td>${numberText(p.average_rating, 2)}</td>
      </tr>`).join('');
  }
  body.querySelectorAll('[data-player-index]').forEach((row) => {
    const select = () => selectPlayer(Number(row.dataset.playerIndex), false);
    row.addEventListener('click', select);
    row.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(); } });
  });
  dom('tableCount').textContent = `Showing ${filtered.length} of ${state.players.length} selected players`;
  markSelected();
}
function selectPlayer(index, scroll) {
  const p = state.players[index];
  if (!p) return;
  state.activePlayer = index;
  writeSession(ACTIVE_PLAYER_STORAGE_KEY, index);
  if (!dom('detailPanel')) {
    if (scroll) window.location.assign('player-analytics.html');
    else markSelected();
    return;
  }
  dom('detailPanel').innerHTML = `
    <div class="detail-top"><span class="detail-avatar">${escapeHTML(initials(p.name))}</span><div><strong>${escapeHTML(p.name)}</strong><small>${escapeHTML(p.position)} · ${escapeHTML(p.group)} · ${escapeHTML(state.result.team)}</small></div></div>
    <div class="detail-stat-grid">
      <div class="detail-stat"><span>Model score (not probability)</span><strong>${percent(p.model_score_percent)}</strong></div>
      <div class="detail-stat"><span>Average rating</span><strong>${numberText(p.average_rating, 2)}</strong></div>
      <div class="detail-stat"><span>Historical goals</span><strong>${numberText(p.goals)}</strong></div>
      <div class="detail-stat"><span>Historical assists</span><strong>${numberText(p.assists)}</strong></div>
    </div>
    <hr class="detail-divider"><div class="detail-subhead">Last three available ratings</div>
    <div class="form-chips">${Array.isArray(p.last_three_ratings) && p.last_three_ratings.length ? p.last_three_ratings.map((v) => `<span>${numberText(v, 1)}</span>`).join('') : '<span>No rating history</span>'}</div>
    <p class="detail-foot">${numberText(p.historical_match_rows)} historical player rows in this club's dataset. Model scores are uncalibrated classification outputs.</p>`;
  markSelected();
  if (scroll && dom('player-analytics')) dom('player-analytics').scrollIntoView({ behavior: 'smooth', block: 'start' });
}
function markSelected() {
  document.querySelectorAll('[data-player-index]').forEach((element) => {
    const active = Number(element.dataset.playerIndex) === state.activePlayer;
    element.classList.toggle(element.classList.contains('pitch-player') ? 'selected' : 'active', active);
    if (element.tagName === 'BUTTON') element.setAttribute('aria-pressed', String(active));
  });
}
async function copyTeamSheet() {
  if (!state.result) return;
  const result = state.result;
  const starterLines = result.starting_xi.map((p) => `- ${p.name} (${p.position}) — model score ${percent(p.model_score_percent)}`);
  const benchLines = result.substitutes.map((p) => `- ${p.name} (${p.position}) — model score ${percent(p.model_score_percent)}`);
  const text = [`PITCHPILOT AI | SQUAD SELECTION`, `Team: ${result.team}`, `Manager: ${result.manager}`, `Formation: ${result.formation}`, '', 'STARTING XI', ...starterLines, '', 'SUBSTITUTES', ...benchLines, '', 'Research prototype: retrospective minutes-played proxy classification; scores are not calibrated probabilities.'].join('\n');
  try {
    await navigator.clipboard.writeText(text);
    toast('Team sheet copied to clipboard.');
  } catch {
    const field = document.createElement('textarea');
    field.value = text;
    field.style.position = 'fixed';field.style.opacity = '0';
    document.body.append(field);field.select();
    const success = document.execCommand('copy');field.remove();
    toast(success ? 'Team sheet copied to clipboard.' : 'Copy is unavailable in this browser.', !success);
  }
}
function setupNavigation() {
  if (!dom('menuToggle')) return;
  const close = () => { dom('sidebar').classList.remove('open'); dom('menuBackdrop').hidden = true; document.body.classList.remove('menu-open'); };
  dom('menuToggle').addEventListener('click', () => { dom('sidebar').classList.add('open'); dom('menuBackdrop').hidden = false; document.body.classList.add('menu-open'); });
  dom('closeSidebar').addEventListener('click', close);
  dom('menuBackdrop').addEventListener('click', close);
  document.addEventListener('keydown', (event) => { if (event.key === 'Escape') close(); });
  document.querySelectorAll('.nav-item').forEach(link => link.addEventListener('click', close));
}

function bindEvents() {
  setupNavigation();
  if (isSquadPage()) {
    dom('selectionForm').addEventListener('submit', onGenerate);
    dom('teamSelect').addEventListener('change', () => {
      invalidateResult(); clearFormError(); updateAvailablePlayers(); updateManagerSuggestions();
    });
    dom('managerInput').addEventListener('input', () => {
      state.managerEdited = true;
      dom('managerInput').dataset.autofilled = 'false';
      invalidateResult(); clearFormError(); renderManagerSuggestions();
    });
    dom('managerInput').addEventListener('focus', renderManagerSuggestions);
    dom('managerInput').addEventListener('keydown', onManagerKeydown);
    document.addEventListener('pointerdown', event => {
      if (!dom('managerCombobox').contains(event.target)) closeManagerSuggestions();
    });
    dom('formationSelect').addEventListener('change', () => { updateFormationGuide(); invalidateResult(); clearFormError(); });
    POSITIONS.forEach(p => dom('bench' + p).addEventListener('input', () => { updateBenchTotal(); invalidateResult(); clearFormError(); }));
    dom('copyButton').addEventListener('click', copyTeamSheet);
    dom('printButton').addEventListener('click', () => { if (state.result) window.print(); });
    updateBenchTotal();
  }
  if (isAnalyticsPage()) {
    dom('playerSearch').addEventListener('input', () => { if (state.result) renderTable(); });
    dom('positionFilter').addEventListener('change', () => { if (state.result) renderTable(); });
  }
}

bindEvents();
initialize();
