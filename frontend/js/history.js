/**
 * History page — browse and filter interview sessions.
 */
class HistoryApp {
    constructor() {
        this.filterInput = document.getElementById('candidate-filter');
        this.btnSearch = document.getElementById('btn-search');
        this.btnClear = document.getElementById('btn-clear');
        this.sessionsTbody = document.getElementById('sessions-tbody');
        this.paginationControls = document.getElementById('pagination-controls');
        this.sessionDetail = document.getElementById('session-detail');
        this.detailTitle = document.getElementById('detail-title');
        this.chunksContainer = document.getElementById('chunks-container');
        this.btnCloseDetail = document.getElementById('btn-close-detail');
        this.btnRenameDetail = document.getElementById('btn-rename-detail');

        this.limit = 20;
        this.offset = 0;
        this.total = 0;
        this.currentCandidate = '';
        this.selectedRow = null;
        this.currentDetailSessionId = null;
        this.currentDetailCandidateName = null;
        this.currentDetailStartedAt = null;

        const { protocol, hostname } = window.location;
        this.apiBase = `${protocol}//${hostname}:5555`;

        this._bindEvents();
        // Defer data loading until after the page has painted so the
        // shell is visible immediately rather than stalling on the DB query.
        requestAnimationFrame(() => this.fetchSessions());
    }

    _bindEvents() {
        this.btnSearch.addEventListener('click', () => this._search());
        this.btnClear.addEventListener('click', () => this._clear());
        this.filterInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') this._search();
        });
        this.btnCloseDetail.addEventListener('click', () => {
            this.sessionDetail.classList.add('hidden');
            if (this.selectedRow) {
                this.selectedRow.classList.remove('selected');
                this.selectedRow = null;
            }
            this.currentDetailSessionId = null;
        });
        this.btnRenameDetail.addEventListener('click', () => this._startDetailRename());
    }

    _search() {
        this.currentCandidate = this.filterInput.value.trim();
        this.offset = 0;
        this.fetchSessions();
    }

    _clear() {
        this.filterInput.value = '';
        this.currentCandidate = '';
        this.offset = 0;
        this.fetchSessions();
    }

    async fetchSessions() {
        const params = new URLSearchParams({
            limit: this.limit,
            offset: this.offset,
        });
        if (this.currentCandidate) {
            params.set('candidate', this.currentCandidate);
        }

        try {
            const resp = await fetch(`${this.apiBase}/api/history/sessions?${params}`);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            this.total = data.total || 0;
            this._renderTable(data.sessions || []);
            this._renderPagination();
        } catch (err) {
            this.sessionsTbody.innerHTML = `<tr><td colspan="6" class="no-sessions">Failed to load sessions: ${err.message}</td></tr>`;
        }
    }

    _renderTable(sessions) {
        if (sessions.length === 0) {
            this.sessionsTbody.innerHTML = '<tr><td colspan="6" class="no-sessions">No sessions found.</td></tr>';
            return;
        }
        this.sessionsTbody.innerHTML = '';
        for (const session of sessions) {
            const tr = document.createElement('tr');
            tr.dataset.sessionId = session.session_id;

            const date = session.started_at ? new Date(session.started_at) : null;
            const dateStr = date ? date.toLocaleString() : '-';
            const duration = this._formatDuration(session.started_at, session.ended_at);
            const statusBadge = `<span class="status-badge ${session.status || 'active'}">${session.status || 'active'}</span>`;

            tr.innerHTML = `
                <td>${dateStr}</td>
                <td class="candidate-cell">${this._esc(session.candidate_name)}</td>
                <td>${duration}</td>
                <td>${session.chunk_count != null ? session.chunk_count : '-'}</td>
                <td>${statusBadge}</td>
                <td>
                    <div class="actions-cell">
                        <button class="btn btn-secondary btn-view" style="font-size:12px;padding:4px 10px;" type="button">View</button>
                        <button class="btn btn-secondary btn-rename" style="font-size:12px;padding:4px 10px;" type="button">Rename</button>
                        <button class="btn btn-danger btn-delete" style="font-size:12px;padding:4px 10px;" type="button">Delete</button>
                    </div>
                </td>
            `;

            tr.querySelector('.btn-view').addEventListener('click', () => {
                this._selectRow(tr);
                this.loadSessionDetail(session.session_id, session.candidate_name, session.started_at);
            });
            tr.querySelector('.btn-rename').addEventListener('click', () => {
                this._startInlineRename(tr, session);
            });
            tr.querySelector('.btn-delete').addEventListener('click', () => {
                this._confirmDeleteSession(session.session_id, session.candidate_name);
            });

            this.sessionsTbody.appendChild(tr);
        }
    }

    _selectRow(tr) {
        if (this.selectedRow) this.selectedRow.classList.remove('selected');
        this.selectedRow = tr;
        tr.classList.add('selected');
    }

    _renderPagination() {
        const totalPages = Math.ceil(this.total / this.limit) || 1;
        const currentPage = Math.floor(this.offset / this.limit) + 1;

        this.paginationControls.innerHTML = '';

        const prevBtn = document.createElement('button');
        prevBtn.className = 'btn btn-secondary';
        prevBtn.style.fontSize = '12px';
        prevBtn.style.padding = '4px 12px';
        prevBtn.textContent = 'Prev';
        prevBtn.disabled = this.offset === 0;
        prevBtn.addEventListener('click', () => {
            this.offset = Math.max(0, this.offset - this.limit);
            this.fetchSessions();
        });

        const pageInfo = document.createElement('span');
        pageInfo.textContent = `Page ${currentPage} of ${totalPages} (${this.total} total)`;

        const nextBtn = document.createElement('button');
        nextBtn.className = 'btn btn-secondary';
        nextBtn.style.fontSize = '12px';
        nextBtn.style.padding = '4px 12px';
        nextBtn.textContent = 'Next';
        nextBtn.disabled = this.offset + this.limit >= this.total;
        nextBtn.addEventListener('click', () => {
            this.offset += this.limit;
            this.fetchSessions();
        });

        this.paginationControls.appendChild(prevBtn);
        this.paginationControls.appendChild(pageInfo);
        this.paginationControls.appendChild(nextBtn);
    }

    async loadSessionDetail(sessionId, candidateName, startedAt) {
        this.currentDetailSessionId = sessionId;
        this.currentDetailCandidateName = candidateName;
        this.currentDetailStartedAt = startedAt;
        const date = startedAt ? new Date(startedAt).toLocaleString() : '';
        this.detailTitle.textContent = `${this._esc(candidateName)} — ${date}`;
        this.chunksContainer.innerHTML = '<div style="padding:16px;color:#7b8fa6;">Loading chunks...</div>';
        this.sessionDetail.classList.remove('hidden');
        this.sessionDetail.scrollIntoView({ behavior: 'smooth', block: 'start' });

        try {
            const resp = await fetch(`${this.apiBase}/api/history/sessions/${encodeURIComponent(sessionId)}`);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            this._renderChunks(data.chunks || []);
        } catch (err) {
            this.chunksContainer.innerHTML = `<div style="padding:16px;color:#c0392b;">Failed to load session: ${err.message}</div>`;
        }
    }

    _renderChunks(chunks) {
        if (chunks.length === 0) {
            this.chunksContainer.innerHTML = '<div style="padding:16px;color:#7b8fa6;">No chunks recorded for this session.</div>';
            return;
        }
        this.chunksContainer.innerHTML = '';
        for (const chunk of chunks) {
            this.chunksContainer.appendChild(this._buildChunkAccordion(chunk));
        }
    }

    _buildChunkAccordion(chunk) {
        const wrapper = document.createElement('div');
        wrapper.className = 'chunk-accordion';

        const processedAt = chunk.processed_at ? new Date(chunk.processed_at).toLocaleTimeString() : '-';
        const statusBadge = chunk.processing_status === 'failed'
            ? ' <span class="status-badge failed">failed</span>'
            : '';

        const header = document.createElement('div');
        header.className = 'chunk-header';
        header.innerHTML = `
            <span>Chunk ${chunk.chunk_index} &nbsp;&middot;&nbsp; ${processedAt}${statusBadge}</span>
            <span class="chunk-arrow">&#9654;</span>
        `;

        const body = document.createElement('div');
        body.className = 'chunk-body';

        if (chunk.processing_status === 'failed' || (chunk.model_results && chunk.model_results.error)) {
            const errCard = document.createElement('div');
            errCard.className = 'model-card error-card';
            errCard.innerHTML = `<div class="model-card-title">Error</div><div class="model-card-content">${this._esc(chunk.model_results && chunk.model_results.error ? chunk.model_results.error : 'Processing failed')}</div>`;
            body.appendChild(errCard);
        } else {
            const grid = document.createElement('div');
            grid.className = 'model-cards-grid';
            const modelResults = chunk.model_results || {};
            for (const [modelName, results] of Object.entries(modelResults)) {
                grid.appendChild(this._buildModelCard(modelName, results));
            }
            if (grid.children.length === 0) {
                grid.innerHTML = '<div style="color:#7b8fa6;font-size:13px;">No model results available.</div>';
            }
            body.appendChild(grid);
        }

        header.addEventListener('click', () => {
            const isOpen = header.classList.toggle('open');
            body.classList.toggle('open', isOpen);
        });

        wrapper.appendChild(header);
        wrapper.appendChild(body);
        return wrapper;
    }

    _buildModelCard(modelName, results) {
        const card = document.createElement('div');
        card.className = 'model-card';

        const title = document.createElement('div');
        title.className = 'model-card-title';
        title.textContent = this._modelLabel(modelName);

        const content = document.createElement('div');
        content.className = 'model-card-content';
        content.innerHTML = this._renderModelContent(modelName, results);

        card.appendChild(title);
        card.appendChild(content);
        return card;
    }

    _modelLabel(modelName) {
        const labels = {
            whisper: 'Whisper (Transcript)',
            mediapipe: 'MediaPipe (Vision)',
            vocaltone: 'Vocal Tone',
            clifton_fusion: 'Clifton Strengths',
            interviewer_ollama: 'Interviewer Questions',
        };
        return labels[modelName] || modelName;
    }

    _renderModelContent(modelName, results) {
        if (!results || typeof results !== 'object') return this._esc(String(results));

        if (results.error) {
            return `<span style="color:#c0392b;">${this._esc(results.error)}</span>`;
        }

        switch (modelName) {
            case 'whisper':
                return `
                    <div class="transcript-text">${this._esc(results.transcript || '-')}</div>
                    ${results.confidence != null ? `<div>Confidence: ${(results.confidence * 100).toFixed(0)}%</div>` : ''}
                `;
            case 'mediapipe':
                return `
                    ${results.dominant_emotion ? `<div>Emotion: <strong>${this._esc(results.dominant_emotion)}</strong></div>` : ''}
                    ${results.engagement_score != null ? `<div>Engagement: <strong>${(results.engagement_score * 100).toFixed(0)}%</strong></div>` : ''}
                    ${results.posture_score != null ? `<div>Posture: <strong>${(results.posture_score * 100).toFixed(0)}%</strong></div>` : ''}
                `;
            case 'vocaltone':
                return `
                    ${results.emotion ? `<div>Emotion: <strong>${this._esc(results.emotion)}</strong></div>` : ''}
                    ${results.confidence != null ? `<div>Confidence: ${(results.confidence * 100).toFixed(0)}%</div>` : ''}
                    ${results.pitch_mean != null ? `<div>Pitch: ${results.pitch_mean.toFixed(0)} Hz</div>` : ''}
                    ${results.tempo != null ? `<div>Tempo: ${results.tempo.toFixed(0)} BPM</div>` : ''}
                `;
            case 'clifton_fusion':
                return `
                    ${results.predicted_domain ? `<div>Domain: <strong>${this._esc(results.predicted_domain)}</strong></div>` : ''}
                    ${results.confidence != null ? `<div>Confidence: ${(results.confidence * 100).toFixed(0)}%</div>` : ''}
                `;
            case 'interviewer_ollama':
                return `<div class="transcript-text">${this._esc(results.questions_text || '-')}</div>`;
            default:
                // Generic fallback for any future model — renders top-level string/number fields
                return Object.entries(results)
                    .filter(([, v]) => typeof v === 'string' || typeof v === 'number')
                    .slice(0, 6)
                    .map(([k, v]) => `<div>${this._esc(k)}: <strong>${this._esc(String(v))}</strong></div>`)
                    .join('') || '<div>No displayable data</div>';
        }
    }

    _startInlineRename(tr, session) {
        const cell = tr.querySelector('.candidate-cell');
        const orig = session.candidate_name;
        cell.innerHTML = `
            <input class="rename-input" type="text" value="${this._esc(orig)}" maxlength="255">
            <button class="btn btn-primary" style="font-size:11px;padding:3px 8px;" type="button">Save</button>
            <button class="btn btn-secondary" style="font-size:11px;padding:3px 8px;" type="button">Cancel</button>
        `;
        const input = cell.querySelector('.rename-input');
        const [saveBtn, cancelBtn] = cell.querySelectorAll('button');
        input.focus();
        input.select();
        saveBtn.addEventListener('click', () => this._saveRename(tr, session, input.value.trim()));
        cancelBtn.addEventListener('click', () => { cell.textContent = orig; });
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') this._saveRename(tr, session, input.value.trim());
            if (e.key === 'Escape') { cell.textContent = orig; }
        });
    }

    async _saveRename(tr, session, newName) {
        if (!newName) return;
        try {
            const resp = await fetch(
                `${this.apiBase}/api/history/sessions/${encodeURIComponent(session.session_id)}`,
                {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ candidate_name: newName }),
                }
            );
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            session.candidate_name = newName;
            tr.querySelector('.candidate-cell').textContent = newName;
            if (this.currentDetailSessionId === session.session_id) {
                this.currentDetailCandidateName = newName;
                const date = session.started_at ? new Date(session.started_at).toLocaleString() : '';
                this.detailTitle.textContent = `${newName} — ${date}`;
            }
        } catch (err) {
            alert(`Rename failed: ${err.message}`);
            tr.querySelector('.candidate-cell').textContent = session.candidate_name;
        }
    }

    async _confirmDeleteSession(sessionId, candidateName) {
        if (!window.confirm(`Delete session for "${candidateName}"?\n\nThis will permanently remove all chunk results and cannot be undone.`)) return;
        try {
            const resp = await fetch(
                `${this.apiBase}/api/history/sessions/${encodeURIComponent(sessionId)}`,
                { method: 'DELETE' }
            );
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            if (this.currentDetailSessionId === sessionId) {
                this.sessionDetail.classList.add('hidden');
                this.currentDetailSessionId = null;
                if (this.selectedRow) {
                    this.selectedRow.classList.remove('selected');
                    this.selectedRow = null;
                }
            }
            if (this.offset > 0 && this.total - 1 <= this.offset) {
                this.offset = Math.max(0, this.offset - this.limit);
            }
            this.fetchSessions();
        } catch (err) {
            alert(`Delete failed: ${err.message}`);
        }
    }

    _startDetailRename() {
        if (!this.currentDetailSessionId) return;
        const orig = this.currentDetailCandidateName || '';
        this.detailTitle.innerHTML = `
            <input class="rename-input" type="text" value="${this._esc(orig)}" maxlength="255">
            <button class="btn btn-primary" style="font-size:11px;padding:3px 8px;" type="button">Save</button>
            <button class="btn btn-secondary" style="font-size:11px;padding:3px 8px;" type="button">Cancel</button>
        `;
        const input = this.detailTitle.querySelector('.rename-input');
        const [saveBtn, cancelBtn] = this.detailTitle.querySelectorAll('button');
        input.focus();
        input.select();
        const restore = () => {
            const date = this.currentDetailStartedAt ? new Date(this.currentDetailStartedAt).toLocaleString() : '';
            this.detailTitle.textContent = `${orig} — ${date}`;
        };
        saveBtn.addEventListener('click', () => this._saveDetailRename(input.value.trim(), orig));
        cancelBtn.addEventListener('click', restore);
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') this._saveDetailRename(input.value.trim(), orig);
            if (e.key === 'Escape') restore();
        });
    }

    async _saveDetailRename(newName, orig) {
        if (!newName || !this.currentDetailSessionId) return;
        try {
            const resp = await fetch(
                `${this.apiBase}/api/history/sessions/${encodeURIComponent(this.currentDetailSessionId)}`,
                {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ candidate_name: newName }),
                }
            );
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            this.currentDetailCandidateName = newName;
            const date = this.currentDetailStartedAt ? new Date(this.currentDetailStartedAt).toLocaleString() : '';
            this.detailTitle.textContent = `${newName} — ${date}`;
            if (this.selectedRow) {
                const cell = this.selectedRow.querySelector('.candidate-cell');
                if (cell && !cell.querySelector('input')) cell.textContent = newName;
            }
        } catch (err) {
            alert(`Rename failed: ${err.message}`);
            const date = this.currentDetailStartedAt ? new Date(this.currentDetailStartedAt).toLocaleString() : '';
            this.detailTitle.textContent = `${orig} — ${date}`;
        }
    }

    _formatDuration(startedAt, endedAt) {
        if (!startedAt) return '-';
        if (!endedAt) return 'In progress';
        const ms = new Date(endedAt) - new Date(startedAt);
        if (isNaN(ms) || ms < 0) return '-';
        const totalSec = Math.floor(ms / 1000);
        const mins = Math.floor(totalSec / 60);
        const secs = totalSec % 60;
        return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
    }

    _esc(str) {
        if (str == null) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.historyApp = new HistoryApp();
});
