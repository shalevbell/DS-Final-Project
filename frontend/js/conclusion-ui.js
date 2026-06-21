/**
 * Shared session conclusion rendering for live interview and history pages.
 */
const ConclusionUI = {
    escapeHtml(value) {
        return String(value ?? '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    },

    renderQuestionList(items, emptyText) {
        if (!items || !items.length) {
            return `<p class="conclusion-empty">${this.escapeHtml(emptyText)}</p>`;
        }
        return `<ul class="conclusion-list">${items.map((item) => {
            const chunkLabel = item.chunk !== undefined && item.chunk !== null
                ? `<small>Chunk ${item.chunk}</small>`
                : '';
            return `<li>${this.escapeHtml(item.text)}${chunkLabel}</li>`;
        }).join('')}</ul>`;
    },

    render(conclusion) {
        if (!conclusion) return '';

        const warmup = (conclusion.questions && conclusion.questions.resume_warmup) || [];
        const followups = (conclusion.questions && conclusion.questions.ai_followup) || [];
        const analytics = conclusion.analytics_summary || {};
        const highlights = conclusion.highlights || [];
        const nextSteps = conclusion.recommended_next_steps || [];

        const resumeBlock = conclusion.resume
            ? `<a class="conclusion-resume-link" href="${this.escapeHtml(conclusion.resume.download_url)}" target="_blank" rel="noopener">Download resume: ${this.escapeHtml(conclusion.resume.filename)}</a>`
            : '<p class="conclusion-empty">No resume was attached for this session.</p>';

        return `
            <section class="conclusion-section">
                <h3>Executive Summary</h3>
                <p class="conclusion-summary">${this.escapeHtml(conclusion.executive_summary || '')}</p>
            </section>

            <section class="conclusion-section">
                <h3>Session Overview</h3>
                <div class="conclusion-meta-grid">
                    <div class="conclusion-meta-item"><span>Candidate</span><strong>${this.escapeHtml(conclusion.candidate_name || '-')}</strong></div>
                    <div class="conclusion-meta-item"><span>Target Role</span><strong>${this.escapeHtml(conclusion.target_role || 'Not specified')}</strong></div>
                    <div class="conclusion-meta-item"><span>Duration</span><strong>${this.escapeHtml(conclusion.duration || '-')}</strong></div>
                    <div class="conclusion-meta-item"><span>Chunks Analyzed</span><strong>${conclusion.chunks_processed ?? 0}</strong></div>
                    <div class="conclusion-meta-item"><span>Total Questions</span><strong>${conclusion.questions?.total_count ?? 0}</strong></div>
                    <div class="conclusion-meta-item"><span>Primary Clifton Domain</span><strong>${this.escapeHtml(analytics.primary_clifton_domain || 'N/A')}</strong></div>
                </div>
            </section>

            <section class="conclusion-section">
                <h3>Resume File</h3>
                ${resumeBlock}
            </section>

            <section class="conclusion-section">
                <h3>Resume-Based Warm-Up Questions</h3>
                ${this.renderQuestionList(warmup, 'No resume-based warm-up questions were generated.')}
            </section>

            <section class="conclusion-section">
                <h3>AI Follow-Up Questions</h3>
                ${this.renderQuestionList(followups, 'No AI follow-up questions were generated during this session.')}
            </section>

            <section class="conclusion-section">
                <h3>Behavior & Voice Highlights</h3>
                <ul class="conclusion-list">
                    ${analytics.avg_stress_level != null ? `<li>Average stress level: <strong>${analytics.avg_stress_level}%</strong></li>` : ''}
                    ${analytics.avg_engagement_pct != null ? `<li>Average engagement: <strong>${analytics.avg_engagement_pct}%</strong></li>` : ''}
                    ${analytics.eye_contact_consistency_pct != null ? `<li>Eye contact consistency: <strong>${analytics.eye_contact_consistency_pct}%</strong></li>` : ''}
                    ${analytics.posture_stability_pct != null ? `<li>Posture stability: <strong>${analytics.posture_stability_pct}%</strong></li>` : ''}
                    ${(analytics.dominant_emotions || []).length ? `<li>Dominant visual emotions: <strong>${analytics.dominant_emotions.join(', ')}</strong></li>` : ''}
                    ${(analytics.dominant_voice_emotions || []).length ? `<li>Dominant voice emotions: <strong>${analytics.dominant_voice_emotions.join(', ')}</strong></li>` : ''}
                </ul>
            </section>

            <section class="conclusion-section">
                <h3>Key Highlights</h3>
                <ul class="conclusion-list">${highlights.map((item) => `<li>${this.escapeHtml(item)}</li>`).join('') || '<li>No highlights available.</li>'}</ul>
            </section>

            <section class="conclusion-section">
                <h3>Recommended Next Steps</h3>
                <ul class="conclusion-list">${nextSteps.map((item) => `<li>${this.escapeHtml(item)}</li>`).join('') || '<li>Review the session recording and notes with the hiring team.</li>'}</ul>
            </section>

            ${conclusion.interview_requirements ? `
            <section class="conclusion-section">
                <h3>Interviewer Requirements</h3>
                <p class="conclusion-summary">${this.escapeHtml(conclusion.interview_requirements)}</p>
            </section>` : ''}
        `;
    },

    setTitle(titleEl, conclusion) {
        if (!titleEl) return;
        titleEl.textContent = `Interview Conclusion — ${conclusion.candidate_name || 'Candidate'}`;
    },
};
