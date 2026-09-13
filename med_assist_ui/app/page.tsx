'use client'

import { useState } from 'react'
import { ArrowUpRight, Check, ChevronRight, ExternalLink, FlaskConical, Search, Sparkles } from 'lucide-react'

type EvidenceItem = { pmid?: string; title?: string; snippet?: string; chunk_text?: string; url?: string; score?: number | string; combined_score?: number; source_dataset?: string }
type AskResult = { answer: string; rationale?: string; confidence_level: string; evidence_used: EvidenceItem[]; used_web_fallback?: boolean }

const examples = [
  'Does metformin improve survival in pancreatic cancer?',
  'Is CRISPR base editing safe in vivo?',
  'Does sleep deprivation alter microglial activation?',
]

function Pipeline({ loading, result }: { loading: boolean; result: AskResult | null }) {
  const stages = [
    ['01', 'Retrieved', 'Searching the corpus'],
    ['02', 'Confidence Checked', 'Ranking high-signal matches'],
    ['03', 'Verified', 'Cross-referencing citations'],
    ['04', 'Answer', 'Synthesis ready'],
  ]
  const completedCount = result ? 4 : loading ? 2 : 0
  return <aside className="pipeline" aria-label="Processing pipeline">
    <div className="eyebrow">CHAIN OF CUSTODY</div>
    <div className="trace">
      {stages.map(([number, title, detail], index) => {
        const isComplete = index < completedCount
        const isActive = loading && index === completedCount
        return <div className={`trace-step ${isComplete ? 'complete' : ''} ${isActive ? 'active' : ''}`} key={title}>
        <div className="trace-node">{isComplete ? <Check size={12} strokeWidth={3} /> : <span>{number}</span>}</div>
        {index < stages.length - 1 && <div className="trace-line" />}
        <div className="trace-copy"><div className="trace-title">{title}</div><div className="trace-time">{isActive ? 'IN PROGRESS' : isComplete ? 'COMPLETE' : 'STANDBY'}</div><div className="trace-detail">{detail}</div></div>
      </div>})}
    </div>
  </aside>
}

export default function Page() {
  const [question, setQuestion] = useState(examples[0])
  const [result, setResult] = useState<AskResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const runAnalysis = async () => {
    if (!question.trim() || loading) return
    setLoading(true)
    setError('')
    try {
      const apiBase = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'
      const response = await fetch(`${apiBase}/api/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question.trim() }),
      })
      if (!response.ok) {
        const payload = await response.json()
        throw new Error(payload.detail ?? 'The analysis could not be completed.')
      }
      setResult(await response.json())
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'The analysis could not be completed.')
    } finally {
      setLoading(false)
    }
  }

  const evidenceToShow = result?.evidence_used ?? []

  return <main className="medassist-shell">
    <header className="topbar">
      <div className="brand-mark"><FlaskConical size={18} /><span>MA / 01</span></div>
      <div className="status"><span className="status-dot" /> SYSTEM ONLINE <span className="status-separator">/</span> CORPUS: PUBMED <span className="status-separator">/</span> v2.4.1</div>
      <button className="about-button">METHOD <ArrowUpRight size={14} /></button>
    </header>

    <div className="content-wrap">
      <section className="hero">
        <div className="hero-kicker"><span className="rule" /> BIOMEDICAL EVIDENCE ENGINE <span className="rule" /></div>
        <h1>MedAssist</h1>
        <p className="subtitle">Evidence-Grounded Biomedical Reasoning</p>
        <p className="hero-copy">Ask a clinical question. Receive a traceable synthesis from the biomedical literature — with uncertainty made explicit.</p>
      </section>

      <section className="query-section" aria-label="Research question">
        <div className="section-label"><span>01</span> INPUT / RESEARCH QUESTION</div>
        <div className="query-box">
          <Search size={19} className="query-icon" aria-hidden="true" />
          <input aria-label="Research question" value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter' && !e.nativeEvent.isComposing && e.keyCode !== 229) runAnalysis() }} />
          <button className="run-button" onClick={runAnalysis} disabled={loading}>{loading ? 'ANALYZING...' : 'RUN ANALYSIS'} {!loading && <ChevronRight size={16} />}</button>
        </div>
        <div className="examples"><span>TRY:</span>{examples.map((item, index) => <button key={item} onClick={() => setQuestion(item)} className={question === item ? 'active' : ''}>{String(index + 1).padStart(2, '0')} {item}</button>)}</div>
        {loading && <p role="status" className="loading-message"><span className="spinner" /> Running the evidence pipeline. This may take 20-30 seconds.</p>}
        {error && <p role="alert" className="error-message">{error}</p>}
      </section>

      <div className="workspace">
        <div className="results-column">
          <div className="section-label"><span>02</span> SYNTHESIS / CURRENT FINDINGS</div>
          <article className="verdict-panel">
            <div className="verdict-meta"><span>MODEL VERDICT</span><span>QUERY ID: MA-240318-7F</span></div>
            <div className="verdict-row"><div className={`verdict-stamp ${result ? 'resolved' : 'pending'}`}>{loading ? 'ANALYZING' : result ? result.answer.replaceAll('_', ' ').toUpperCase() : 'AWAITING QUERY'}</div><div className="confidence-readout"><div className="confidence-label">CONFIDENCE <strong>{result?.confidence_level?.toUpperCase() ?? '—'}</strong></div><div className="waveform" aria-label="Confidence waveform"><span /></div></div></div>
            <div className="rationale-block"><span className="rationale-label">GROUNDED SYNTHESIS</span><p className="answer-copy">{result?.rationale ?? (loading ? 'The orchestrator is retrieving and verifying biomedical evidence.' : 'Submit a research question to receive an evidence-grounded synthesis from the biomedical literature.')}</p></div>
            <div className="answer-foot"><Sparkles size={14} /> {result?.used_web_fallback ? 'LOCAL CORPUS + WEB FALLBACK' : 'UNCERTAINTY PRESERVED'} <span>{result ? 'JUST NOW' : 'READY'}</span></div>
          </article>

          <div className="evidence-heading"><div className="section-label"><span>03</span> EVIDENCE / PRIMARY SOURCES</div><span className="source-count">{String(evidenceToShow.length).padStart(2, '0')} RECORDS</span></div>
          <div className="evidence-list">{evidenceToShow.length ? evidenceToShow.map((item, index) => { const sourceUrl = item.pmid ? `https://pubmed.ncbi.nlm.nih.gov/${item.pmid.replace('PMID ', '')}/` : item.url; const excerpt = item.snippet ?? item.chunk_text; return <article className="specimen" key={`${item.pmid ?? item.url ?? 'record'}-${index}`}><div className="specimen-top"><a href={sourceUrl ?? '#'} target="_blank" rel="noreferrer">{item.pmid ? `PMID ${item.pmid.replace('PMID ', '')}` : `RECORD ${String(index + 1).padStart(2, '0')}`} {sourceUrl && <ExternalLink size={12} />}</a><span>{item.source_dataset ?? 'BIOMEDICAL CORPUS'}</span></div><h3>{item.title ?? `Evidence record ${String(index + 1).padStart(2, '0')}`}</h3><p>{excerpt ?? 'No excerpt was returned for this record.'}</p><div className="specimen-bottom"><span>RELEVANCE SCORE</span><strong>{item.score ?? item.combined_score?.toFixed(3) ?? '—'}</strong><div className="score-ticks"><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /></div></div></article> }) : <p className="answer-copy">Evidence records will appear here after analysis.</p>}</div>
        </div>
        <Pipeline loading={loading} result={result} />
      </div>
    </div>
    <footer><span>MEDASSIST / RESEARCH PREVIEW</span><span>FOR RESEARCH USE ONLY · NOT MEDICAL ADVICE</span></footer>
  </main>
}
