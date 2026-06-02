export const meta = {
  name: 'best-practices-audit',
  description: 'Audit the repo against each best-practices sub-skill, document findings as md, and write a scoped ticket per finding',
  phases: [
    { title: 'Audit', detail: 'one agent per best-practices sub-skill, runs in parallel, writes findings/<domain>.md' },
    { title: 'Ticket', detail: 'one agent per finding investigates and writes tickets/<ID>.md' },
  ],
}

// --- configuration ----------------------------------------------------------
// args lets the workflow be re-pointed at another repo / skill cache without editing.
const REPO = (args && args.repo) || '/workspace/LLM-Home-Controller'
const SKILL_BASE = (args && args.skillBase)
  || '/home/vscode/.claude/plugins/cache/allada-skills/best-practices/0.3.0/skills'
const OUT = `${REPO}/.best-practices-audit`

// The "sub ones": the domain best-practices skills that apply to this repo.
// meta-best-practices is intentionally excluded — it governs authoring the rule
// library itself, not auditing a codebase.
const DOMAINS = [
  {
    key: 'containers', abbr: 'CON', skill: 'containers-best-practices',
    families: 'DOCKER-* and the container-side UV-* family',
    surface: '.devcontainer/Dockerfile, .devcontainer/devcontainer.json, '
      + '.devcontainer/ha-core/{Dockerfile,devcontainer.json,docker-compose.yml}, '
      + '.dockerignore (check existence AND coverage), .gitignore (check .env is ignored)',
  },
  {
    key: 'python', abbr: 'PY', skill: 'python-best-practices',
    families: 'the PY-* family',
    surface: 'pyproject.toml (everything OUTSIDE [tool.uv]/[dependency-groups]), '
      + 'custom_components/llm_home_controller/**/*.py, tests/**/*.py, conftest.py, '
      + '.python-version, the [tool.ruff] config, .github/workflows/*.yml',
  },
  {
    key: 'uv', abbr: 'UV', skill: 'uv-best-practices',
    families: 'the UVP-* family',
    surface: 'pyproject.toml ([tool.uv], [dependency-groups], [project] requires-python), '
      + 'uv.lock, .python-version, .github/workflows/*.yml (uv usage in CI)',
  },
]

// --- schemas ----------------------------------------------------------------
const FINDING = {
  type: 'object',
  required: ['ruleId', 'severity', 'title', 'location', 'recommendation'],
  properties: {
    ruleId: { type: 'string', description: 'Stable rule ID, e.g. DOCKER-005 / PY-012 / UVP-003' },
    severity: { type: 'string', enum: ['high', 'medium', 'low'] },
    title: { type: 'string' },
    location: { type: 'string', description: 'file:line or file the issue lives in' },
    evidence: { type: 'string', description: 'short snippet / observation proving the finding is real' },
    recommendation: { type: 'string' },
  },
}
const AUDIT_SCHEMA = {
  type: 'object',
  required: ['domain', 'findings'],
  properties: {
    domain: { type: 'string' },
    findingsFile: { type: 'string' },
    findings: { type: 'array', items: FINDING },
  },
}
const TICKET_SCHEMA = {
  type: 'object',
  required: ['ticketId', 'file', 'title', 'severity'],
  properties: {
    ticketId: { type: 'string' },
    file: { type: 'string' },
    title: { type: 'string' },
    severity: { type: 'string', enum: ['high', 'medium', 'low'] },
  },
}

// --- prompts ----------------------------------------------------------------
function auditPrompt(d) {
  return `You are auditing the repository at ${REPO} against the "${d.skill}" best-practices skill.
This skill owns ${d.families}.

STEP 1 — Load the rules. Read these in full:
  - ${SKILL_BASE}/${d.skill}/SKILL.md   (the rule index — every rule ID + one-line summary)
  - every file under ${SKILL_BASE}/${d.skill}/references/   (the full What / Why / How / When-NOT-to-apply for each rule)

STEP 2 — Audit the repo. The primary surface for this domain:
  ${d.surface}
Read the ACTUAL file contents with Read / Grep / Bash — never guess. For every rule that is
violated, only partially satisfied, or where the repo could be meaningfully hardened, record one
finding citing the exact rule ID. Rules:
  - Confirm each finding against the real file. Do not report anything you have not seen in the file.
  - Honor each rule's "When NOT to apply" — skip rules that genuinely do not apply to this repo.
  - No speculative or false-positive findings. Quality over quantity.
Severity: high = security / correctness / breaks reproducibility or the build; medium = maintainability,
config drift, missing hardening; low = polish / nice-to-have.

STEP 3 — Document. Write a markdown report to ${OUT}/findings/${d.key}.md containing:
  - An H1 title and a one-paragraph summary (files reviewed, # findings by severity).
  - A summary table: Rule ID | Severity | Title | Location.
  - One H2 section per finding (ordered high -> low severity): the rule ID, severity, the offending
    file:line with a short fenced evidence snippet, 1-2 lines on why it matters, and the concrete
    recommended change.
If you find ZERO real issues, still write the file stating the domain is clean and what you checked.

STEP 4 — Return the structured findings via the output tool (domain, findingsFile path, findings[]).`
}

function ticketPrompt(d, f, ticketId) {
  return `You are writing ONE well-scoped engineering ticket for a single best-practices audit finding,
for the repository at ${REPO}. The ticket must be self-contained: an engineer with zero prior context
should be able to pick it up and fix it.

Ticket ID:        ${ticketId}
Source rule:      ${f.ruleId}   (skill: ${d.skill})
Finding title:    ${f.title}
Severity:         ${f.severity}
Reported at:      ${f.location}
Evidence:         ${f.evidence || '(see file)'}
Recommendation:   ${f.recommendation}

STEP 1 — Investigate to confirm and scope. Open the referenced file(s) and the surrounding code/config
in the repo. Capture PRECISE current context: exact path(s), line numbers, the current snippet, and
anything that affects the fix (related config, callers, tests that would break). If you need the
rationale, read the rule's entry under ${SKILL_BASE}/${d.skill}/references/. If investigation shows the
finding is actually a non-issue, say so plainly in the ticket and mark it accordingly — do not invent a fix.

STEP 2 — Write the ticket to ${OUT}/tickets/${ticketId}.md with EXACTLY this shape:

# ${ticketId}: <concise imperative title>

**Severity:** ${f.severity}  |  **Rule:** ${f.ruleId} (${d.skill})  |  **Area:** <files / subsystem>

## Context
<What is wrong today and why it matters, grounded in this repo — not generic advice.>

## Current state
<Exact file:line references + a fenced snippet of what exists now.>

## Proposed fix
<The minimal, surgical change. Show before/after or the exact edit. Keep it scoped to this finding only.>

## Acceptance criteria
- [ ] <verifiable checks a reviewer can confirm>

## Notes / risks
<Edge cases, related rule IDs, ordering vs other tickets, anything to watch.>

STEP 3 — Return the ticket metadata (ticketId, file path written, title, severity).`
}

// --- run --------------------------------------------------------------------
phase('Audit')
log(`Auditing ${DOMAINS.length} best-practices sub-skills: ${DOMAINS.map(d => d.key).join(', ')}`)

const audits = (await parallel(DOMAINS.map(d => () =>
  agent(auditPrompt(d), { label: `audit:${d.key}`, phase: 'Audit', schema: AUDIT_SCHEMA })
    .then(a => ({ d, a }))
))).filter(Boolean)

// Assign deterministic, collision-free ticket IDs from audit order.
const work = []
for (const { d, a } of audits) {
  (a.findings || []).forEach((f, i) => {
    work.push({ d, f, ticketId: `BP-${d.abbr}-${String(i + 1).padStart(2, '0')}` })
  })
}
log(`Audits complete — ${work.length} findings -> writing ${work.length} tickets`)

phase('Ticket')
const tickets = (await parallel(work.map(({ d, f, ticketId }) => () =>
  agent(ticketPrompt(d, f, ticketId), { label: `ticket:${ticketId}`, phase: 'Ticket', schema: TICKET_SCHEMA })
))).filter(Boolean)

return {
  domains: audits.map(({ d, a }) => ({
    domain: d.key,
    skill: d.skill,
    findingsFile: `.best-practices-audit/findings/${d.key}.md`,
    findings: (a.findings || []).map(f => ({ ruleId: f.ruleId, severity: f.severity, title: f.title })),
  })),
  tickets: tickets.map(t => ({ id: t.ticketId, severity: t.severity, title: t.title, file: t.file })),
}
