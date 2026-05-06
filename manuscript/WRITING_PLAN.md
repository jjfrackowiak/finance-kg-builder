# Article Writing Plan — "Ontology as a Hyperparameter"

## Status legend
- `[ ]` not started
- `[~]` in progress
- `[x]` done

---

## Phase 0 — Infrastructure

| # | Task | Status | Notes |
|---|------|--------|-------|
| 0.1 | Tectonic installed, `main.tex` compiles | `[x]` | v0.16.9, produces `main.pdf` |
| 0.2 | All 9 section stubs created | `[x]` | all `\TODO{Write this section}` |
| 0.3 | `arxiv-latex-mcp` configured | `[x]` | in `.claude/settings.json` |
| 0.4 | Neo4j MCP available | `[x]` | global config |
| 0.5 | Fill missing BibTeX entries | `[ ]` | see §BibTeX TODOs below |
| 0.6 | Create main architecture figure | `[x]` | `figures/architecture.png` + `figures/pipeline_detail.png` (from seminar slides) |

### BibTeX TODOs (`references.bib`)
- `choi2020geval` — GEval, Choi & Jung 2020, PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC7250612/
- `seo2023structuralquality` — Paulheim et al. structural vs downstream quality metrics (2024 per slides)
- `aksw2024llmkgbench` — LLM-KG-Bench, AKSW 2024
- `mynarz2023testdriven` — Mynarz & Hanikova, Test-driven KG (already partially in .bib, verify)
- ontology embedding paper — identify & add key

---

## Phase 1 — Immovable sections (write now, no experiments needed)

### §02 Related Work — `sections/02_related_work.tex`

Cluster structure (from seminar slide 7):

| Cluster | Papers to cite | Task |
|---------|---------------|------|
| KG Construction from Text | Buitelaar 2005, Hogan 2021, LlamaIndex, LangChain, Neo4j LLM Builder, KGGen (Mo 2025), AutoGraph-R1 (Tsang 2025), Mynarz 2023 | `[ ]` |
| KG Quality & Task-Based Eval | Paulheim 2017, Paulheim 2024, GEval (Choi 2020), KGrEaT (Heist 2023), KGBench (Bloem 2021), LLM-KG-Bench (AKSW 2024) | `[ ]` |
| Task-Aware Graph Construction | AuGraph (Cucumides 2025), FinReflectKG (Arun 2025) | `[ ]` |
| Graph Features for Prediction | PathSim (Sun 2011), metapath2vec (Dong 2017), R-GCN (Schlichtkrull 2018), Wang 2017, Ji 2021 | `[ ]` |
| Gap statement | No lightweight predictor-driven ontology feedback loop exists | `[ ]` |

Steps:
- `[ ]` Fetch each paper's methodology section via `arxiv-latex-mcp` (`get_paper_section`)
- `[ ]` Fill in missing BibTeX entries (Phase 0.5)
- `[ ]` Write ~1000-word Related Work ending with explicit gap statement: *"No lightweight error-driven ontology adaptation"*
- `[ ]` Compile and verify all `\cite{}` keys resolve

### §03 Research Hypotheses — `sections/03_hypotheses.tex`

Three hypotheses (verbatim from seminar slide 11):
- **H1:** Task-aware KG construction improves prediction accuracy in node-level price movement classification.
- **H2:** Task-aware optimization enables smaller knowledge graphs while maintaining or improving downstream task performance.
- **H3:** Iterative, task-driven construction produces more informative graph structures and facilitates the identification of effective ontologies.

Steps:
- `[ ]` Write preamble (1 paragraph motivating the hypotheses)
- `[ ]` State each hypothesis formally (numbered, bold, testable)
- `[ ]` Add one sentence per hypothesis explaining how it will be tested (metric → section cross-ref)
- `[ ]` Compile and verify

### §04 Methodology — `sections/04_methodology.tex`

Sub-sections needed:
1. **System Overview** — feedback loop figure (`\ref{fig:architecture}`), high-level narrative
2. **Ontology as Hyperparameter** — formal definition: ontology $\mathcal{O}$, variant $G_k$, metric $m(G_k)$
3. **LLM Agents** — Ontology Evolution Agent (`orchestrator.py`) + KG Builder Agent (`pipeline/`)
4. **Feature Vector Construction** — 4-component vector from `feature_engineering.py`:
   - Node type histogram: 16 hash buckets
   - Rel type histogram: 24 hash buckets (`compute_relation_type_counts`)
   - Metapath counts: 48 hash buckets (2-hop & 3-hop typed paths)
   - Temporal novelty stats: 8 scalars
   - Topology features appended last
5. **Downstream Predictor** — logistic regression / XGBoost, AUC/F1 as feedback signal
6. **Feedback Loop Formalization** — algorithm box: `\begin{algorithm}`

Steps:
- `[ ]` Create architecture figure (`figures/architecture.pdf` or `.png`) — hand-drawn or TikZ
- `[ ]` Write §4.1 with `\ref{fig:architecture}`
- `[ ]` Formalize ontology definition (equations)
- `[ ]` Describe each agent with reference to codebase module paths
- `[ ]` Write feature vector section with equation for $\mathbf{f}_t$
- `[ ]` Write algorithm box for the feedback loop
- `[ ]` Compile and verify

---

## Phase 2 — Experiment-dependent sections (need full experimental run first)

### Pre-condition: run the full pipeline at scale
- `[ ]` Define experiment protocol: which hyperparams, how many iterations, which tickers
- `[ ]` Run pipeline (`orchestrator.py`) at scale
- `[ ]` Export result JSONs and Neo4j state

### §05 Experimental Setup — `sections/05_experiments.tex`
- `[ ]` Describe FNSPID dataset (cite `dong2024fnspid`)
- `[ ]` Define task: next-day binary price movement classification
- `[ ]` Define baselines: static KG (no feedback), bag-of-words, raw price features
- `[ ]` Document hyperparameter search space: `min_count`, `max_count`, `lookback_days`
- `[ ]` Evaluation protocol: temporal train/test split (flag: tricky — see seminar slide 17)

### §06 Results — `sections/06_results.tex`
- `[ ]` Query AUC/F1 per iteration from Neo4j (`read-neo4j-cypher`)
- `[ ]` Plot AUC curve over iterations → `figures/results_auc.pdf`
- `[ ]` KG size vs performance tradeoff plot (for H2) → `figures/kg_size_vs_auc.pdf`
- `[ ]` Ontology evolution trace (which relation types added/removed)
- `[ ]` Table: baseline comparison

---

## Phase 3 — Framing sections (written after body is locked)

### §01 Introduction — `sections/01_introduction.tex`
- `[ ]` Write after §02–04 are locked
- `[ ]` Source: `Research Project Description.pdf` for motivation + gap

### §07 Discussion — `sections/07_discussion.tex`
- `[ ]` Interpret results in light of H1/H2/H3
- `[ ]` Limitations: test split challenge, LLM cost, single domain
- `[ ]` Explainability gap: tracing performance change to specific ontology decision

### §08 Conclusion — `sections/08_conclusion.tex`
- `[ ]` Summarize contributions (from seminar slide 18)
- `[ ]` Future work: Article 2 (finance + pharma), KGrEaT integration, cloud scaling

### §00 Abstract — `sections/00_abstract.tex`
- `[ ]` Write last — distill from all sections
- `[ ]` Target: ~250 words, structured (motivation / method / result / contribution)

---

## Figures checklist

| File | Content | Needed for | Status |
|------|---------|-----------|--------|
| `figures/architecture.png` | Feedback loop diagram | §04 | `[x]` |
| `figures/pipeline_detail.png` | Detailed processing pipeline | §04 | `[x]` |
| `figures/results_auc.pdf` | AUC over iterations | §06 | `[ ]` |
| `figures/kg_size_vs_auc.pdf` | KG size vs performance | §06 (H2) | `[ ]` |
| `figures/ontology_evolution.pdf` | Relation types added/removed per step | §06 | `[ ]` |

---

## Final pre-submission checklist

- `[ ]` All `\TODO{}` markers removed or resolved
- `[ ]` All `\cite{}` keys resolve without warnings
- `[ ]` `tectonic main.tex` compiles clean (no errors, no missing refs)
- `[ ]` Abstract ≤ 250 words
- `[ ]` All figures referenced in text with `\ref{fig:...}`
- `[ ]` Spell-check pass
- `[ ]` Co-authors review pass
