# Manuscript — Article 1

**Title:** Ontology as a Hyperparameter: Task-Aware Knowledge Graph Construction Using LLMs  
**Authors:** Jan Frąckowiak, Piotr Wójcik, Michał Sierakowski (University of Warsaw)  
**Goal:** Write, compile, and iterate on the full LaTeX manuscript for Article 1 of the PhD thesis.

---

## What This Article Argues

Knowledge graph ontology should be treated as a **hyperparameter optimized for downstream task performance**, not a fixed design choice. We demonstrate a closed feedback loop:

1. LLM agents (Ontology Agent + KG Builder Agent) generate multiple candidate KG variants under different ontology configurations
2. Graph-derived features (subgraph embeddings, topology, metapaths) are fed into downstream predictors (logistic regression, XGBoost)
3. Predictor performance (AUC, F1) feeds back to refine the ontology
4. Repeat until convergence or budget exhausted

Primary empirical setting: **FNSPID** financial news → next-day stock price movement prediction.

---

## Directory Layout

```
manuscript/
  CLAUDE.md             ← this file
  main.tex              ← preamble, \input{} each section, bibliography
  references.bib        ← all citations (BibTeX)
  sections/
    00_abstract.tex
    01_introduction.tex
    02_related_work.tex
    03_hypotheses.tex
    04_methodology.tex
    05_experiments.tex
    06_results.tex
    07_discussion.tex
    08_conclusion.tex
  figures/              ← PDF/PNG exports of plots and diagrams
  *.pdf / *.docx        ← reference materials (do not edit)
```

---

## Compiling the Paper

Install tectonic (first time only):
```bash
brew install tectonic
```

Compile from the `manuscript/` directory:
```bash
tectonic main.tex
```

Output: `main.pdf` in the same directory. Tectonic downloads missing LaTeX packages automatically on first run.

---

## MCP Tools and When to Use Them

### `arxiv-latex-mcp` — fetch raw LaTeX source of any arXiv paper
**This is the most important writing tool.** Fetches the actual `.tex` source of arXiv papers — not lossy PDF text — so math, equations, and methodology descriptions are read exactly as written.

Already installed in `.claude/settings.json` — no setup needed.

Available tools:
- `get_paper_prompt` — full flattened LaTeX of a paper (give it the arXiv ID e.g. `2308.10537`)
- `get_paper_abstract` — just the abstract
- `list_paper_sections` — section headings
- `get_paper_section` — a specific section by path

Use for: Related Work (read methodology sections of cited papers verbatim), verifying how a cited paper defines its evaluation metrics, understanding math we reference.

---

### `mcp-neo4j-cypher` (Neo4j / AuraDB)
Use to **query real experimental results** directly from the knowledge graph database.  
Available tools: `get-neo4j-schema`, `read-neo4j-cypher`.  
Use for: node/relationship counts per experiment step, AUC/F1 scores stored as graph metadata, ontology evolution traces, subgraph statistics.  
Example queries:
```cypher
-- Count entities by type at a given step
MATCH (n) WHERE n.step = 2 RETURN labels(n)[0] AS type, count(*) AS cnt ORDER BY cnt DESC

-- Check candidate tags on relationships
MATCH ()-[r]->() WHERE 'step_2_candidate_1' IN coalesce(r.candidate_tags, [])
RETURN type(r), count(*) AS cnt
```

---

### `context7`
Use to **look up current documentation** for any library or framework cited in the paper: LlamaIndex, LangChain, Neo4j Python driver, XGBoost, sentence-transformers, etc.  
Do not use for general writing or business logic.

---

### `WebSearch` / `WebFetch`
Use to:
- Find BibTeX entries for cited papers (arxiv, ACM DL, Semantic Scholar)
- Verify publication year/venue/authors before citing
- Retrieve full abstracts when writing Related Work
- Check if a more recent version of a cited paper exists

---

## Claude Code Skills (invoke in conversation, no install needed)

These are prompt-based skills callable during a writing session. They are **not** MCP servers — just invoke them by name when needed:

| Skill | When to use |
|-------|-------------|
| `ml-paper-writing-assistant` | Drafting sections with conference template awareness; citation verification via Semantic Scholar |
| `academic-paper-creator` | Formatting passes — applying two-column layout, headers, proper LaTeX structure |
| `academic-paper-writing-review` | Review pass before submission — checks structure, argument flow, citation completeness |
| `write-latex` | Converting prose drafts to well-structured semantic LaTeX |

Note: availability depends on what is installed in the active Claude Code environment. Try invoking before assuming available.

---

## Tools Evaluated and Rejected

- **mcp-latex-server** (RobertoDure): exposes create/edit/validate/compile tools but **requires a full MacTeX installation** and duplicates functionality already covered by Read/Edit/Write tools + tectonic via bash. Skip.

---

## Article Structure and Section Ownership

Each section lives in its own `.tex` file under `sections/`. When working as a subagent on a section, **only edit that section's file**. Read `main.tex` and `references.bib` for style and bib context. Never rewrite another section.

### Writing status

| File | Section | Status | Key sources |
|------|---------|--------|-------------|
| `00_abstract.tex` | Abstract | last — distilled from all sections | everything |
| `01_introduction.tex` | Introduction | after body is done | Research Project Description.pdf, gap from Related Work |
| `02_related_work.tex` | **Related Work** | **write now (immovable)** | `references.bib` + arxiv-latex-mcp for paper content |
| `03_hypotheses.tex` | **Research Hypotheses** | **write now (immovable)** | 3 hypotheses from seminar slides |
| `04_methodology.tex` | **Methodology** | **write now (immovable)** | `kg_builder_llm/` codebase — pipeline/, ml/, mutations/ |
| `05_experiments.tex` | Experimental Setup | fill later (needs scale) | FNSPID dataset, baseline definitions |
| `06_results.tex` | Results | fill later (needs scale) | result JSONs, Neo4j MCP, figures/ |
| `07_discussion.tex` | Discussion | fill later | interpret results, limitations |
| `08_conclusion.tex` | Conclusion | fill later | Article 2 as continuation |

The **immovable sections** (Related Work, Hypotheses, Methodology) do not depend on the scale of experiments and can be written and locked now. Sections 05–08 await a larger experimental run.

---

## Writing Style

- LaTeX, plain `article` class (12pt, a4paper, onehalfspacing) — already set in `main.tex`
- Third person, passive where conventional in ML papers ("we propose", "results indicate")
- Equations for the feedback loop formalization, feature vector composition
- All claims backed by either: (a) citation, (b) Neo4j query result, or (c) codebase reference
- Figures preferred over tables for trends; tables for precise numbers
- No placeholder text left in submitted draft — mark incomplete sections with `% TODO: ...` comments

---

## Key Literature (for `references.bib`)

Use these **exact BibTeX keys** when citing — they match `references.bib`.

| BibTeX key | Paper |
|------------|-------|
| `dong2024fnspid` | FNSPID Dataset, ACM KDD 2024 |
| `heist2023kgreat` | Heist et al., KGrEaT, CIKM 2023 |
| `bloem2021kgbench` | Bloem et al., KGBench, ESWC 2021 |
| `tsang2025autographr1` | AutoGraph-R1, RL for KG construction, arXiv 2025 |
| `mo2025kggen` | KGGen, KG extraction from text, arXiv 2025 |
| `mynarz2023testdriven` | Mynarz & Hanikova, Test-driven KG, KGC workshop 2023 |
| `cucumides2025augraph` | Cucumides & Geerts, Task-Aware Graph Construction, TaDA@VLDB 2025 |
| `arun2025finreflectkg` | Arun et al., FinReflectKG, ACM AI in Finance 2025 |

TODOs still to fill in `references.bib`: GEval (`choi2020geval`), Paulheim structural quality (`seo2023structuralquality`), LLM-KG-Bench (`aksw2024llmkgbench`), ontology embedding paper.

---

## Codebase Reference Points

When describing the methodology, map to these actual modules:

| Concept | Code location |
|---------|--------------|
| Ontology evolution agent | `kg_builder_llm/pipeline/orchestrator.py` |
| Candidate evaluation | `kg_builder_llm/pipeline/evaluator.py` |
| Feature vector construction | `kg_builder_llm/ml/feature_engineering.py` — `build_day_feature_vector` |
| Subgraph features (4-component vector) | `kg_builder_llm/ml/subgraph_features.py` |
| Topology features | `kg_builder_llm/ml/topology_features.py` |
| Relationship chain extraction | `kg_builder_llm/ml/` chain extraction logic |
| Entity resolution | `kg_builder_llm/core/` |
| Neo4j I/O | `kg_builder_llm/core/` |

Feature vector breakdown (for Methods section):
- Node type histogram: 16 hash buckets
- Rel type histogram: 24 hash buckets (`compute_relation_type_counts`)
- Metapath counts: 48 hash buckets (2-hop and 3-hop typed paths)
- Temporal novelty stats: 8 scalars
- Topology features: appended last (degree, mention counts)

---

## Caution

- The Neo4j orchestrator **wipes the database** at the start of each run — query results reflect the *last completed* experiment only
- Do not commit `.env`, API keys, or credentials anywhere in this directory
- PDF/docx reference files in this directory are read-only source materials
