# NPT Agents — Evidence-Grounded Multi-Agent Simulation

This repository implements an **evidence-grounded, multi-agent simulation** of diplomatic statements in the **Nuclear Non-Proliferation Treaty (NPT) review process**.

The system ingests official NPT documents (statements and working papers), structures them into a queryable database, classifies policy positions using LLMs, and simulates country-level negotiation dynamics by **retrieving and citing real textual evidence**.

The project is designed as a **research engineering prototype** rather than a toy demo, with attention to:
- incremental data processing (no re-classification unless necessary),
- reproducibility,
- and clear separation between data, modeling, and presentation layers.

---

## What This Project Does

At a high level, the system supports:

1. **Document ingestion**
   - Extracts paragraphs from NPT statements and working papers (PDFs)
   - Stores them in a structured SQLite database

2. **LLM-based policy classification**
   - Assigns each paragraph to one or more NPT policy issues
   - Generates concise position summaries and confidence scores
   - Designed to be incremental to avoid re-processing existing data

3. **Evidence indexing**
   - Groups classified paragraphs into per-country, per-issue “buckets”
   - Builds semantic embeddings for evidence retrieval

4. **Multi-agent simulation**
   - Simulates multi-round diplomatic exchanges
   - Each agent retrieves and cites relevant evidence
   - Outputs both internal “issue expert” drafts and final delegation statements

5. **Interactive visualization**
   - A Streamlit UI to explore simulation outputs and supporting evidence

---

## Project Structure

```
NPT_AGENTS/
│
├── statements/ # Official NPT statements (PDF)
├── working_papers/ # NPT working papers (PDF)
│
├── positions.db # Structured paragraph-level database (local)
│
├── extract_paragraphs_to_db.py
├── classify_paragraphs_with_gpt.py
├── export_buckets_python_only.py
├── build_bucket_embeddings.py
├── build_lite_buckets.py
│
├── agent_simulation_new.py # Core multi-agent simulation logic
├── streamlit_app.py # Streamlit UI for results
│
├── cli.py # Unified command-line interface
├── config.py # Centralized configuration
├── constants.py # Domain constants (e.g., NPT issue list)
│
├── outputs/ # Generated artifacts (buckets, logs, embeddings)
├── requirements.txt # Minimal dependencies (for Streamlit deployment)
├── requirements-full.txt # Full local development dependencies
└── README.md
```

---

## Design Principles

- **Incremental & safe by default**  
  Existing database entries are never dropped or re-classified unless explicitly intended.

- **Evidence-grounded generation**  
  All simulated statements are anchored in retrieved textual evidence.

- **Separation of concerns**  
  Extraction, classification, indexing, simulation, and visualization are independent steps.

- **Low-overhead engineering**  
  The project avoids heavy frameworks in favor of clarity and inspectability.

---

## Configuration

All runtime parameters (paths, models, thresholds, simulation settings) are centralized in:

```python
config.py
```

## Quick Start (Typical Workflow)
```bash
# 1. Extract paragraphs from NEW PDFs (safe & incremental)
python cli.py extract

# 2. Classify only unprocessed paragraphs
python cli.py classify

# 3. Build evidence buckets and embeddings
python cli.py export
python cli.py embed
python cli.py lite

# 4. Run the multi-agent simulation
python cli.py simulate

# 5. Launch the interactive UI
python cli.py ui

```

---

## Example Use Cases

- Exploring how different countries frame the same NPT issue

- Testing how evidence selection affects simulated negotiation dynamics

- Prototyping agent-based models for treaty review processes

- Demonstrating evidence-grounded LLM systems for policy analysis

---

## Notes on Data & Cost

- LLM-based classification and embedding steps incur API costs.
- The system is designed to avoid re-processing existing data.
- Backups and safety checks are intentionally emphasized.

---

## Status

This project is an active research prototype.
It prioritizes transparency, modularity, and analytical clarity over production hardening.

---

## License

This repository is provided for research and demonstration purposes.