# Healthcare Information Retrieval with PySpark

An end-to-end, local Apache Spark application for information storage and retrieval over a small clinical corpus. The system builds a positional index, derives term statistics and TF-IDF vectors, and answers single-term, phrase, Boolean AND, and Boolean AND NOT queries with cosine-similarity ranking.

**Stack:** Python 3.14.7 · PySpark 4.3.0 · Apache Spark 4.3.0 · OpenJDK 25.0.3

[Overview](#overview) · [Quick start](#quick-start) · [Architecture](#architecture) · [Dataset](#dataset) · [Project structure](#project-structure) · [Requirements](#requirements) · [Installation](#installation) · [Usage](#usage) · [Output files](#output-files) · [How it works](#how-it-works) · [Sample results](#sample-results) · [Screenshots](#screenshots) · [Troubleshooting](#troubleshooting) · [Limitations](#limitations-and-future-work)

---

## Overview

The project implements a complete lexical retrieval pipeline:

1. **Positional index construction** — reads the corpus, tokenizes each document, assigns 1-based token positions, groups postings by term, and writes a human-readable index and a JSON index.
2. **Data processing** — loads the JSON index, reconstructs a Spark DataFrame of `(term, doc_id, position)` rows, and computes raw TF, DF, IDF, weighted TF, TF-IDF, document vector lengths, and normalized TF-IDF.
3. **Query processing and ranking** — supports single-term search, exact phrase search, Boolean AND, and Boolean AND NOT. Phrase matching uses contiguous positional joins; ranking uses cosine similarity over normalized vectors.

The implementation runs in Spark `local[*]` mode and is intended for a small, inspectable corpus rather than a production cluster.

### Key features

- Exact positional index with 1-based token positions.
- Raw and sublinear term-frequency weighting.
- Document frequency and base-10 inverse document frequency.
- TF-IDF matrix and L2-normalized document vectors.
- Exact phrase matching through consecutive-position self-joins.
- Boolean AND (set intersection) and Boolean AND NOT (set difference).
- Interactive command-line query loop.
- Human-readable text reports plus a machine-readable JSON index.
- Deterministic numeric document ordering (`d2` before `d10`) in the index and matrix outputs.

---

## Quick start

### PowerShell

```powershell
cd D:\path\to\IR-Project

# Use the same interpreter that has PySpark installed
$env:PYSPARK_PYTHON = (py -c "import sys; print(sys.executable)")

# Build the positional index
spark-submit src\positional_index.py

# Compute the retrieval model and start the query loop
spark-submit src\retrieval_ranking\main.py
```

### Bash / Git Bash

```bash
cd /path/to/IR-Project

# Point this at the interpreter where PySpark is installed
export PYSPARK_PYTHON="C:/Users/<you>/AppData/Local/Python/pythoncore-3.14-64/python.exe"

spark-submit src/positional_index.py
spark-submit src/retrieval_ranking/main.py
```

On Windows, `py -c "import sys; print(sys.executable)"` prints the path to use. In Git Bash, assign that path directly to `PYSPARK_PYTHON`.

Run both commands from the project root; the scripts use relative `data/` and `outputs/` paths.

The index builder must run before the retrieval application because the retrieval application reads `outputs/positional_index.json`.

For a clean regeneration, delete `outputs/` and run both commands again.

---

## Architecture

```text
data/d1.txt … data/d10.txt
            │
            ▼
┌──────────────────────────────────────────────┐
│ Positional Index Construction                │
│ Spark RDDs · tokenize · 1-based positions    │
│ groupByKey · aggregate and sort postings     │
└──────────────────────────────────────────────┘
            │  outputs/positional_index.json
            ▼
┌──────────────────────────────────────────────┐
│ Data Processing                              │
│ Spark DataFrames · TF · DF · IDF · TF-IDF    │
│ vector lengths · L2 normalization            │
└──────────────────────────────────────────────┘
            │  TF / DF-IDF / TF-IDF / normalized matrices
            ▼
┌──────────────────────────────────────────────┐
│ Query Processing and Ranking                 │
│ phrase joins · Boolean set operations        │
│ query vectors · cosine similarity · ranking  │
└──────────────────────────────────────────────┘
            │
            ▼
outputs/*.txt  +  interactive console results
```

### Components

| Component | File | Responsibility |
|---|---|---|
| Positional Index Construction | `src/positional_index.py` | Reads the corpus, tokenizes documents, assigns 1-based positions, builds the term-to-postings index, and writes text and JSON outputs. |
| Application Entry Point | `src/retrieval_ranking/main.py` | Creates the Spark session, computes and prints the matrices, and runs the interactive query loop. |
| Data Processing | `src/retrieval_ranking/data_processor.py` | Loads the JSON index, flattens postings into a DataFrame, and computes TF, DF, IDF, weighted TF, TF-IDF, vector lengths, and normalized weights. |
| Query Processing and Ranking | `src/retrieval_ranking/query_engine.py` | Implements single-term search, phrase matching, Boolean AND/AND NOT, query-vector construction, cosine similarity, ranking, and query reports. |
| Formatting and Output Utilities | `src/retrieval_ranking/utils.py` | Formats numeric values, prints tables, writes table files, appends query sections, and sorts document IDs numerically. |

---

## Dataset

The corpus contains ten short clinical documents. Tokenization is lowercase whitespace splitting.

| Document | Content |
|---|---|
| `d1.txt` | patient care |
| `d2.txt` | patient safety needs careful monitoring |
| `d3.txt` | medical research improves treatment |
| `d4.txt` | medical treatment supports patient recovery |
| `d5.txt` | patient care patient safety medical research patient care |
| `d6.txt` | hospital staff provide patient care |
| `d7.txt` | clinical teams support patient recovery through treatment |
| `d8.txt` | medical research uses patient records and clinical data |
| `d9.txt` | patient monitoring patient monitoring supports care |
| `d10.txt` | health information systems improve patient safety medical records clinical services |

Corpus statistics:

| Metric | Value |
|---|---|
| Documents | 10 |
| Indexed tokens | 60 |
| Vocabulary | 28 unique terms |
| IDF base | 10 |
| Spark master | `local[*]` |

---

## Project structure

```text
IR-Project/
├── data/
│   ├── d1.txt
│   ├── d2.txt
│   ├── ...
│   └── d10.txt
├── outputs/                       # generated index, matrices, and query reports
├── screenshots/                   # captured console and matrix views
├── src/
│   ├── positional_index.py        # positional index construction
│   └── retrieval_ranking/
│       ├── __init__.py
│       ├── main.py                # application entry point and CLI
│       ├── data_processor.py      # TF, DF/IDF, TF-IDF, lengths, normalization
│       ├── query_engine.py        # phrase/Boolean queries, similarity, ranking
│       └── utils.py               # table formatting and file writers
├── .gitignore
└── README.MD
```

---

## Requirements

| Component | Verified version | Notes |
|---|---|---|
| Operating system | Windows 11 (10.0, amd64) | The interactive input helper uses `msvcrt`. |
| Java | OpenJDK 25.0.3 | Set `JAVA_HOME` to the JDK installation. Java 17+ is recommended. |
| Apache Spark | 4.3.0 (Hadoop 3 build) | `spark-submit` must be on `PATH`. |
| Python | 3.14.7 | Use the same interpreter for PySpark and `PYSPARK_PYTHON`. |
| PySpark | 4.3.0 | Install in the Python environment used by Spark. |
| Optional | Hadoop `winutils.exe` | Not required for local file I/O; its absence produces a warning. |

Verify the environment:

```powershell
java -version
spark-submit --version
py -c "import pyspark; print(pyspark.__version__)"
py -c "import sys; print(sys.executable)"
```

The final command prints the interpreter path that should be assigned to `PYSPARK_PYTHON`.

---

## Installation

1. Install a JDK and set `JAVA_HOME`.
2. Install Apache Spark and add its `bin` directory to `PATH`.
3. Install Python and PySpark in the same environment:

   ```powershell
   py -m pip install pyspark==4.3.0
   ```

4. Clone or download the repository and open a terminal in the project root.

### Configuration

| Variable | Purpose | Example |
|---|---|---|
| `JAVA_HOME` | JDK location | `C:\Program Files\Eclipse Adoptium\jdk-25...` |
| `SPARK_HOME` | Spark installation | `C:\spark-4.3.0-bin-hadoop3` |
| `PATH` | Locate `spark-submit` | `%SPARK_HOME%\bin` |
| `PYSPARK_PYTHON` | Python interpreter for Spark workers | Output of `py -c "import sys; print(sys.executable)"` |

Using a different interpreter for the driver and the workers is the most common cause of Python worker connection errors.

---

## Usage

### Query syntax

Queries are case-insensitive. Boolean keywords must be surrounded by spaces.

| Query type | Syntax | Example |
|---|---|---|
| Single term | One token | `patient` |
| Phrase | Two or more tokens | `patient care` |
| Boolean AND | phrase `AND` phrase | `patient care AND medical research` |
| Boolean AND NOT | phrase `AND NOT` phrase | `patient care AND NOT medical research` |
| Exit | `quit` or an empty line | `quit` |

The parser supports one Boolean operator and does not implement `OR`, parentheses, or quoted phrases. Each Boolean operand is treated as a whitespace-separated phrase.

### Example session

```text
Enter query (or 'quit' to exit): patient care

--- Phrase Query: "patient care" ---
Query vector length = 0.4006
...
Ranking (highest similarity first): d1 > d5 > d6

Enter query (or 'quit' to exit): patient care AND NOT medical research

--- Boolean AND NOT: "patient care AND NOT medical research" ---
...
Ranking (highest similarity first): d1 > d6
```

---

## Output files

| File | Produced by | Contents |
|---|---|---|
| `outputs/positional_index.txt` | `positional_index.py` | Human-readable postings: term, document IDs, and 1-based positions. |
| `outputs/positional_index.json` | `positional_index.py` | Nested index: `term → document → [positions]`. |
| `outputs/tf_matrix.txt` | `data_processor.py` | Raw term frequency for every term and document. |
| `outputs/weighted_tf_matrix.txt` | `data_processor.py` | Weighted TF: `1 + log10(TF)`. |
| `outputs/df_idf.txt` | `data_processor.py` | Document frequency and IDF for every term. |
| `outputs/tfidf_matrix.txt` | `data_processor.py` | TF-IDF weights. |
| `outputs/doc_lengths.txt` | `data_processor.py` | L2 length of each document vector. |
| `outputs/normalized_tfidf_matrix.txt` | `data_processor.py` | TF-IDF divided by the document vector length. |
| `outputs/query_single_<term>.txt` | `query_engine.py` | Single-term query vector, matches, similarities, and ranking. |
| `outputs/query_phrase_<terms>.txt` | `query_engine.py` | Phrase starts, token ranges, query vector, similarities, and ranking. |
| `outputs/query_and_<phrase1>_<phrase2>.txt` | `query_engine.py` | Intersection matches, query vector, similarities, and ranking. |
| `outputs/query_andnot_<phrase1>_<phrase2>.txt` | `query_engine.py` | Difference matches, first-phrase query vector, similarities, and ranking. |

The `outputs/` directory is created automatically. Its contents are generated files and are ignored by Git.

---

## How it works

### Positional index construction

Each document is lowercased and split on whitespace. Every token becomes a posting:

```text
(term, (doc_id, 1-based position))
```

Postings are grouped by term. For each term, positions are grouped by document and sorted. The final structure is:

```text
term → { doc_id: [position, ...] }
```

Document IDs are sorted by their numeric suffix so that `d2` precedes `d10` in the index and matrix outputs.

### Term statistics and normalization

| Quantity | Definition |
|---|---|
| `TF(t, d)` | Number of occurrences of term `t` in document `d`. |
| `DF(t)` | Number of distinct documents containing `t`. |
| `IDF(t)` | `log10(N / DF(t))`, where `N` is the number of documents. |
| `WeightedTF(t, d)` | `1 + log10(TF(t, d))` when `TF > 0`; otherwise `0`. |
| `TFIDF(t, d)` | `WeightedTF(t, d) × IDF(t)`. |
| `||d||₂` | `sqrt(Σ TFIDF(t, d)²)`. |
| `NormalizedTFIDF(t, d)` | `TFIDF(t, d) / ||d||₂`. |

### Query processing and ranking

- **Single-term query:** finds all documents containing the term and ranks them by cosine similarity.
- **Phrase query:** requires consecutive positions. For a phrase `t₁ … tₘ` starting at position `p`, term `tᵢ` must occur at `p + i − 1`.
- **Boolean AND:** intersects the document sets that contain the first and second phrases.
- **Boolean AND NOT:** subtracts the second phrase's document set from the first phrase's document set.
- **Query vector:** each query term is weighted by its IDF, then L2-normalized.
- **Similarity:** the dot product of the normalized query vector and the normalized document vector.
- **Ranking:** documents are sorted by descending similarity.

### Algorithm summary

| Stage | Algorithm | Spark operations |
|---|---|---|
| Index construction | Lowercase whitespace tokenization; assign 1-based positions; group postings by term and document. | `flatMap`, `groupByKey`, `map` |
| Term statistics | Count term occurrences per document; count distinct documents per term. | `groupBy`, `count`, `countDistinct` |
| Weighting | Sublinear TF `1 + log10(TF)`; IDF `log10(N/DF)`; multiply; L2-normalize. | `withColumn`, `log10`, `sqrt`, `sum` |
| Phrase matching | Join postings for each phrase term on document and require consecutive positions. | Self-joins and position filters |
| Boolean retrieval | Set intersection for AND; set difference for AND NOT. | Python set operations over collected document IDs |
| Ranking | Dot product of normalized query and document vectors; sort descending. | Driver-side accumulation and sort |

---

## Sample results

| Query | Matching documents | Ranking |
|---|---|---|
| `patient care` | `d1`, `d5`, `d6` | `d1` 1.0000 > `d5` 0.5473 > `d6` 0.2253 |
| `medical research` | `d3`, `d5`, `d8` | `d5` 0.6324 > `d3` 0.4715 > `d8` 0.2969 |
| `patient care AND medical research` | `d5` | `d5` 0.8296 |
| `patient care AND NOT medical research` | `d1`, `d6` | `d1` 1.0000 > `d6` 0.2253 |

### Verification notes

- The corpus produces 10 documents, 60 indexed tokens, and 28 vocabulary terms.
- All TF, weighted TF, DF/IDF, TF-IDF, document-length, and normalized matrices are written to `outputs/`.
- The four queries return the expected document sets and rankings.
- The query vector is rounded to four decimals before the dot product. As a result, the `medical research` similarities for `d5` and `d8` are `0.6324` and `0.2969` rather than the full-precision `0.6325` and `0.2970`. The ranking is unchanged.
- Phrase start positions are set-valued. Spark does not guarantee collection order, so the display order of multiple start positions may vary between runs.

---

## Screenshots

| View | Image |
|---|---|
| Positional index | [Positional Index](screenshots/Positional%20Index.png) |
| Raw TF matrix | [Raw TF Matrix](screenshots/Raw%20TF%20Matrix.png) |
| Weighted TF matrix | [Weighted TF Matrix](screenshots/Weighted%20TF%20Matrix.png) |
| DF and IDF | [DF and IDF Table](screenshots/DF%20and%20IDF%20Table.png) |
| TF-IDF matrix | [TF-IDF Matrix](screenshots/TF-IDF%20Matrix.png) |
| Document vector lengths | [Document Vector Lengths](screenshots/Document%20Vector%20Lengths.png) |
| Normalized TF-IDF matrix | [Normalized TF-IDF Matrix](screenshots/Normalized%20TF-IDF%20Matrix.png) |
| Phrase query | [Phrase Query - patient care](screenshots/Phrase%20Query%20-patient%20care.png) |
| Boolean AND | [Boolean AND](screenshots/Boolean%20AND.png) |
| Boolean AND NOT | [Boolean AND NOT](screenshots/Boolean%20AND%20NOT.png) |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError: outputs\positional_index.json` | The retrieval application was started before the index builder. | Run `spark-submit src\positional_index.py` first. |
| Python worker failed to connect | `PYSPARK_PYTHON` points to a different interpreter. | Set it to the exact output of `py -c "import sys; print(sys.executable)"`. |
| `spark-submit` is not recognized | Spark's `bin` directory is not on `PATH`. | Add `%SPARK_HOME%\bin` to `PATH`. |
| `Unsupported class file major version` | Java version is incompatible with the Spark build. | Use a JDK supported by the installed Spark version and set `JAVA_HOME`. |
| `winutils.exe` warning | Hadoop native utilities are absent on Windows. | Ignore for local file I/O, or install matching Hadoop winutils. |
| Port `4040` already in use | Another Spark application is running. | Stop the other application or ignore; Spark selects another UI port. |
| No query prompt | Spark is still computing the matrices, or the console is not focused. | Wait for the prompt and click the console window before typing. |
| Empty or stale outputs | The previous run was interrupted or `outputs/` was not cleaned. | Delete `outputs/`, rerun the index builder, then run the retrieval application. |

---

## Limitations and future work

| Limitation | Possible improvement |
|---|---|
| No stemming or lemmatization | Add a stemmer or lemmatizer and rebuild the index. |
| No stop-word removal | Add a configurable stop-word list and compare results. |
| Whitespace tokenization only | Add a Unicode-aware tokenizer and punctuation filtering. |
| Exact phrase matching only | Add proximity search with a configurable maximum distance. |
| One Boolean operator, no parentheses | Add a recursive-descent parser and expression tree with `OR` support. |
| Local mode and driver-side collection | Write partitioned indexes, use `aggregateByKey`, and run on a cluster. |
| Dense term-by-document matrices | Use sparse vectors or keep the relational representation. |
| Windows-specific `msvcrt` input | Use `input()` on non-Windows platforms or a platform abstraction. |
| No automated tests | Add unit tests for tokenization, formulas, phrase matching, and set operations. |

Additional extensions could include precision/recall evaluation, a larger corpus, a web or desktop interface, and Parquet persistence for the index.

---

## License

This project was developed for academic and educational purposes. No open-source license is declared.

---

## Acknowledgements

Built with [Apache Spark](https://spark.apache.org/), [PySpark](https://spark.apache.org/docs/latest/api/python/), and Python.
