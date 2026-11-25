# IR Project — PySpark Information Retrieval

Information Storage and Retrieval project built with Apache Spark (PySpark).

## Prerequisites

- Java (JDK 8 or 11)
- Apache Spark (pre-built binary for Hadoop 3, e.g. Spark 3.5.x)
- Python (matching your Spark version, e.g. PySpark 3.5.x)
- Python 3.14 is used by this project

Verify with:

```
java -version
spark-submit --version
py -c "import pyspark; print(pyspark.__version__)"
```

Set the Python path for PySpark before running:

```
export PYSPARK_PYTHON="C:/Users/yhaya/AppData/Local/Python/pythoncore-3.14-64/python.exe"
```

## How to Run — PowerShell

```powershell
Remove-Item -Force -Recurse outputs\
$env:PYSPARK_PYTHON="C:\Users\yhaya\AppData\Local\Python\pythoncore-3.14-64\python.exe"
spark-submit src\positional_index.py
spark-submit src\retrieval_ranking\main.py
```

## How to Run — Bash (Git Bash / MINGW64)

```bash
rm -rf outputs/
export PYSPARK_PYTHON="C:/Users/yhaya/AppData/Local/Python/pythoncore-3.14-64/python.exe"
spark-submit src/positional_index.py
spark-submit src/retrieval_ranking/main.py
```

## What Each Part Does

### Part A: Positional Index (`src/positional_index.py`)

Reads `data/d1.txt` to `d10.txt`, tokenizes by whitespace, tracks 1-based positions, and builds the positional index using Spark RDDs.

Creates:
- `outputs/positional_index.txt` — human-readable positional index
- `outputs/positional_index.json` — structured positional index (used by Part B)

### Part B: Retrieval & Ranking (`src/retrieval_ranking/main.py`)

Loads the Positional Index, computes TF, DF, IDF, Weighted TF, TF-IDF, document vector lengths, and Normalized TF-IDF using PySpark DataFrames.

Prints interactive queries:
- Single term search (e.g., `patient`)
- Phrase queries (e.g., `patient care`)
- Boolean AND (e.g., `patient care AND medical research`)
- Boolean AND NOT (e.g., `patient care AND NOT medical research`)

All results and matrices are saved to `outputs/` as `.txt` files.

## Project Structure

```
IR-Project/
├── data/
│   └── d1.txt ... d10.txt
├── outputs/
├── src/
│   ├── positional_index.py
│   └── retrieval_ranking/
│       ├── main.py
│       ├── data_processor.py
│       ├── query_engine.py
│       └── utils.py
└── README.md
```

## Notes

- Always run Part A before Part B (Part B needs the JSON file from Part A)
- Delete `outputs/` to regenerate all outputs
- If interactive input is slow, wait a moment after the prompt appears — Spark is still processing