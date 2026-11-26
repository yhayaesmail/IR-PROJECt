import os
import sys
import io
import json
import contextlib
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
RETRIEVAL_DIR = os.path.join(SRC_DIR, "retrieval_ranking")
for p in (SRC_DIR, RETRIEVAL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from utils import sort_doc_ids, fmt4
from data_processor import compute_all
from query_engine import find_term_docs, find_phrase_positions, build_query_vector, compute_similarity
from positional_index import extract_doc_id, tokenize_with_positions, aggregate_doc_positions, format_human_readable

BG = "#f5f6f7"
CARD = "#ffffff"
FG = "#111111"
ACCENT = "#2c3e50"
ACCENT2 = "#34495e"
MUTED = "#6b7280"
BORDER = "#d1d5db"
BTN = "#2c3e50"
BTN_TEXT = "#ffffff"


def make_tree(parent):
    frame = ttk.Frame(parent)
    frame.pack(fill="both", expand=True, padx=8, pady=8)
    tree = ttk.Treeview(frame, show="headings", style="IR.Treeview")
    vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)
    return tree


def set_table(tree, columns, rows):
    for c in tree.get_children():
        tree.delete(c)
    tree["columns"] = columns
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=max(110, min(200, len(col) * 10)), anchor="center", stretch=True)
    for row in rows:
        tree.insert("", "end", values=[str(x) for x in row])


class IRApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("IR Project — PySpark Retrieval")
        self.geometry("1280x820")
        self.minsize(1000, 600)
        self.configure(bg=BG)
        self.data_dir = None
        self.output_dir = os.path.join(PROJECT_ROOT, "outputs")
        self.spark = None
        self._apply_style()
        self.build_ui()

    def _apply_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD, relief="flat")
        style.configure("TLabel", background=BG, foreground=FG, font=("Segoe UI", 10))
        style.configure("Header.TLabel", background=ACCENT, foreground="#ffffff", font=("Segoe UI", 16, "bold"))
        style.configure("Sub.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Section.TLabel", background=CARD, foreground=ACCENT, font=("Segoe UI", 11, "bold"))
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=8)
        style.configure("IR.TButton", background=BTN, foreground=BTN_TEXT, borderwidth=0)
        style.map("IR.TButton", background=[("active", ACCENT2), ("pressed", ACCENT)])
        style.configure("TEntry", fieldbackground="#ffffff", foreground=FG, bordercolor=BORDER, padding=5)
        style.configure("TCombobox", fieldbackground="#ffffff", foreground=FG, padding=5)
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(16, 8), font=("Segoe UI", 10, "bold"))
        style.configure("IR.Treeview", background=CARD, foreground=FG, fieldbackground=CARD, font=("Segoe UI", 9), rowheight=24)
        style.configure("IR.Treeview.Heading", background=ACCENT, foreground="#ffffff", font=("Segoe UI", 9, "bold"))
        style.map("IR.Treeview", background=[("selected", "#dbeafe")])

    def build_ui(self):
        header = tk.Frame(self, bg=ACCENT)
        header.pack(fill="x")
        tk.Label(header, text="IR Project — PySpark Retrieval", bg=ACCENT, fg="#ffffff", font=("Segoe UI", 16, "bold")).pack(side="left", padx=16, pady=12)
        tk.Label(header, text="Positional Index • TF-IDF • Phrase & Boolean Search", bg=ACCENT, fg="#e5e7eb", font=("Segoe UI", 9)).pack(side="left", padx=4, pady=12)

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        sidebar = tk.Frame(body, bg=CARD, width=300, highlightbackground=BORDER, highlightthickness=1)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        pad = {"padx": 14, "pady": 6}

        tk.Label(sidebar, text="Dataset", bg=CARD, fg=ACCENT, font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=14, pady=(14, 4))
        self.data_dir_var = tk.StringVar(value="")
        self._add_labeled_entry(sidebar, "Folder of .txt files", self.data_dir_var, "...", self.select_dataset, pad)

        self.output_dir_var = tk.StringVar(value=self.output_dir)
        self._add_labeled_entry(sidebar, "Output folder", self.output_dir_var, "...", self.select_output, pad)

        ttk.Separator(sidebar, orient="horizontal").pack(fill="x", padx=14, pady=8)

        tk.Label(sidebar, text="Query", bg=CARD, fg=ACCENT, font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=14, pady=(4, 4))
        tk.Label(sidebar, text="Mode", bg=CARD, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w", padx=14)
        self.mode_var = tk.StringVar(value="Auto")
        self.mode_combo = ttk.Combobox(sidebar, textvariable=self.mode_var, values=["Auto", "Term", "Phrase", "AND", "AND NOT"], state="readonly")
        self.mode_combo.pack(fill="x", padx=14, pady=4)
        tk.Label(sidebar, text="Query entry", bg=CARD, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w", padx=14, pady=(6, 0))
        self.query_var = tk.StringVar(value="")
        self.query_entry = ttk.Entry(sidebar, textvariable=self.query_var)
        self.query_entry.pack(fill="x", padx=14, pady=4)
        self.query_entry.bind("<Return>", lambda e: self.run_retrieval_ranking())
        tk.Label(sidebar, text="Term: 'patient'\nPhrase: 'patient care'\nAND: 'a AND b'\nAND NOT: 'a AND NOT b'", bg=CARD, fg=MUTED, font=("Segoe UI", 8), justify="left").pack(anchor="w", padx=14, pady=4)

        ttk.Separator(sidebar, orient="horizontal").pack(fill="x", padx=14, pady=8)

        self.btn_positional = ttk.Button(sidebar, text="positional_index", style="IR.TButton", command=self.run_positional_index)
        self.btn_positional.pack(fill="x", padx=14, pady=4)
        self.btn_retrieval = ttk.Button(sidebar, text="retrieval_ranking", style="IR.TButton", command=self.run_retrieval_ranking)
        self.btn_retrieval.pack(fill="x", padx=14, pady=4)
        self.btn_clear = ttk.Button(sidebar, text="Clear output", command=self.clear_output)
        self.btn_clear.pack(fill="x", padx=14, pady=4)
        self.progress = ttk.Progressbar(sidebar, mode="indeterminate", length=200)

        self.notebook = ttk.Notebook(body)
        self.notebook.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        self.tab_positional = tk.Frame(self.notebook, bg=CARD)
        self.tab_tf = tk.Frame(self.notebook, bg=CARD)
        self.tab_weighted = tk.Frame(self.notebook, bg=CARD)
        self.tab_dfidf = tk.Frame(self.notebook, bg=CARD)
        self.tab_tfidf = tk.Frame(self.notebook, bg=CARD)
        self.tab_norm = tk.Frame(self.notebook, bg=CARD)
        self.tab_vectors = tk.Frame(self.notebook, bg=CARD)
        self.tab_query = tk.Frame(self.notebook, bg=CARD)
        self.tab_log = tk.Frame(self.notebook, bg=CARD)

        self.notebook.add(self.tab_positional, text="Positional Index")
        self.notebook.add(self.tab_tf, text="Term Frequency (TF)")
        self.notebook.add(self.tab_weighted, text="Weighted TF")
        self.notebook.add(self.tab_dfidf, text="DF / IDF")
        self.notebook.add(self.tab_tfidf, text="TF-IDF")
        self.notebook.add(self.tab_norm, text="Normalized TF-IDF")
        self.notebook.add(self.tab_vectors, text="Document Vectors")
        self.notebook.add(self.tab_query, text="Query Results")
        self.notebook.add(self.tab_log, text="Log")

        self.tree_positional = make_tree(self.tab_positional)
        self.tree_tf = make_tree(self.tab_tf)
        self.tree_weighted = make_tree(self.tab_weighted)
        self.tree_dfidf = make_tree(self.tab_dfidf)
        self.tree_tfidf = make_tree(self.tab_tfidf)
        self.tree_norm = make_tree(self.tab_norm)
        self.tree_vectors = make_tree(self.tab_vectors)
        self.tree_query = make_tree(self.tab_query)
        self.log = tk.Text(self.tab_log, wrap="word", bg=CARD, fg=FG, insertbackground=FG, font=("Consolas", 9), borderwidth=0)
        self.log.pack(fill="both", expand=True, padx=8, pady=8)

    def _add_labeled_entry(self, parent, label, var, btn_text, command, pad):
        tk.Label(parent, text=label, bg=CARD, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w", padx=14, pady=(4, 0))
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", padx=14, pady=4)
        ent = ttk.Entry(row, textvariable=var)
        ent.pack(side="left", fill="x", expand=True)
        ttk.Button(row, text=btn_text, width=3, command=command).pack(side="left", padx=(4, 0))

    def select_dataset(self):
        folder = filedialog.askdirectory(title="Select dataset folder (.txt files)")
        if folder:
            self.data_dir_var.set(folder)
            self.data_dir = folder

    def select_output(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.output_dir_var.set(folder)
            self.output_dir = folder

    def clear_output(self):
        self.log.delete("1.0", "end")

    def append_output(self, text):
        if threading.current_thread() is threading.main_thread():
            self._append(text)
        else:
            self.after(0, self._append, text)

    def _append(self, text):
        self.log.insert("end", text)
        self.log.see("end")

    def safe_set_table(self, tree, columns, rows):
        def apply():
            set_table(tree, columns, rows)
        if threading.current_thread() is threading.main_thread():
            apply()
        else:
            self.after(0, apply)

    def set_busy(self, busy):
        def apply():
            if busy:
                self.progress.pack(fill="x", padx=14, pady=8)
                self.progress.start(10)
                self.btn_positional.state(["disabled"])
                self.btn_retrieval.state(["disabled"])
            else:
                self.progress.stop()
                self.progress.pack_forget()
                self.btn_positional.state(["!disabled"])
                self.btn_retrieval.state(["!disabled"])
        if threading.current_thread() is threading.main_thread():
            apply()
        else:
            self.after(0, apply)

    def show_error_async(self, message):
        self.after(0, lambda: messagebox.showerror("Error", message))

    def show_warning_async(self, message):
        self.after(0, lambda: messagebox.showwarning("Warning", message))

    def get_spark(self):
        if self.spark is None:
            os.environ["PYSPARK_PYTHON"] = sys.executable
            os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
            os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
            existing_pp = os.environ.get("PYTHONPATH", "")
            os.environ["PYTHONPATH"] = os.pathsep.join([RETRIEVAL_DIR, SRC_DIR] + ([existing_pp] if existing_pp else []))
            from pyspark.sql import SparkSession
            self.spark = (
                SparkSession.builder
                .appName("IR_GUI")
                .master("local[*]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.driver.bindAddress", "127.0.0.1")
                .config("spark.bindAddress", "127.0.0.1")
                .config("spark.python.worker.reuse", "true")
                .config("spark.python.worker.faulthandler.enabled", "true")
                .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
                .config("spark.ui.enabled", "false")
                .getOrCreate()
            )
            self.spark.sparkContext.setLogLevel("WARN")
        return self.spark

    def build_index(self, data_dir, output_dir):
        file_pairs = []
        for fname in sorted(os.listdir(data_dir)):
            if fname.endswith(".txt"):
                fpath = os.path.join(data_dir, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                file_pairs.append((fpath, content))
        spark = self.get_spark()
        sc = spark.sparkContext
        files_rdd = sc.parallelize(file_pairs, 1)
        term_positions = files_rdd.flatMap(lambda x: tokenize_with_positions(x[1], extract_doc_id(x[0])))
        grouped = term_positions.groupByKey()
        results = grouped.map(lambda x: (x[0], aggregate_doc_positions(list(x[1])))).collect()
        results.sort(key=lambda x: x[0])
        index_dict = {}
        for term, doc_positions in results:
            index_dict[term] = {doc_id: doc_positions[doc_id] for doc_id in sort_doc_ids(doc_positions.keys())}
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "positional_index.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(index_dict, f, indent=2, ensure_ascii=False)
        txt_path = os.path.join(output_dir, "positional_index.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Positional Index\nTerm Document IDs and 1-based positions\n")
            for term, doc_positions in results:
                f.write(format_human_readable(term, doc_positions) + "\n")
        return index_dict, len(file_pairs)

    def populate_positional_table(self, index_dict):
        rows = []
        for term in sorted(index_dict.keys()):
            docs = index_dict[term]
            for doc_id in sort_doc_ids(docs.keys()):
                pos_str = ", ".join(str(p) for p in docs[doc_id])
                rows.append([term, doc_id, pos_str])
        self.safe_set_table(self.tree_positional, ["Term", "Document", "Positions"], rows)

    def run_positional_index(self):
        data_dir = self.data_dir_var.get().strip()
        if not data_dir or not os.path.isdir(data_dir):
            self.show_warning_async("Please select a dataset folder containing .txt files.")
            return
        output_dir = self.output_dir_var.get().strip() or self.output_dir
        self.set_busy(True)
        self.append_output(f"\n=== positional_index: {data_dir} ===\n")

        def worker():
            try:
                index_dict, doc_count = self.build_index(data_dir, output_dir)
                self.append_output(f"Documents found: {doc_count}\nVocabulary size: {len(index_dict)}\n")
                self.append_output(f"Wrote {output_dir}/positional_index.json and .txt\n")
                self.populate_positional_table(index_dict)
                self.after(0, self.notebook.select, self.tab_positional)
            except Exception as e:
                self.show_error_async(str(e))
                self.append_output(f"\nERROR: {e}\n")
            finally:
                self.set_busy(False)

        threading.Thread(target=worker, daemon=True).start()

    def run_retrieval_ranking(self):
        output_dir = self.output_dir_var.get().strip() or self.output_dir
        json_path = os.path.join(output_dir, "positional_index.json")
        if not os.path.isfile(json_path):
            self.show_warning_async("Run positional_index first to create positional_index.json.")
            return
        self.set_busy(True)
        self.append_output("\n=== retrieval_ranking ===\n")

        def worker():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    index_dict = json.load(f)
                self.populate_positional_table(index_dict)
                self.compute_and_query(index_dict, output_dir)
            except Exception as e:
                self.show_error_async(str(e))
                self.append_output(f"\nERROR: {e}\n")
            finally:
                self.set_busy(False)

        threading.Thread(target=worker, daemon=True).start()

    def compute_and_query(self, index_dict, output_dir):
        spark = self.get_spark()
        doc_ids_set = set()
        for docs in index_dict.values():
            doc_ids_set.update(docs.keys())
        doc_ids = sort_doc_ids(list(doc_ids_set))
        N = len(doc_ids)
        data = compute_all(spark, index_dict, doc_ids, N)
        df_pos = data["df_pos"]
        idf_map = data["idf_map"]
        norm_matrix = data["norm_matrix"]
        terms = data["terms"]

        self.populate_matrices(data, doc_ids, N)

        mode = self.mode_var.get()
        query = self.query_var.get().strip().lower()
        q_result = self.build_query_result(mode, query, df_pos, idf_map, norm_matrix, doc_ids)
        self.safe_set_table(self.tree_query, q_result["columns"], q_result["rows"])
        self.append_output(f"[INFO] N={N}  terms={len(terms)}  mode={mode}  query='{query}'\n")
        if q_result["log"]:
            self.append_output(q_result["log"])
        self.append_output("\n=== DONE ===\n")

    def populate_matrices(self, data, doc_ids, N):
        terms = data["terms"]
        cols = ["Term"] + doc_ids
        tf_rows = [[t] + [data["tf_matrix"][t][d] for d in doc_ids] for t in terms]
        self.safe_set_table(self.tree_tf, cols, tf_rows)
        wtf_rows = [[t] + [fmt4(data["weighted_tf_matrix"][t][d]) for d in doc_ids] for t in terms]
        self.safe_set_table(self.tree_weighted, cols, wtf_rows)
        tfidf_rows = [[t] + [fmt4(data["tfidf_matrix"][t][d]) for d in doc_ids] for t in terms]
        self.safe_set_table(self.tree_tfidf, cols, tfidf_rows)
        norm_rows = [[t] + [fmt4(data["norm_matrix"][t][d]) for d in doc_ids] for t in terms]
        self.safe_set_table(self.tree_norm, cols, norm_rows)
        dfidf_rows = [[t, data["df_map"][t], fmt4(data["idf_map"][t])] for t in terms]
        self.safe_set_table(self.tree_dfidf, ["Term", "DF", f"IDF = log10({N}/DF)"], dfidf_rows)
        vec_rows = [[d, fmt4(data["doc_lengths"][d])] for d in doc_ids]
        self.safe_set_table(self.tree_vectors, ["Document", "Vector length"], vec_rows)

    def build_query_result(self, mode, query, df_pos, idf_map, norm_matrix, doc_ids):
        if not query:
            return {"columns": ["Message"], "rows": [["No query entered."]], "log": ""}
        lower = query
        is_and_not = " and not " in lower
        is_and = " and " in lower
        if mode == "Auto":
            if is_and_not:
                mode = "AND NOT"
            elif is_and:
                mode = "AND"
            elif " " in lower:
                mode = "Phrase"
            else:
                mode = "Term"
        if mode == "AND NOT":
            parts = lower.split(" and not ", 1)
            p1 = parts[0].strip().split()
            p2 = parts[1].strip().split() if len(parts) > 1 else []
            p1_docs = find_phrase_positions(df_pos, p1)
            p2_docs = find_phrase_positions(df_pos, p2)
            matched = sorted(set(p1_docs.keys()) - set(p2_docs.keys()))
            q_terms = p1
            kind = "Boolean AND NOT"
            phrase_info = self._phrase_rows(p1_docs, p1) + self._phrase_rows(p2_docs, p2)
        elif mode == "AND":
            parts = lower.split(" and ", 1)
            p1 = parts[0].strip().split()
            p2 = parts[1].strip().split() if len(parts) > 1 else []
            p1_docs = find_phrase_positions(df_pos, p1)
            p2_docs = find_phrase_positions(df_pos, p2)
            matched = sorted(set(p1_docs.keys()) & set(p2_docs.keys()))
            q_terms = p1 + p2
            kind = "Boolean AND"
            phrase_info = self._phrase_rows(p1_docs, p1) + self._phrase_rows(p2_docs, p2)
        elif mode == "Phrase":
            term_list = lower.split()
            phrase_docs = find_phrase_positions(df_pos, term_list)
            matched = sorted(phrase_docs.keys())
            q_terms = term_list
            kind = "Phrase Query"
            phrase_info = self._phrase_rows(phrase_docs, term_list)
        else:
            term_docs = find_term_docs(df_pos, lower)
            matched = sorted(term_docs.keys())
            q_terms = [lower]
            kind = "Single Term Search"
            phrase_info = [[d, "term occurrence", ", ".join(str(p) for p in sorted(term_docs[d]))] for d in matched]
        q_norm, q_len = build_query_vector(q_terms, idf_map)
        sims = compute_similarity(q_terms, q_norm, norm_matrix, matched)
        ranked = sorted(sims.items(), key=lambda x: -x[1])
        rows = [[d, fmt4(sims[d])] for d, _ in ranked]
        if not rows:
            rows = [["No matching documents", "0.0000"]]
        ranked_str = " > ".join(d for d, _ in ranked) if ranked else "(none)"
        log = f"[{kind}] query='{query}'\nQuery terms: {', '.join(q_terms)}\nQuery vector length = {fmt4(q_len)}\nMatched documents: {matched}\nRanking (highest first): {ranked_str}\n"
        columns = ["Matched document", "Similarity"]
        if phrase_info:
            phrase_log = "\n".join(f"  {r[0]} | {r[1]} | {r[2]}" for r in phrase_info)
            log = log + "\nPositions:\n" + phrase_log + "\n"
        return {"columns": columns, "rows": rows, "log": log}

    def _phrase_rows(self, phrase_docs, term_list):
        rows = []
        n = len(term_list)
        for d in sorted(phrase_docs.keys()):
            starts = phrase_docs[d]
            ranges = ", ".join(f"{s}-{s+n-1}" for s in starts)
            starts_str = ", ".join(str(s) for s in starts)
            rows.append([d, starts_str, ranges])
        return rows


def main():
    app = IRApp()
    app.mainloop()


if __name__ == "__main__":
    main()
