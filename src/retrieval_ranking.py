import os
import json
import math
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions.builtin import countDistinct


def fmt4(value):
    return f"{value:.4f}"


def print_table(title, headers, rows):
    print(f"\n=== {title} ===")
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print(" | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)))


def main():
    spark = (
        SparkSession.builder
        .appName("IR_Retrieval_Ranking")
        .master("local[*]")
        .getOrCreate()
    )
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    N = 10
    doc_ids = [f"d{i}" for i in range(1, N + 1)]

    json_path = os.path.join(output_dir, "positional_index.json")
    with open(json_path, "r", encoding="utf-8") as f:
        index_dict = json.load(f)

    rows = []
    for term, docs in index_dict.items():
        for doc_id, positions in docs.items():
            for pos in positions:
                rows.append((term, doc_id, pos))

    df_pos = spark.createDataFrame(rows, ["term", "doc_id", "position"])
    print(f"\n[INFO] Positional index loaded: {df_pos.count()} postings")

    df_tf = df_pos.groupBy("term", "doc_id").agg(
        F.count("position").alias("raw_tf")
    )

    df_df_idf = df_tf.groupBy("term").agg(
        countDistinct("doc_id").alias("DF"),
        F.round(F.log10(F.lit(N) / countDistinct("doc_id")), 4).alias("IDF")
    )

    df_tfidf = df_tf.join(df_df_idf, on="term", how="inner")
    df_tfidf = df_tfidf.withColumn(
        "weighted_tf",
        F.when(F.col("raw_tf") > 0,
               F.round(1 + F.log10(F.col("raw_tf")), 4))
        .otherwise(F.lit(0))
    )
    df_tfidf = df_tfidf.withColumn(
        "tfidf",
        F.round(F.col("weighted_tf") * F.col("IDF"), 4)
    )

    tfidf_rows = df_tfidf.select(
        "term", "doc_id", "raw_tf", "weighted_tf", "tfidf", "DF", "IDF"
    ).collect()

    terms = sorted(set(r.term for r in tfidf_rows))

    tf_matrix = {t: {d: 0 for d in doc_ids} for t in terms}
    weighted_tf_matrix = {t: {d: 0.0 for d in doc_ids} for t in terms}
    tfidf_matrix = {t: {d: 0.0 for d in doc_ids} for t in terms}
    df_map = {}
    idf_map = {}

    for r in tfidf_rows:
        tf_matrix[r.term][r.doc_id] = r.raw_tf
        weighted_tf_matrix[r.term][r.doc_id] = float(r.weighted_tf)
        tfidf_matrix[r.term][r.doc_id] = float(r.tfidf)
        df_map[r.term] = r.DF
        idf_map[r.term] = float(r.IDF)

    tf_rows_print = []
    for t in terms:
        tf_rows_print.append([t] + [tf_matrix[t][d] for d in doc_ids])
    print_table("Raw TF Matrix", ["Term"] + doc_ids, tf_rows_print)

    wtf_rows_print = []
    for t in terms:
        wtf_rows_print.append([t] + [fmt4(weighted_tf_matrix[t][d]) for d in doc_ids])
    print_table("Weighted TF Matrix (1 + log10(TF))", ["Term"] + doc_ids, wtf_rows_print)

    dfidf_rows_print = []
    for t in terms:
        dfidf_rows_print.append([t, df_map[t], fmt4(idf_map[t])])
    print_table("DF and IDF", ["Term", "DF", "IDF = log10(10/DF)"], dfidf_rows_print)

    tfidf_rows_print = []
    for t in terms:
        tfidf_rows_print.append([t] + [fmt4(tfidf_matrix[t][d]) for d in doc_ids])
    print_table("TF-IDF Matrix", ["Term"] + doc_ids, tfidf_rows_print)

    df_doc_lengths = df_tfidf.groupBy("doc_id").agg(
        F.round(F.sqrt(F.sum(F.col("tfidf") ** 2)), 4).alias("vector_length")
    ).orderBy("doc_id")

    doc_lengths = {r.doc_id: float(r.vector_length) for r in df_doc_lengths.collect()}

    dl_rows_print = [[d, fmt4(doc_lengths[d])] for d in doc_ids]
    print_table("Document Vector Lengths", ["Document", "Vector length"], dl_rows_print)

    df_norm = df_tfidf.join(df_doc_lengths, on="doc_id", how="inner")
    df_norm = df_norm.withColumn(
        "normalized_tfidf",
        F.round(F.col("tfidf") / F.col("vector_length"), 4)
    )

    norm_rows = df_norm.select("term", "doc_id", "normalized_tfidf").collect()

    norm_matrix = {t: {d: 0.0 for d in doc_ids} for t in terms}
    for r in norm_rows:
        norm_matrix[r.term][r.doc_id] = float(r.normalized_tfidf)

    norm_rows_print = []
    for t in terms:
        norm_rows_print.append([t] + [fmt4(norm_matrix[t][d]) for d in doc_ids])
    print_table("Normalized TF-IDF Matrix", ["Term"] + doc_ids, norm_rows_print)

    def find_phrase_positions(term1, term2):
        df_phrase = df_pos.alias("a").join(
            df_pos.alias("b"),
            (F.col("a.doc_id") == F.col("b.doc_id")) &
            (F.col("b.position") == F.col("a.position") + 1) &
            (F.col("a.term") == term1) &
            (F.col("b.term") == term2)
        ).select(F.col("a.doc_id"), F.col("a.position").alias("start"))

        result = {}
        for r in df_phrase.collect():
            result.setdefault(r.doc_id, []).append(r.start)
        return result

    pc_docs = find_phrase_positions("patient", "care")
    pc_doc_set = set(pc_docs.keys())

    mr_docs = find_phrase_positions("medical", "research")
    mr_doc_set = set(mr_docs.keys())

    pc_print = []
    for d in sorted(pc_docs.keys()):
        starts = pc_docs[d]
        ranges = ", ".join(f"{s}-{s+1}" for s in starts)
        starts_str = ", ".join(str(s) for s in starts)
        pc_print.append([d, starts_str, ranges])
    print_table('Phrase Query: "patient care"',
                ["Matched document", "Starting positions", "Phrase token ranges"],
                pc_print)

    mr_print = []
    for d in sorted(mr_docs.keys()):
        starts = mr_docs[d]
        ranges = ", ".join(f"{s}-{s+1}" for s in starts)
        starts_str = ", ".join(str(s) for s in starts)
        mr_print.append([d, starts_str, ranges])
    print_table('Phrase Query: "medical research"',
                ["Matched document", "Starting positions", "Phrase token ranges"],
                mr_print)

    def build_query_vector(terms_list):
        q_tfidf = {t: 1.0 * idf_map.get(t, 0.0) for t in terms_list}
        q_len = math.sqrt(sum(v ** 2 for v in q_tfidf.values()))
        q_norm = {t: round(v / q_len, 4) for t, v in q_tfidf.items()}
        return q_norm, q_len

    def compute_similarity(query_terms, q_norm):
        results = {}
        for d in doc_ids:
            sim = 0.0
            for t in query_terms:
                sim += q_norm.get(t, 0.0) * norm_matrix.get(t, {}).get(d, 0.0)
            results[d] = round(sim, 4)
        return results

    q_pc, qlen_pc = build_query_vector(["patient", "care"])
    sim_pc = compute_similarity(["patient", "care"], q_pc)
    sim_pc_matched = {d: sim_pc[d] for d in sorted(pc_doc_set)}

    print(f'\n--- Similarity for phrase query: "patient care" ---')
    print(f'Query vector length = {fmt4(qlen_pc)}')
    qpc_rows = []
    for t in ["patient", "care"]:
        qpc_rows.append([t, fmt4(1.0), fmt4(idf_map[t]), fmt4(q_pc[t])])
    print_table("Query terms", ["Query term", "Weighted TF", "Query TF-IDF", "Normalized query weight"], qpc_rows)

    pc_sim_rows = [[d, fmt4(sim_pc_matched[d])] for d in sim_pc_matched]
    print_table("Matched document Similarity", ["Matched document", "Similarity"], pc_sim_rows)
    ranked_pc = sorted(sim_pc_matched.items(), key=lambda x: -x[1])
    ranked_str = " > ".join(d for d, _ in ranked_pc)
    print(f'Ranking (highest similarity first): {ranked_str}')

    q_mr, qlen_mr = build_query_vector(["medical", "research"])
    sim_mr = compute_similarity(["medical", "research"], q_mr)
    sim_mr_matched = {d: sim_mr[d] for d in sorted(mr_doc_set)}

    print(f'\n--- Similarity for phrase query: "medical research" ---')
    print(f'Query vector length = {fmt4(qlen_mr)}')
    qmr_rows = []
    for t in ["medical", "research"]:
        qmr_rows.append([t, fmt4(1.0), fmt4(idf_map[t]), fmt4(q_mr[t])])
    print_table("Query terms", ["Query term", "Weighted TF", "Query TF-IDF", "Normalized query weight"], qmr_rows)

    mr_sim_rows = [[d, fmt4(sim_mr_matched[d])] for d in sim_mr_matched]
    print_table("Matched document Similarity", ["Matched document", "Similarity"], mr_sim_rows)
    ranked_mr = sorted(sim_mr_matched.items(), key=lambda x: -x[1])
    ranked_str = " > ".join(d for d, _ in ranked_mr)
    print(f'Ranking (highest similarity first): {ranked_str}')

    and_docs = pc_doc_set & mr_doc_set
    and_not_docs = pc_doc_set - mr_doc_set

    q_and, qlen_and = build_query_vector(["patient", "care", "medical", "research"])
    sim_and = compute_similarity(["patient", "care", "medical", "research"], q_and)
    sim_and_matched = {d: sim_and[d] for d in sorted(and_docs)}

    print('\n--- Boolean AND: "patient care AND medical research" ---')
    print("Boolean AND query terms: patient, care, medical, research")
    print(f"Expected intersection (documents containing both phrases): {sorted(and_docs)}")
    qand_rows = []
    for t in ["patient", "care", "medical", "research"]:
        qand_rows.append([t, fmt4(1.0), fmt4(idf_map[t]), fmt4(q_and[t])])
    print_table("Query terms", ["Query term", "Weighted TF", "Query TF-IDF", "Normalized query weight"], qand_rows)
    print(f'Query vector length = {fmt4(qlen_and)}')
    and_sim_rows = [[d, fmt4(sim_and_matched[d])] for d in sim_and_matched]
    print_table("Matched document Similarity", ["Matched document", "Similarity"], and_sim_rows)
    ranked_and = sorted(sim_and_matched.items(), key=lambda x: -x[1])
    ranked_str = " > ".join(d for d, _ in ranked_and)
    print(f'Ranking (highest similarity first): {ranked_str}')

    q_andnot, qlen_andnot = build_query_vector(["patient", "care"])
    sim_andnot = compute_similarity(["patient", "care"], q_andnot)
    sim_andnot_matched = {d: sim_andnot[d] for d in sorted(and_not_docs)}

    print('\n--- Boolean AND NOT: "patient care AND NOT medical research" ---')
    print("Boolean AND NOT query terms: patient, care")
    print("Logic: documents matching the first phrase but not the second phrase")
    print(f"Expected result: {sorted(and_not_docs)}")
    qandnot_rows = []
    for t in ["patient", "care"]:
        qandnot_rows.append([t, fmt4(1.0), fmt4(idf_map[t]), fmt4(q_andnot[t])])
    print_table("Query terms", ["Query term", "Weighted TF", "Query TF-IDF", "Normalized query weight"], qandnot_rows)
    print(f'Query vector length = {fmt4(qlen_andnot)}')
    andnot_sim_rows = [[d, fmt4(sim_andnot_matched[d])] for d in sim_andnot_matched]
    print_table("Matched document Similarity", ["Matched document", "Similarity"], andnot_sim_rows)
    ranked_andnot = sorted(sim_andnot_matched.items(), key=lambda x: -x[1])
    ranked_str = " > ".join(d for d, _ in ranked_andnot)
    print(f'Ranking (highest similarity first): {ranked_str}')

    print("\n" + "=" * 60)
    print("VALIDATION AGAINST PDF REFERENCE OUTPUTS")
    print("=" * 60)

    expected_doc_lengths = {
        "d1": 0.4006, "d2": 1.6625, "d3": 1.2796, "d4": 1.1590,
        "d5": 0.9540, "d6": 1.7778, "d7": 2.0093, "d8": 2.0318,
        "d9": 1.2155, "d10": 2.4755
    }

    all_pass = True
    for d, expected in expected_doc_lengths.items():
        actual = doc_lengths.get(d, 0.0)
        if abs(actual - expected) > 0.001:
            print(f"[FAIL] Doc length {d}: expected {expected}, got {actual}")
            all_pass = False
        else:
            print(f"[OK]   Doc length {d}: {actual}")

    expected_sim_pc = {"d1": 1.0000, "d5": 0.5473, "d6": 0.2253}
    for d, expected in expected_sim_pc.items():
        actual = sim_pc.get(d, 0.0)
        if abs(actual - expected) > 0.001:
            print(f"[FAIL] Similarity 'patient care' {d}: expected {expected}, got {actual}")
            all_pass = False
        else:
            print(f"[OK]   Similarity 'patient care' {d}: {actual}")

    expected_sim_mr = {"d3": 0.4715, "d5": 0.6325, "d8": 0.2970}
    for d, expected in expected_sim_mr.items():
        actual = sim_mr.get(d, 0.0)
        if abs(actual - expected) > 0.001:
            print(f"[FAIL] Similarity 'medical research' {d}: expected {expected}, got {actual}")
            all_pass = False
        else:
            print(f"[OK]   Similarity 'medical research' {d}: {actual}")

    expected_sim_and = {"d5": 0.8296}
    for d, expected in expected_sim_and.items():
        actual = sim_and.get(d, 0.0)
        if abs(actual - expected) > 0.001:
            print(f"[FAIL] Similarity AND {d}: expected {expected}, got {actual}")
            all_pass = False
        else:
            print(f"[OK]   Similarity AND {d}: {actual}")

    expected_sim_andnot = {"d1": 1.0000, "d6": 0.2253}
    for d, expected in expected_sim_andnot.items():
        actual = sim_andnot.get(d, 0.0)
        if abs(actual - expected) > 0.001:
            print(f"[FAIL] Similarity AND NOT {d}: expected {expected}, got {actual}")
            all_pass = False
        else:
            print(f"[OK]   Similarity AND NOT {d}: {actual}")

    if all_pass:
        print("\n[RESULT] ALL VALIDATIONS PASSED")
    else:
        print("\n[RESULT] SOME VALIDATIONS FAILED - check values above")

    spark.stop()


if __name__ == "__main__":
    main()