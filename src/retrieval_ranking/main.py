import os
from pyspark.sql import SparkSession
from utils import fmt4, print_table, write_table_file
from data_processor import load_index, compute_all
from query_engine import run_single_term_query, run_phrase_query, run_boolean_and, run_boolean_and_not


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

    json_path = os.path.join(output_dir, "positional_index.json")
    index_dict, doc_ids, N = load_index(json_path)

    data = compute_all(spark, index_dict, doc_ids, N)

    df_pos = data["df_pos"]
    terms = data["terms"]
    tf_matrix = data["tf_matrix"]
    weighted_tf_matrix = data["weighted_tf_matrix"]
    tfidf_matrix = data["tfidf_matrix"]
    df_map = data["df_map"]
    idf_map = data["idf_map"]
    doc_lengths = data["doc_lengths"]
    norm_matrix = data["norm_matrix"]

    tf_rows_print = []
    for t in terms:
        tf_rows_print.append([t] + [tf_matrix[t][d] for d in doc_ids])
    print_table("Raw TF Matrix", ["Term"] + doc_ids, tf_rows_print)
    write_table_file(os.path.join(output_dir, "tf_matrix.txt"), "Raw TF Matrix", ["Term"] + doc_ids, tf_rows_print)

    wtf_rows_print = []
    for t in terms:
        wtf_rows_print.append([t] + [fmt4(weighted_tf_matrix[t][d]) for d in doc_ids])
    print_table("Weighted TF Matrix (1 + log10(TF))", ["Term"] + doc_ids, wtf_rows_print)
    write_table_file(os.path.join(output_dir, "weighted_tf_matrix.txt"), "Weighted TF Matrix", ["Term"] + doc_ids, wtf_rows_print)

    dfidf_rows_print = []
    for t in terms:
        dfidf_rows_print.append([t, df_map[t], fmt4(idf_map[t])])
    print_table("DF and IDF", ["Term", "DF", f"IDF = log10({N}/DF)"], dfidf_rows_print)
    write_table_file(os.path.join(output_dir, "df_idf.txt"), "DF and IDF", ["Term", "DF", f"IDF = log10({N}/DF)"], dfidf_rows_print)

    tfidf_rows_print = []
    for t in terms:
        tfidf_rows_print.append([t] + [fmt4(tfidf_matrix[t][d]) for d in doc_ids])
    print_table("TF-IDF Matrix", ["Term"] + doc_ids, tfidf_rows_print)
    write_table_file(os.path.join(output_dir, "tfidf_matrix.txt"), "TF-IDF Matrix", ["Term"] + doc_ids, tfidf_rows_print)

    dl_rows_print = [[d, fmt4(doc_lengths[d])] for d in doc_ids]
    print_table("Document Vector Lengths", ["Document", "Vector length"], dl_rows_print)
    write_table_file(os.path.join(output_dir, "doc_lengths.txt"), "Document Vector Lengths", ["Document", "Vector length"], dl_rows_print)

    norm_rows_print = []
    for t in terms:
        norm_rows_print.append([t] + [fmt4(norm_matrix[t][d]) for d in doc_ids])
    print_table("Normalized TF-IDF Matrix", ["Term"] + doc_ids, norm_rows_print)
    write_table_file(os.path.join(output_dir, "normalized_tfidf_matrix.txt"), "Normalized TF-IDF Matrix", ["Term"] + doc_ids, norm_rows_print)

    while True:
        user_input = input("\nEnter query (or 'quit' to exit): ").strip()
        if user_input.lower() == "quit":
            break
        if not user_input:
            continue
        lower_input = user_input.lower()
        if " and not " in lower_input:
            parts = lower_input.split(" and not ")
            phrase1 = parts[0].strip().split()
            phrase2 = parts[1].strip().split()
            run_boolean_and_not(phrase1, phrase2, df_pos, idf_map, norm_matrix, doc_ids, output_dir)
        elif " and " in lower_input:
            parts = lower_input.split(" and ")
            phrase1 = parts[0].strip().split()
            phrase2 = parts[1].strip().split()
            run_boolean_and(phrase1, phrase2, df_pos, idf_map, norm_matrix, doc_ids, output_dir)
        elif " " in lower_input:
            term_list = lower_input.split()
            run_phrase_query(term_list, df_pos, idf_map, norm_matrix, doc_ids, output_dir)
        else:
            run_single_term_query(lower_input, df_pos, idf_map, norm_matrix, doc_ids, output_dir)

    spark.stop()


if __name__ == "__main__":
    main()