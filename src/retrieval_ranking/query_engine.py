import math
import os
from pyspark.sql import functions as F
from utils import fmt4, print_table, write_table_file, append_table_file


def find_phrase_positions(df_pos, term_list):
    if len(term_list) < 2:
        return {}
    df_phrase = df_pos.alias("t0").filter(F.col("t0.term") == term_list[0]).select(F.col("t0.doc_id"), F.col("t0.position").alias("pos0"))
    for i in range(1, len(term_list)):
        df_next = df_pos.alias(f"t{i}").filter(F.col(f"t{i}.term") == term_list[i]).select(F.col(f"t{i}.doc_id"), F.col(f"t{i}.position").alias(f"pos{i}"))
        df_phrase = df_phrase.join(df_next, df_phrase.doc_id == df_next.doc_id)
        df_phrase = df_phrase.filter(F.col(f"pos{i}") == F.col(f"pos{i-1}") + 1)
    result = {}
    for r in df_phrase.select(F.col("t0.doc_id").alias("doc_id"), F.col("pos0")).distinct().collect():
        result.setdefault(r.doc_id, []).append(r.pos0)
    return result


def find_term_docs(df_pos, term):
    result = {}
    for r in df_pos.filter(F.col("term") == term).select("doc_id", "position").collect():
        result.setdefault(r.doc_id, []).append(r.position)
    return result


def build_query_vector(terms_list, idf_map):
    q_tfidf = {t: 1.0 * idf_map.get(t, 0.0) for t in terms_list}
    q_len = math.sqrt(sum(v ** 2 for v in q_tfidf.values()))
    q_norm = {t: round(v / q_len, 4) for t, v in q_tfidf.items()}
    return q_norm, q_len


def compute_similarity(query_terms, q_norm, norm_matrix, matched_docs):
    results = {}
    for d in matched_docs:
        sim = 0.0
        for t in query_terms:
            sim += q_norm.get(t, 0.0) * norm_matrix.get(t, {}).get(d, 0.0)
        results[d] = round(sim, 4)
    return results























def run_single_term_query(term, df_pos, idf_map, norm_matrix, doc_ids, output_dir):
    term_docs = find_term_docs(df_pos, term)
    matched = sorted(term_docs.keys())
    if not matched:
        print(f"\nNo documents found containing term: {term}")
        return
    q_terms = [term]
    q_norm, q_len = build_query_vector(q_terms, idf_map)
    sims = compute_similarity(q_terms, q_norm, norm_matrix, matched)
    print(f"\n--- Term Search: \"{term}\" ---")
    print(f"Query vector length = {fmt4(q_len)}")
    q_rows = [[term, fmt4(1.0), fmt4(idf_map.get(term, 0.0)), fmt4(q_norm.get(term, 0.0))]]
    print_table("Query terms", ["Query term", "Weighted TF", "Query TF-IDF", "Normalized query weight"], q_rows)
    sim_rows = [[d, fmt4(sims[d])] for d in matched]
    print_table("Matched document Similarity", ["Matched document", "Similarity"], sim_rows)
    ranked = sorted(sims.items(), key=lambda x: -x[1])
    ranked_str = " > ".join(d for d, _ in ranked)
    print(f"Ranking (highest similarity first): {ranked_str}")
    qpath = os.path.join(output_dir, f"query_single_{term}.txt")
    write_table_file(qpath, f'Term Search: "{term}"', ["Query term", "Weighted TF", "Query TF-IDF", "Normalized query weight"], q_rows)
    append_table_file(qpath, "Matched document Similarity", ["Matched document", "Similarity"], sim_rows)
    with open(qpath, "a", encoding="utf-8") as f:
        f.write(f"Ranking (highest similarity first): {ranked_str}\n")


def run_phrase_query(term_list, df_pos, idf_map, norm_matrix, doc_ids, output_dir):
    phrase_docs = find_phrase_positions(df_pos, term_list)
    matched = sorted(phrase_docs.keys())
    if not matched:
        print(f"\nNo documents found containing phrase: {' '.join(term_list)}")
        return
    q_terms = term_list
    q_norm, q_len = build_query_vector(q_terms, idf_map)
    sims = compute_similarity(q_terms, q_norm, norm_matrix, matched)
    phrase_str = " ".join(term_list)
    print(f"\n--- Phrase Query: \"{phrase_str}\" ---")
    print(f"Query vector length = {fmt4(q_len)}")
    q_rows = []
    for t in q_terms:
        q_rows.append([t, fmt4(1.0), fmt4(idf_map.get(t, 0.0)), fmt4(q_norm.get(t, 0.0))])
    print_table("Query terms", ["Query term", "Weighted TF", "Query TF-IDF", "Normalized query weight"], q_rows)
    pc_print = []
    for d in matched:
        starts = phrase_docs[d]
        ranges = ", ".join(f"{s}-{s+len(term_list)-1}" for s in starts)
        starts_str = ", ".join(str(s) for s in starts)
        pc_print.append([d, starts_str, ranges])
    print_table('Phrase Query Results', ["Matched document", "Starting positions", "Phrase token ranges"], pc_print)
    sim_rows = [[d, fmt4(sims[d])] for d in matched]
    print_table("Matched document Similarity", ["Matched document", "Similarity"], sim_rows)
    ranked = sorted(sims.items(), key=lambda x: -x[1])
    ranked_str = " > ".join(d for d, _ in ranked)
    print(f"Ranking (highest similarity first): {ranked_str}")
    qpath = os.path.join(output_dir, f"query_phrase_{'_'.join(term_list)}.txt")
    write_table_file(qpath, f'Phrase Query: "{phrase_str}"', ["Query term", "Weighted TF", "Query TF-IDF", "Normalized query weight"], q_rows)
    append_table_file(qpath, "Phrase Query Results", ["Matched document", "Starting positions", "Phrase token ranges"], pc_print)
    append_table_file(qpath, "Matched document Similarity", ["Matched document", "Similarity"], sim_rows)
    with open(qpath, "a", encoding="utf-8") as f:
        f.write(f"Ranking (highest similarity first): {ranked_str}\n")


def run_boolean_and(phrase1_terms, phrase2_terms, df_pos, idf_map, norm_matrix, doc_ids, output_dir):
    p1_docs = find_phrase_positions(df_pos, phrase1_terms)
    p2_docs = find_phrase_positions(df_pos, phrase2_terms)
    and_docs = set(p1_docs.keys()) & set(p2_docs.keys())
    matched = sorted(and_docs)
    if not matched:
        print(f"\nNo documents found for AND query")
        return
    q_terms = phrase1_terms + phrase2_terms
    q_norm, q_len = build_query_vector(q_terms, idf_map)
    sims = compute_similarity(q_terms, q_norm, norm_matrix, matched)
    phrase1_str = " ".join(phrase1_terms)
    phrase2_str = " ".join(phrase2_terms)
    print(f"\n--- Boolean AND: \"{phrase1_str} AND {phrase2_str}\" ---")
    print(f"Boolean AND query terms: {', '.join(q_terms)}")
    print(f"Documents containing both phrases: {matched}")
    q_rows = []
    for t in q_terms:
        q_rows.append([t, fmt4(1.0), fmt4(idf_map.get(t, 0.0)), fmt4(q_norm.get(t, 0.0))])
    print_table("Query terms", ["Query term", "Weighted TF", "Query TF-IDF", "Normalized query weight"], q_rows)
    print(f"Query vector length = {fmt4(q_len)}")
    sim_rows = [[d, fmt4(sims[d])] for d in matched]
    print_table("Matched document Similarity", ["Matched document", "Similarity"], sim_rows)
    ranked = sorted(sims.items(), key=lambda x: -x[1])
    ranked_str = " > ".join(d for d, _ in ranked)
    print(f"Ranking (highest similarity first): {ranked_str}")
    qpath = os.path.join(output_dir, f"query_and_{'_'.join(phrase1_terms)}_{'_'.join(phrase2_terms)}.txt")
    write_table_file(qpath, f'Boolean AND: "{phrase1_str} AND {phrase2_str}"', ["Query term", "Weighted TF", "Query TF-IDF", "Normalized query weight"], q_rows)
    append_table_file(qpath, "Matched document Similarity", ["Matched document", "Similarity"], sim_rows)
    with open(qpath, "a", encoding="utf-8") as f:
        f.write(f"Ranking (highest similarity first): {ranked_str}\n")


def run_boolean_and_not(phrase1_terms, phrase2_terms, df_pos, idf_map, norm_matrix, doc_ids, output_dir):
    p1_docs = find_phrase_positions(df_pos, phrase1_terms)
    p2_docs = find_phrase_positions(df_pos, phrase2_terms)
    andnot_docs = set(p1_docs.keys()) - set(p2_docs.keys())
    matched = sorted(andnot_docs)
    if not matched:
        print(f"\nNo documents found for AND NOT query")
        return
    q_terms = phrase1_terms
    q_norm, q_len = build_query_vector(q_terms, idf_map)
    sims = compute_similarity(q_terms, q_norm, norm_matrix, matched)
    phrase1_str = " ".join(phrase1_terms)
    phrase2_str = " ".join(phrase2_terms)
    print(f"\n--- Boolean AND NOT: \"{phrase1_str} AND NOT {phrase2_str}\" ---")
    print(f"Boolean AND NOT query terms: {', '.join(q_terms)}")
    print(f"Logic: documents matching the first phrase but not the second phrase")
    print(f"Expected result: {matched}")
    q_rows = []
    for t in q_terms:
        q_rows.append([t, fmt4(1.0), fmt4(idf_map.get(t, 0.0)), fmt4(q_norm.get(t, 0.0))])
    print_table("Query terms", ["Query term", "Weighted TF", "Query TF-IDF", "Normalized query weight"], q_rows)
    print(f"Query vector length = {fmt4(q_len)}")
    sim_rows = [[d, fmt4(sims[d])] for d in matched]
    print_table("Matched document Similarity", ["Matched document", "Similarity"], sim_rows)
    ranked = sorted(sims.items(), key=lambda x: -x[1])
    ranked_str = " > ".join(d for d, _ in ranked)
    print(f"Ranking (highest similarity first): {ranked_str}")
    qpath = os.path.join(output_dir, f"query_andnot_{'_'.join(phrase1_terms)}_{'_'.join(phrase2_terms)}.txt")
    write_table_file(qpath, f'Boolean AND NOT: "{phrase1_str} AND NOT {phrase2_str}"', ["Query term", "Weighted TF", "Query TF-IDF", "Normalized query weight"], q_rows)
    append_table_file(qpath, "Matched document Similarity", ["Matched document", "Similarity"], sim_rows)
    with open(qpath, "a", encoding="utf-8") as f:
        f.write(f"Ranking (highest similarity first): {ranked_str}\n")