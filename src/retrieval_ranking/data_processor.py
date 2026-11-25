import json
from pyspark.sql import functions as F
from pyspark.sql.functions.builtin import countDistinct
from utils import sort_doc_ids


def load_index(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        index_dict = json.load(f)
    doc_ids = set()
    for docs in index_dict.values():
        doc_ids.update(docs.keys())
    doc_ids = sort_doc_ids(list(doc_ids))
    N = len(doc_ids)
    return index_dict, doc_ids, N


def compute_all(spark, index_dict, doc_ids, N):
    rows = []
    for term, docs in index_dict.items():
        for doc_id, positions in docs.items():
            for pos in positions:
                rows.append((term, doc_id, pos))

    df_pos = spark.createDataFrame(rows, ["term", "doc_id", "position"])

    df_tf = df_pos.groupBy("term", "doc_id").agg(
        F.count("position").alias("raw_tf")
    )

    df_df_idf = df_tf.groupBy("term").agg(
        countDistinct("doc_id").alias("DF"),
        F.log10(F.lit(N) / countDistinct("doc_id")).alias("IDF")
    )

    df_tfidf = df_tf.join(df_df_idf, on="term", how="inner")
    df_tfidf = df_tfidf.withColumn(
        "weighted_tf",
        F.when(F.col("raw_tf") > 0, 1 + F.log10(F.col("raw_tf"))).otherwise(F.lit(0))
    )
    df_tfidf = df_tfidf.withColumn(
        "tfidf",
        F.greatest(F.col("weighted_tf") * F.col("IDF"), F.lit(0))
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

    df_doc_lengths = df_tfidf.groupBy("doc_id").agg(
        F.sqrt(F.sum(F.col("tfidf") ** 2)).alias("vector_length")
    ).orderBy("doc_id")

    doc_lengths = {r.doc_id: float(r.vector_length) for r in df_doc_lengths.collect()}

    df_norm = df_tfidf.join(df_doc_lengths, on="doc_id", how="inner")
    df_norm = df_norm.withColumn(
        "normalized_tfidf",
        F.when(F.col("vector_length") > 0, F.greatest(F.col("tfidf") / F.col("vector_length"), F.lit(0)))
        .otherwise(F.lit(0))
    )

    norm_rows = df_norm.select("term", "doc_id", "normalized_tfidf").collect()

    norm_matrix = {t: {d: 0.0 for d in doc_ids} for t in terms}
    for r in norm_rows:
        norm_matrix[r.term][r.doc_id] = float(r.normalized_tfidf)

    return {
        "df_pos": df_pos,
        "terms": terms,
        "tf_matrix": tf_matrix,
        "weighted_tf_matrix": weighted_tf_matrix,
        "tfidf_matrix": tfidf_matrix,
        "df_map": df_map,
        "idf_map": idf_map,
        "doc_lengths": doc_lengths,
        "norm_matrix": norm_matrix,
    }