import os
import json
from pyspark import SparkContext, SparkConf


def extract_doc_id(file_path):
    filename = file_path.replace("\\", "/").split("/")[-1]
    if filename.endswith(".txt"):
        filename = filename[:-4]
    return filename


def tokenize_with_positions(content, doc_id):
    tokens = content.lower().split()
    return [(term, (doc_id, idx + 1)) for idx, term in enumerate(tokens)]


def aggregate_doc_positions(items):
    doc_positions = {}
    for doc_id, pos in items:
        doc_positions.setdefault(doc_id, []).append(pos)
    for doc_id in doc_positions:
        doc_positions[doc_id].sort()
    return doc_positions


def sort_doc_ids(doc_ids):
    return sorted(doc_ids, key=lambda x: int(x[1:]))


def format_human_readable(term, doc_positions):
    parts = []
    for doc_id in sort_doc_ids(doc_positions.keys()):
        positions = doc_positions[doc_id]
        pos_str = ", ".join(str(p) for p in positions)
        parts.append(f"{doc_id}: {pos_str}")
    return f"{term} " + "; ".join(parts)


def main():
    conf = (
        SparkConf()
        .setAppName("IR_Positional_Index")
        .setMaster("local[*]")
    )
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    data_dir = "data"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    file_pairs = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".txt"):
            fpath = os.path.join(data_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            file_pairs.append((fpath, content))
    files_rdd = sc.parallelize(file_pairs)

    doc_count = files_rdd.count()
    print(f"[INFO] Documents found: {doc_count}")

    term_positions = files_rdd.flatMap(
        lambda x: tokenize_with_positions(x[1], extract_doc_id(x[0]))
    )

    total_tokens = term_positions.count()
    print(f"[INFO] Total tokens indexed: {total_tokens}")

    grouped = term_positions.groupByKey()
    positional_index = grouped.map(
        lambda x: (x[0], aggregate_doc_positions(list(x[1])))
    )

    results = positional_index.collect()
    results.sort(key=lambda x: x[0])

    vocab_size = len(results)
    print(f"[INFO] Vocabulary size (unique terms): {vocab_size}")

    txt_path = os.path.join(output_dir, "positional_index.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Positional Index\n")
        f.write("Term Document IDs and 1-based positions\n")
        for term, doc_positions in results:
            f.write(format_human_readable(term, doc_positions) + "\n")
    print(f"[INFO] Wrote {txt_path}")

    json_path = os.path.join(output_dir, "positional_index.json")
    index_dict = {}
    for term, doc_positions in results:
        index_dict[term] = {
            doc_id: doc_positions[doc_id]
            for doc_id in sort_doc_ids(doc_positions.keys())
        }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(index_dict, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Wrote {json_path}")

    print("\n--- Positional Index Preview (first 15 terms) ---")
    for term, doc_positions in results[:15]:
        print(format_human_readable(term, doc_positions))
    if len(results) > 15:
        print(f"... ({len(results) - 15} more terms)")

    sc.stop()


if __name__ == "__main__":
    main()