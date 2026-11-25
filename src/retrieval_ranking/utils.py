import os


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


def write_table_file(path, title, headers, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"=== {title} ===\n")
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        f.write(header_line + "\n")
        f.write("-" * len(header_line) + "\n")
        for row in rows:
            f.write(" | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + "\n")
        f.write("\n")


def append_table_file(path, title, headers, rows):
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"=== {title} ===\n")
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        f.write(header_line + "\n")
        f.write("-" * len(header_line) + "\n")
        for row in rows:
            f.write(" | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + "\n")
        f.write("\n")


def sort_doc_ids(doc_ids):
    return sorted(doc_ids, key=lambda x: int(x[1:]))