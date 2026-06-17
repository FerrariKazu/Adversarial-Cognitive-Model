#!/usr/bin/env python3
"""Check parquet file structure."""
import pyarrow.parquet as pq

parquet_path = "/home/ferrarikazu/.cache/huggingface/hub/datasets--cifar10/snapshots/0b2714987fa478483af9968de7c934580d0bb9a2/plain_text/test-00000-of-00001.parquet"
table = pq.read_table(parquet_path)
print(f"Columns: {table.column_names}")
print(f"Rows: {table.num_rows}")
print(f"Schema: {table.schema}")
# Check first row
row = table.slice(0, 1)
for col in table.column_names:
    val = row.column(col).to_pylist()[0]
    print(f"  {col}: type={type(val).__name__}, value={str(val)[:100]}")
