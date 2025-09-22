import sqlite3
import tiktoken

DB_PATH = "positions.db"
MODEL = "gpt-4"  # 可选: "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"

# 1. 加载 tokenizer
encoding = tiktoken.encoding_for_model(MODEL)

# 2. 读取数据库中的所有 full_text
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT full_text FROM npt_positions")
rows = cursor.fetchall()
conn.close()

# 3. 统计总 token 数
total_tokens = 0
for row in rows:
    text = row[0]
    tokens = encoding.encode(text)
    total_tokens += len(tokens)

# 4. 输出估算
print(f"Total token count for classification: {total_tokens:,}")

# 5. 粗略成本估算
USD_PER_1K = 0.01  # 例如 GPT-4-turbo input token 每千个 $0.01
estimated_cost = (total_tokens / 1000) * USD_PER_1K
print(f"Estimated cost: ${estimated_cost:.2f} for {MODEL}")
