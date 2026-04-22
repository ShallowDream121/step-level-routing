import numpy as np

confidences = []

with open('./confidence/confidence_log_blocks_1024_aime.csv', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) >= 2:
            try:
                if int(parts[3])>=2:
                    conf_val = float(parts[1])
                    confidences.append(conf_val)
            except ValueError:
                continue


conf_array = np.array(confidences)

percentiles = [5, 10, 20, 30, 50]

print(f"{'百分位':<10} | {'置信度阈值':<15}")
print("-" * 30)

for p in percentiles:

    val = np.percentile(conf_array, p)
    print(f"{p}%\t\t | {val:.5f}")

print("-" * 30)
print(f"总数据行数: {len(confidences)}")