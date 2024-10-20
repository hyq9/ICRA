import pandas as pd

# 读取 Excel 文件
file_path = '虚假评论结果.xlsx'
df = pd.read_excel(file_path)

# 将“虚假评论标记”列中的“可能虚假”删除
df = df[df['虚假评论标记'] != '可能虚假']

# 删除“评论真实性”这一列
df = df.drop(columns=['评论真实性'])

# 保存处理后的数据到新的 Excel 文件
output_file_path = '处理后的虚假评论结果.xlsx'
df.to_excel(output_file_path, index=False)
