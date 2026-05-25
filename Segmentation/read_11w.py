import pandas as pd
print("正在读取文件...")
df = pd.read_excel('./古汉语词典及词频5以上bigram总结表v9.xlsx')
print("文件读取完成。")
print(df.head())

select_column = df['词条']
output_path = './dict_all.txt'
with open(output_path, 'w', encoding = 'utf-8') as f:
    i = 0
    for item in select_column:
        f.write(str(item) + '\n')
        i += 1
        if i%1000 == 0:
            print(f"已经写入{i}行\n")