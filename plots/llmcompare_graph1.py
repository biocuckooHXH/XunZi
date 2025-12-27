import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import itertools
import numpy as np

# 1. 读取数据
file_path = 'metrics_with_CTD_for_Disgenet.csv'  # 请确保文件名正确
df = pd.read_csv(file_path)

# 2. 准备颜色 (已修正由'O'到'0'的拼写错误)
custom_colors = [
    '#FC757B', '#F97F5F', '#FAA26F', '#FDCD94', '#FEE199', 
    '#B0D6A9', '#93DCB0', '#65BDBA', '#3C9BC9', '#CAD961', 
    '#83B756'  
]

# 3. 建立模型到颜色的映射
# 这样可以确保同一个模型在所有图中都是同一个颜色
model_col = df.columns[0] # 假设第一列是模型名
models = df[model_col].unique()

# 创建循环迭代器，防止模型数量超过颜色数量时报错
color_cycle = itertools.cycle(custom_colors)
model_color_map = {model: next(color_cycle) for model in models}

# 4. 创建保存文件夹
output_dir = 'metric_charts_custom'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 5. 设置绘图风格
sns.set_theme(style="white")  # 改为white，去掉网格线
plt.rcParams['font.family'] = 'Arial'  # 改为Arial字体
plt.rcParams['font.size'] = 14  # 增大基础字体
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 2  # 加粗坐标轴边框

# 6. 批量绘图
metrics = df.columns[1:] # 获取所有指标列

for metric in metrics:
    plt.figure(figsize=(8, 6), dpi=300) # 增加宽度，减少高度
    
    # 绘制柱状图
    ax = sns.barplot(
        data=df, 
        x=model_col, 
        y=metric, 
        palette=model_color_map, # 应用自定义颜色映射
        edgecolor='black',       # 添加黑色边框
        linewidth=1.5,           # 加粗边框到2.0
        width=0.7                # 柱子宽度更窄，间隔更大
    )

    # 标题和标签
    plt.ylabel(f'{metric}', fontsize=20, fontweight='bold')  # y轴使用指标名称
    plt.xlabel('') # 不需要X轴标签

    # 调整X轴标签角度和大小（防止重叠）
    plt.xticks(rotation=45, fontsize=15, fontweight='bold', ha='right')  # 右对齐，防止混乱
    plt.yticks(fontsize=20, fontweight='bold')

    # 设置Y轴范围：MCC需要显示负值，其他指标固定为0-1
    if 'MCC' in metric or 'Matthews' in metric:
        # MCC指标可能有负值，动态调整范围
        y_min = df[metric].min()
        y_max = df[metric].max()
        # 留出一些边距
        y_range = y_max - y_min
        plt.ylim(y_min - 0.1 * y_range, 1)
    else:
        # 其他指标固定范围0-1
        plt.ylim(0, 1)

    # 去除多余边框，只保留左边和底部
    sns.despine()
    
    # 如果是MCC指标，特殊处理
    if 'MCC' in metric or 'Matthews' in metric:
        # 将X轴移到y=0的位置
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['bottom'].set_linewidth(2)
        # Y轴只显示正半轴的轴线（从0到最大值）
        ax.spines['left'].set_bounds(0, ax.get_ylim()[1])
        # 只显示正值的Y轴刻度
        yticks = ax.get_yticks()
        # 过滤掉负值刻度
        positive_yticks = [tick for tick in yticks if tick >= 0]
        ax.set_yticks(positive_yticks)
        # 强制X轴标签保持在底部位置
        ax.xaxis.tick_bottom()  # 刻度标签显示在底部
        ax.xaxis.set_label_position('bottom')
        # 重新应用标签旋转和对齐，确保在spine移动后依然正确
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', 
                 fontsize=15, fontweight='bold')
    
    # 为x轴和y轴分别设置刻度线
    # x轴刻度：每个标签都有刻度
    # MCC指标需要更大的pad值避免被负值柱子遮挡
    x_pad = 30 if ('MCC' in metric or 'Matthews' in metric) else 10
    ax.tick_params(axis='x', which='major', 
                   width=2,           # 刻度线粗细
                   length=6,          # 刻度线长度
                   direction='out',   # 刻度线向外
                   bottom=True,       # 显示底部刻度
                   top=False,         # 不显示顶部刻度
                   labelsize=15,      # x轴刻度标签大小（增大）
                   pad=x_pad)         # 刻度标签与坐标轴的距离
    
    # y轴刻度：每个值都有刻度
    ax.tick_params(axis='y', which='major', 
                   width=2,           # 刻度线粗细
                   length=6,          # 刻度线长度
                   direction='out',   # 刻度线向外
                   left=True,         # 显示左侧刻度
                   right=False,       # 不显示右侧刻度
                   labelsize=20,      # y轴刻度标签大小（增大字体）
                   pad=8)             # 刻度标签与坐标轴的距离

    # 调整布局以适应所有元素
    plt.tight_layout()
    
    # 保存PNG格式
    save_path_png = os.path.join(output_dir, f'{metric}_colored.png')
    plt.savefig(save_path_png, dpi=300)
    print(f"已生成PNG: {save_path_png}")
    
    # 保存PDF格式
    save_path_pdf = os.path.join(output_dir, f'{metric}_colored.pdf')
    plt.savefig(save_path_pdf, format='pdf')
    print(f"已生成PDF: {save_path_pdf}")
    
    plt.close()

print("所有图表已使用新配色生成完毕！")

