import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import itertools
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# 1. 读取数据
file_path = 'Merge_ALL_5_fold_metric1.csv'  # 请确保文件名正确
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

# ===== 可调参数：x轴标签(刻度文字)字号 =====
# 说明：x轴标签大小会同时受 plt.xticks / ax.tick_params(labelsize) / (MCC分支)plt.setp 影响，
# 这里统一用变量控制，避免“看起来怎么改都不生效”。
X_TICK_FONTSIZE = 20
X_TICK_FONTSIZE_MCC = 20

# 6. 批量绘图
metrics = df.columns[1:] # 获取所有指标列

for metric in metrics:
    is_mcc = ('MCC' in metric) or ('Matthews' in metric)

    # ===== MCC：显示负值柱子，但“0~1正轴部分”与其它图完全一致，且不显示负半轴y坐标轴 =====
    if is_mcc:
        # 说明：
        # - 0~1 正区间使用与其它图相同的主坐标轴区域（因此长度一致）
        # - 负值柱子单独画在下方一个小子图里（占用原本为x标签预留的底部空白）
        # - 负值子图隐藏y轴（不显示负半轴的坐标轴/刻度）
        fig = plt.figure(figsize=(8, 6), dpi=300)

        # 主轴位置与其它图保持一致（对应原来的 subplots_adjust 参数）
        # left=0.20, right=0.95, top=0.95, bottom=0.35  => width=0.75, height=0.60
        ax_pos = fig.add_axes([0.20, 0.35, 0.75, 0.60])
        # 负值子图放在主轴正下方并严格贴合：
        # - top(负值子图)=bottom(主轴)=0.35，让y=0分割线与上半图底边框重合
        # - 下半图高度缩小，用“画布高度”压缩负值柱子的视觉长度（更短一些）
        #   y0 = 0.35 - height
        ax_neg = fig.add_axes([0.20, 0.23, 0.75, 0.12])

        # 画两次同一组柱子：上面显示0~1，下面显示负值区间
        sns.barplot(
            data=df,
            x=model_col,
            y=metric,
            order=models,
            palette=model_color_map,
            edgecolor='black',
            linewidth=1.5,
            width=0.7,
            ax=ax_pos
        )
        sns.barplot(
            data=df,
            x=model_col,
            y=metric,
            order=models,
            palette=model_color_map,
            edgecolor='black',
            linewidth=1.5,
            width=0.7,
            ax=ax_neg
        )

        # 确保两幅子图的x范围完全一致（不sharex时显式同步）
        ax_neg.set_xlim(ax_pos.get_xlim())

        # ---- 主轴：只显示0~1（正半轴部分与其它指标一致） ----
        ax_pos.set_ylim(0, 1)
        ax_pos.set_ylabel(f'{metric}', fontsize=30, fontweight='bold')
        ax_pos.set_xlabel('')

        # y轴刻度（主轴）：与其它图一致（字号/刻度线参数一致）
        # 其它图的最终效果由 tick_params(labelsize=25) 控制，因此这里也统一为25
        ax_pos.tick_params(axis='y', which='major',
                           width=2, length=6, direction='out',
                           left=True, right=False,
                           labelsize=25, pad=8)
        for lbl in ax_pos.get_yticklabels():
            lbl.set_fontweight('bold')

        # 去掉主轴上/右边框；
        # MCC 的 y=0 处会同时叠加“柱子底边框”+“坐标轴线”，视觉上会比其它图更粗。
        # 解决：隐藏 bottom spine，改用 axhline 画一条 linewidth=2 的 0 线放在最上层，
        # 这样最终看到的粗细就和其它图一致（不会被叠加加粗）。
        sns.despine(ax=ax_pos)
        ax_pos.spines['bottom'].set_visible(False)

        # 主轴：在0线上显示x刻度（与其它图配置一致），但不显示标签（标签放在底部负值子图）
        # 必须在 despine 和 spine 设置之后再配置 tick_params，确保刻度线会显示
        ax_pos.tick_params(axis='x', which='major',
                           width=2, length=6, direction='out',
                           bottom=True, top=False,
                           labelbottom=False, labeltop=False)

        # 画y=0分割线（作为唯一“x轴线”），放在最上层避免被柱子边框叠加显得更粗
        ax_pos.axhline(0, color='black', linewidth=2, zorder=10)

        # ---- 负值子图：只显示负值（柱子能画出来），但不显示y轴负半轴坐标轴 ----
        values = pd.to_numeric(df[metric], errors='coerce')
        y_min = float(np.nanmin(values))

        if y_min < 0:
            # 给负值留很小的边距（避免负值柱子看起来过长）
            y_min_pad = y_min - 0.02 * abs(y_min)
            ax_neg.set_ylim(y_min_pad, 0)
        else:
            # 没有负值时，不需要负值子图；让MCC退化为普通0~1图
            fig.delaxes(ax_neg)
            ax_pos.tick_params(axis='x', which='major',
                               width=2, length=6, direction='out',
                               bottom=True, top=False,
                               labelbottom=True,
                               labelsize=X_TICK_FONTSIZE_MCC,
                               pad=10)
            plt.setp(ax_pos.get_xticklabels(), rotation=45, ha='right',
                     fontsize=X_TICK_FONTSIZE_MCC, fontweight='bold')

            # 保存并关闭
            save_path_png = os.path.join(output_dir, f'{metric}_colored.png')
            fig.savefig(save_path_png, dpi=300)
            print(f"已生成PNG: {save_path_png}")

            save_path_pdf = os.path.join(output_dir, f'{metric}_colored.pdf')
            fig.savefig(save_path_pdf, format='pdf')
            print(f"已生成PDF: {save_path_pdf}")
            plt.close(fig)
            continue

        ax_neg.set_ylabel('')
        ax_neg.set_xlabel('')

        # 隐藏负半轴y坐标轴：不显示左侧轴线和刻度/标签
        ax_neg.yaxis.set_ticks([])
        ax_neg.spines['left'].set_visible(False)
        ax_neg.spines['right'].set_visible(False)
        # 负值子图：不画“第二条x轴线”（避免两套x轴）
        # 但需要每个标签都有刻度线：让bottom spine保持“存在但不可见”，刻度线仍从底部发出
        ax_neg.spines['bottom'].set_visible(True)
        ax_neg.spines['bottom'].set_linewidth(0.0)
        ax_neg.spines['bottom'].set_color((0, 0, 0, 0))  # 透明

        # 不在负值子图顶部再画y=0参考线，避免出现"两条x轴"；0线由主轴底边框提供
        ax_neg.spines['top'].set_visible(False)

        # x轴刻度线：在负值子图的"顶部"（也就是y=0位置）显示刻度线
        # 这样刻度线不会被子图遮挡，且与其它图配置一致
        ax_neg.tick_params(axis='x', which='major',
                           width=2, length=6, direction='out',
                           bottom=False, top=True,  # 刻度线在顶部（y=0）
                           labelbottom=True, labeltop=False)  # 标签在底部

        # x轴标签放在最底部（负值子图），并统一45°倾斜
        x_pad = 8
        ax_neg.xaxis.set_ticks_position('bottom')
        ax_neg.tick_params(axis='x', which='major',
                           labelsize=X_TICK_FONTSIZE_MCC,
                           pad=x_pad,
                           labelrotation=45)
        # 需要先draw一次，确保ticklabel对象已创建后再设置对齐/字重
        fig.canvas.draw()
        for lbl in ax_neg.get_xticklabels():
            lbl.set_horizontalalignment('right')
            lbl.set_fontsize(X_TICK_FONTSIZE_MCC)
            lbl.set_fontweight('bold')

        # 负值子图：不使用sns.despine，避免把我们手动保留的top/bottom spine改掉
        ax_neg.spines['right'].set_visible(False)

        # 保存
        save_path_png = os.path.join(output_dir, f'{metric}_colored.png')
        fig.savefig(save_path_png, dpi=300)
        print(f"已生成PNG: {save_path_png}")

        save_path_pdf = os.path.join(output_dir, f'{metric}_colored.pdf')
        fig.savefig(save_path_pdf, format='pdf')
        print(f"已生成PDF: {save_path_pdf}")

        plt.close(fig)
        continue

    # ===== 非MCC：保持原来的单轴画法 =====
    plt.figure(figsize=(8, 6), dpi=300)  # 增加宽度，减少高度

    ax = sns.barplot(
        data=df,
        x=model_col,
        y=metric,
        palette=model_color_map,
        edgecolor='black',
        linewidth=1.5,
        width=0.7
    )

    plt.ylabel(f'{metric}', fontsize=30, fontweight='bold')
    plt.xlabel('')

    plt.xticks(rotation=45, fontsize=X_TICK_FONTSIZE, fontweight='bold', ha='right')
    plt.yticks(fontsize=30, fontweight='bold')

    plt.ylim(0, 1)
    sns.despine()

    x_pad = 10
    ax.tick_params(axis='x', which='major',
                   width=2, length=6, direction='out',
                   bottom=True, top=False,
                   labelsize=X_TICK_FONTSIZE,
                   pad=x_pad)

    ax.tick_params(axis='y', which='major',
                   width=2, length=6, direction='out',
                   left=True, right=False,
                   labelsize=25, pad=8)

    plt.subplots_adjust(left=0.20, right=0.95, top=0.95, bottom=0.35)

    save_path_png = os.path.join(output_dir, f'{metric}_colored.png')
    plt.savefig(save_path_png, dpi=300)
    print(f"已生成PNG: {save_path_png}")

    save_path_pdf = os.path.join(output_dir, f'{metric}_colored.pdf')
    plt.savefig(save_path_pdf, format='pdf')
    print(f"已生成PDF: {save_path_pdf}")

    plt.close()

print("所有图表已使用新配色生成完毕！")

