import wandb
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, ScalarFormatter
        
# 初始化wandb API
api = wandb.Api()

# 滑动平均函数
def moving_average(x, window_size):
    return np.convolve(x, np.ones(window_size) / window_size, mode='valid')


run_path_1 = "362557272-east-china-normal-university/llama3-8b-WSC/2i43c64j"
run1 = api.run(run_path_1)
history1 = run1.history(samples=40000)

# 提取数据
steps1 = history1["train/global_step"].values
train_loss1 = history1["loss"].interpolate().values

# 只取前一半的数据
half_length = len(steps1) // 2
steps1 = steps1[:half_length]
train_loss1 = train_loss1[:half_length]

# 应用滑动平均
window_size = 200  # 滑动窗口大小，可以根据需要调整
train_loss1_smooth = moving_average(train_loss1, window_size)
steps1_smooth = steps1[:len(train_loss1_smooth)]  # 调整步数以匹配平滑后的数据长度

# ------------------------------------------------------------------------------------

# run_path_2 = "362557272-east-china-normal-university/opt-1-3b-sst2/43pfntmq"  
# run2 = api.run(run_path_2)
# history2 = run2.history(samples=40000)

# steps2 = history2["train/global_step"].values
# train_loss2 = history2["var_A"].interpolate().values

# # 应用滑动平均
# train_loss2_smooth = moving_average(train_loss2, window_size)
# steps2_smooth = steps2[:len(train_loss2_smooth)]

# ------------------------------------------------------------------------------------

run_path_3 = "362557272-east-china-normal-university/llama3-8b-WSC/mwagcxna"  
run3 = api.run(run_path_3)
history3 = run3.history(samples=80000)

steps3 = history3["train/global_step"].values
train_loss3 = history3["loss"].interpolate().values

# 只取前一半的数据
half_length = len(steps3) // 2
steps3 = steps3[:half_length]
train_loss3 = train_loss3[:half_length]

# 应用滑动平均
train_loss3_smooth = moving_average(train_loss3, window_size)
steps3_smooth = steps3[:len(train_loss3_smooth)]

# 创建画布
plt.figure(figsize=(6, 6), dpi=100)  # 增大画布尺寸和分辨率

# 绘制红色曲线（调整线条样式）
plt.plot(
    steps1_smooth, 
    train_loss1_smooth, 
    color='darkred',         # 设置为红色
    linewidth=2,       # 加粗线条
    linestyle='-',       # 实线
    alpha=0.7,           # 透明度设置
    label='ZO-SGD'
)

# plt.plot(
#     steps2_smooth, 
#     train_loss2_smooth, 
#     color='green',        # 更换颜色
#     linewidth=2,
#     linestyle='-',      # 使用虚线区分
#     alpha=0.7,
#     label='FO-SGD'  # 修改图例标签
# )

plt.plot(
    steps3_smooth, 
    train_loss3_smooth, 
    color='darkblue',        # 更换颜色
    linewidth=2,
    linestyle='-',      # 使用虚线区分
    alpha=0.7,
    label='ZOBCD'  # 修改图例标签
)




# # 绘制红色曲线（调整线条样式）
# plt.plot(
#     steps1, 
#     train_loss1, 
#     color='darkred',         # 设置为红色
#     linewidth=2,       # 加粗线条
#     linestyle='-',       # 实线
#     alpha=0.7,           # 透明度设置
#     label='ZO-SGD'
# )

# plt.plot(
#     steps2, 
#     train_loss2, 
#     color='green',        # 更换颜色
#     linewidth=2,
#     linestyle='-',      # 使用虚线区分
#     alpha=0.7,
#     label='FO-SGD'  # 修改图例标签
# )

# plt.plot(
#     steps3, 
#     train_loss3, 
#     color='darkblue',        # 更换颜色
#     linewidth=2,
#     linestyle='-',      # 使用虚线区分
#     alpha=0.7,
#     label='ZOBCD'  # 修改图例标签
# )

# 图形修饰

# acc (0.53, 0.73)
# lora_a (0.53, 0.6)
# loss (0.53, 0.8)

# L2 Norm of LoRA-B
plt.xlabel("Training Steps", fontsize=18)
plt.ylabel("Training Loss", fontsize=18)
plt.title("WSC", fontsize=22, pad=20)
plt.legend(fontsize=18, loc=(0.56, 0.8))  # 图例位置调整)
plt.grid(True, linestyle='--', alpha=0.5)  # 虚线网格

plt.tick_params(axis='both', which='major', labelsize=18)  # 刻度字体放大

# plt.xticks([])  # 隐藏 x 轴的刻度

# ax = plt.gca()
# ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

# 自动调整坐标轴范围
plt.xlim(left=0)
# plt.ylim(bottom=0.4)

# 保存图片（先保存后显示）
plt.savefig(
    'wsc_loss.png',
    dpi=300,                   # 提高输出分辨率
    bbox_inches='tight',       # 去除白边
    facecolor='white'          # 背景设为白色
)

# 显示图形
# plt.show()