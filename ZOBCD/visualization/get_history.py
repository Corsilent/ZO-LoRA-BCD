import wandb

api = wandb.Api()
run = api.run("362557272-east-china-normal-university/opt-1-3b-sst2/ndij0tc3")

# 提取历史数据
history = run.history()

# 打印所有可用的键（指标名称）
with open("history.txt", "w", encoding="utf-8") as f:
    f.write(str(history.columns.tolist()))
    
print("所有history信息已保存到 history.txt")