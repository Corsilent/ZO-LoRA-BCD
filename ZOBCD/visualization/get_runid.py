import wandb

api = wandb.Api()

# 获取项目下的所有运行
runs = api.runs(path="362557272-east-china-normal-university/llama3-8b-WSC")


with open("runid.txt", "w", encoding="utf-8") as f:
    for run in runs:
        f.write(f"Run Name: {run.name}, ID: {run.id}\n")  # \n换行
        print(f"已记录: {run.name} ({run.id})")  # 控制台输出确认

print("所有run信息已保存到 runid.txt")