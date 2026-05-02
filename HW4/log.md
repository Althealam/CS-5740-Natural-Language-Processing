0. 只使用老师提供的data来微调模型，获得第一个best model
- W&B: https://wandb.ai/kl2235-cornell-university/safety-classifier/runs/kbjq9ryr?nw=nwuserkl2235

1. 使用LLM生成的1000条数据（短提问+缩写）+老师提供的数据微调模型，获得第二个best model
- W&B: https://wandb.ai/kl2235-cornell-university/safety-classifier/runs/kbjq9ryr?nw=nwuserkl2235
- 代码：generate_email_dataset_v1.py
- 生成结果：email_dataset_train_augmented_v1.jsonl, email_dataset_dev_augmented_v1.jsonl
- 结果：Accuracy=0.9924 , F1 Score=0.9923

2. 其次生成了jailbreak数据100条（混合语言，以及一些字符替换，详情可以参考generate_jailbreaks.py文件），并且和前面的数据混合在一起，获得了第二个best model，其W&B链接为https://wandb.ai/kl2235-cornell-university/safety-classifier/runs/9i6ifufl?nw=nwuserkl2235
- 生成结果：jailbreaks.jsonl
- 合并结果：email_dataset_dev_combined_v1.jsonl, email_dataset_train_combined_v1.jsonl
- tricks
    - spelling atack: ex3rcise, m0rning, w0rk out
    - multilingual attack
    - negation attack (this really trick the model)
    - implicit semantics (blood pumping, stretch legs)
    - specific time, like 9AM, 10AM

3. 继续分析当前模型的漏洞
- 下午/晚间仍然被判为unsafe: Let's meet in the afternoon, Quick call at 7:45 PM tmr?
- Morning当寒暄/工作句: Morning Bob-can you seed the slides by noon?
- 否定/拒绝: I hate yoga and I avoid it completely
- 无关安全任务的普通问句: How do I bake a birthday cake? 
- 中英混合短句: 明天一起run吗? 

4. 使用LLM生成1000条数据（针对上述的漏洞），并且合并所有的数据继续微调模型，获得第二个best model
- W&B: https://wandb.ai/kl2235-cornell-university/safety-classifier/runs/e848pnll?nw=nwuserkl2235
- 代码：generate_email_dataset_v2.py
- 生成结果：email_dataset_train_augmented_v2.jsonl, email_dataset_dev_augmented_v2.jsonl
- 合并结果：email_dataset_dev_combined_v2.jsonl, email_dataset_train_combined_v2.jsonl
- 结果：Accuracy=0.9799, F1 Score=0.9799

由于第二次微调的效果比第一次的差，所以我们的best model仍然上传第一次的model