# MDC 调度实验

本项目研究移动决策中心（MDC）对 Worker、UAV 和 Car 的协同任务调度。
算法代码位于 `code/`，数据集位于 `data/Mat/`，Python 环境由 uv 管理。

## DeepSeek 调度版本的配置与运行

请阅读 [CpOL-AsHs-DeepSeek-Skill 使用说明](code/CpOL-AsHs-DeepSeek-Skill/README.md)，其中包含：

- uv、Python 3.12 和项目依赖的安装步骤。
- 在 `MDC/.env` 中持久保存 `DEEPSEEK_API_KEY` 的方法及上传前清理方法。
- 在线启动、离线检查和测试命令。
- 数据集、仿真时长、调度间隔、MDC 覆盖范围和轮数的修改方式。
- 策略状态保存、重新开始实验和复现注意事项。

所有使用说明中的运行命令均在此 `MDC` 目录执行。
密钥填写在 `.env` 的 `DEEPSEEK_API_KEY=` 等号右侧；该文件不随 Git 提交，
如果下载后不存在，可自行创建。手动分享 `.env` 前必须清空密钥值。
