# 基于置信度路由的大小模型协同推理框架

## 思路
version1.0 通过prompt使模型将推理分步骤并通过标签</step>分片，在推理过程中，根据</step>进行步骤置信度的计算，依据参数阈值，选择将接下来的推理任务路由到大小模型中。

version2.0 先将问题交给（大/小）模型，打出分步骤的大纲（草稿），再根据草稿将每一步进行路由与推理，置信度算法不变，第一步的路由策略可以尝试优化。

version3.0 将思维链划分为多个Block,通过计算Block的置信度进行路由。

## v1.0问题
模型能力太差，无法在步骤完成时生成</step>标识，使每一步的概念变成硬编码的2048tokens的长度块。

## v2.0问题
模型能力太差,无法根据草稿的步骤实现推理.

## 安装
`python -m pip install -r requirements.txt`

`git clone git@github.com:EleutherAI/lm-evaluation-harness.git`
将lm_tasks目录下文件粘贴至lm-evaluation-harness/lm_eval/tasks/目录.

## 数据集
lm_tasks所用自定义的AIME24_nofigures与GPQA_diamond

## 运行方法
在脚本中配置参数后运行
`bash runbash.sh`
`bash testbash.sh`