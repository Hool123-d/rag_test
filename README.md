# 基于 DeepSeek + Qdrant Cloud 的书籍 RAG（支持增量更新）

这个项目是一个可落地的书籍问答 RAG 模板，包含你要求的增强能力：

- ✅ DeepSeek API 生成回答
- ✅ Qdrant Cloud Free Tier 在线向量数据库
- ✅ 增量更新（新增/改动文件只重建对应分块）
- ✅ BM25 + 向量混合检索
- ✅ Rerank 重排模型
- ✅ 自动去重，减少重复上下文，查询更快
- ✅ Web UI 聊天界面（Streamlit）
- ✅ **本地隔离运行环境（项目内 `.venv` + 本地模型缓存）**

## 1. 启动方式（已简化）

默认推荐使用一个脚本完成环境准备 + 启动：

```bash
./scripts/dev.sh web
```

脚本会自动完成：

1. 在项目内创建 `.venv`
2. 安装/更新 `requirements.txt`
3. 若不存在 `.env`，自动从 `.env.example` 生成
4. 将 Hugging Face 缓存设为项目本地 `.cache/huggingface`

也可切换模式：

```bash
./scripts/dev.sh ingest  # 增量入库
./scripts/dev.sh chat    # 命令行问答
./scripts/dev.sh web     # Web UI
```

## 2. 必填配置

第一次执行后请编辑 `.env`，至少填写：

- `DEEPSEEK_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY`

## 3. 架构

- Embedding：`BAAI/bge-small-zh-v1.5`
- Hybrid Retrieval：
  1) Qdrant 向量召回
  2) 本地 BM25 召回
  3) 分数融合（hybrid score）
  4) 文本去重
  5) Rerank（`BAAI/bge-reranker-base`）
- 生成：DeepSeek `deepseek-chat`

## 4. 数据准备

将书籍放入 `data/books/`，支持：`.pdf` / `.txt` / `.md`。

## 5. 免费线上部署建议

推荐 Hugging Face Spaces（免费）：

1. 建 Space（建议 Streamlit）
2. 上传仓库
3. 在 Secrets 配置 `DEEPSEEK_API_KEY` / `QDRANT_URL` / `QDRANT_API_KEY`
4. 启动即可

## 6. 目录

```text
.
├── scripts/
│   └── dev.sh           # 一键本地环境 + 启动
├── data/books/
├── docs/
│   └── 使用说明.md
├── src/
│   ├── rag_pipeline.py
│   ├── ingest.py
│   ├── chat.py
│   └── web_ui.py
├── .env.example
└── requirements.txt
```
