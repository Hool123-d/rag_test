# 基于 DeepSeek + Qdrant Cloud 的书籍 RAG（支持增量更新）

这个项目是一个可落地的书籍问答 RAG 模板，包含你要求的增强能力：

- ✅ DeepSeek API 生成回答
- ✅ Qdrant Cloud Free Tier 在线向量数据库
- ✅ 增量更新（新增/改动文件只重建对应分块）
- ✅ **BM25 + 向量混合检索**
- ✅ **Rerank 重排模型**
- ✅ **自动去重，减少重复上下文，查询更快**
- ✅ **Web UI 聊天界面（Streamlit）**

## 1. 架构

- Embedding：`BAAI/bge-small-zh-v1.5`
- Hybrid Retrieval：
  1) Qdrant 向量召回  
  2) 本地 BM25 召回  
  3) 分数融合（hybrid_score）  
  4) 文本去重  
  5) Rerank（`BAAI/bge-reranker-base`）
- 生成：DeepSeek `deepseek-chat`

## 2. 快速开始

### 2.1 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2.2 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 填入：

- `DEEPSEEK_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY`

### 2.3 放入书籍

把书籍文件放到 `data/books/`，支持：`.pdf` / `.txt` / `.md`

### 2.4 执行增量入库

```bash
python src/ingest.py
```

### 2.5 命令行问答

```bash
python src/chat.py
```

### 2.6 Web UI 聊天

```bash
streamlit run src/web_ui.py
```

浏览器打开后即可问答，侧边栏支持手动触发增量入库。

## 3. 免费线上部署建议

推荐 Hugging Face Spaces（免费）：

1. 建 Space（建议 Streamlit）
2. 上传仓库
3. 在 Secrets 配置 `DEEPSEEK_API_KEY` / `QDRANT_URL` / `QDRANT_API_KEY`
4. 启动即可

## 4. 目录

```text
.
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
