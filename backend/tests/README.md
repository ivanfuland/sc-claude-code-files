# RAG Chatbot 测试套件

这个测试套件为RAG聊天机器人系统提供全面的测试覆盖，包括单元测试、集成测试和API端点测试。

## 测试结构

```
backend/tests/
├── __init__.py
├── conftest.py              # 测试配置和共享fixtures
├── test_ai_generator.py     # AI生成器单元测试
├── test_api_endpoints.py    # API端点测试 (新增)
├── test_rag_system.py       # RAG系统测试
├── test_search_tools.py     # 搜索工具测试
├── test_vector_store.py     # 向量存储测试
└── README.md               # 本文档
```

## 运行测试

### 所有测试
```bash
uv run pytest backend/tests/ -v
```

### 仅API测试
```bash
uv run pytest backend/tests/test_api_endpoints.py -v
```

### 按标记运行测试
```bash
# API测试
uv run pytest -m "api" -v

# 集成测试
uv run pytest -m "integration" -v

# 单元测试
uv run pytest -m "unit" -v

# 慢速测试
uv run pytest -m "slow" -v
```

### 排除慢速测试
```bash
uv run pytest -m "not slow" -v
```

## API测试功能

新的API测试套件 (`test_api_endpoints.py`) 包含以下测试类别：

### TestQueryEndpoint
测试 `/api/query` 端点的所有功能：
- ✅ 带会话ID的查询
- ✅ 不带会话ID的查询（自动创建）
- ✅ 无效请求处理
- ✅ 缺少必填字段
- ✅ 空字符串查询
- ✅ 服务器错误处理
- ✅ 响应模式验证
- ✅ 长文本查询处理

### TestCoursesEndpoint
测试 `/api/courses` 端点：
- ✅ 课程统计数据获取
- ✅ 响应模式验证
- ✅ 服务器错误处理
- ✅ 无课程情况处理

### TestAPIHeaders
测试API头部和CORS配置：
- ✅ CORS头部设置
- ✅ JSON内容类型处理

### TestAPIValidation
测试API请求验证：
- ✅ 无效JSON处理
- ✅ 额外字段忽略
- ✅ 错误数据类型处理

### TestAPIIntegration
集成测试：
- ✅ 查询和课程端点协同工作
- ✅ 会话持久性

### TestAPIPerformance
性能测试：
- ✅ 并发查询处理
- ✅ API响应时间

## 测试配置

### pytest配置 (pyproject.toml)
```toml
[tool.pytest.ini_options]
testpaths = ["backend/tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--tb=short",
    "--strict-markers",
    "--disable-warnings",
    "--color=yes"
]
markers = [
    "unit: marks tests as unit tests",
    "integration: marks tests as integration tests", 
    "api: marks tests as API tests",
    "slow: marks tests as slow running"
]
asyncio_mode = "auto"
```

## Fixtures

### API测试Fixtures（conftest.py中定义）

#### 核心Fixtures
- `test_app`: 测试用FastAPI应用（不包含静态文件）
- `test_client`: 同步测试客户端
- `async_test_client`: 异步测试客户端

#### 测试数据Fixtures
- `sample_query_request`: 带会话ID的查询请求
- `sample_query_request_no_session`: 不带会话ID的查询请求
- `invalid_query_request`: 无效查询请求
- `expected_query_response`: 期望的查询响应
- `expected_course_stats`: 期望的课程统计

#### Mock Fixtures
- `mock_rag_system_error`: 会抛出错误的Mock RAG系统

## 测试覆盖范围

✅ **已完成**:
- API端点测试 (`/api/query`, `/api/courses`)
- 请求/响应验证
- 错误处理测试
- CORS和头部测试
- 集成测试
- 性能测试
- pytest配置优化

✅ **特殊处理**:
- 解决了静态文件挂载问题（通过创建独立的测试应用）
- 提供了完整的mock系统，避免外部依赖
- 支持异步测试客户端

## 运行结果示例

```
========================== test session starts ===========================
backend/tests/test_api_endpoints.py::TestQueryEndpoint::test_query_with_session_id PASSED
backend/tests/test_api_endpoints.py::TestQueryEndpoint::test_query_without_session_id PASSED
backend/tests/test_api_endpoints.py::TestQueryEndpoint::test_query_invalid_request PASSED
[... 其他测试 ...]
====================== 21 passed, 1 warning in 0.13s =======================
```

## 依赖项

API测试所需的额外依赖已添加到 `pyproject.toml`：
- `pytest-asyncio>=0.21.0` - 异步测试支持
- `httpx>=0.24.0` - 异步HTTP客户端
- `fastapi.testclient` - FastAPI测试客户端

## 注意事项

1. API测试使用独立的测试应用，避免了主应用的静态文件挂载问题
2. 所有API测试使用mock RAG系统，不依赖外部服务
3. 测试标记系统允许选择性运行不同类型的测试
4. 错误处理测试确保API在各种故障情况下的健壮性