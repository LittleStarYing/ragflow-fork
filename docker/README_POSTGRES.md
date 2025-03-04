# RAGFlow PostgreSQL 部署指南

本指南提供了使用 PostgreSQL 作为文档存储引擎部署 RAGFlow 的详细说明。

## 先决条件

- Docker 和 Docker Compose
- Git
- 基本的命令行知识

## 快速开始

1. 克隆仓库（如果尚未克隆）：
   ```bash
   git clone https://github.com/infiniflow/ragflow.git
   cd ragflow
   ```

2. 运行部署脚本：
   ```bash
   ./deploy_custom_postgres.sh
   ```

3. 访问应用：
   ```
   http://localhost:9380
   ```

## 部署详情

### 组件

此部署包含以下服务：

- **PostgreSQL**: 主要文档存储引擎
- **Redis**: 用于缓存和任务队列
- **RAGFlow 服务器**: 主应用服务器

### 配置文件

主要配置文件位于：
- `conf/service_conf_pg_test.yaml`: PostgreSQL 配置

### 环境变量

关键环境变量：
- `DOC_ENGINE=postgresql`: 设置文档引擎为 PostgreSQL
- `PG_TEST=true`: 启用 PostgreSQL 测试模式
- `PYTHONPATH=/ragflow/`: 设置 Python 路径

## 故障排除

### 常见问题

1. **容器无法启动**
   - 检查 Docker 日志：`docker-compose -f docker/docker-compose-postgres-custom.yml logs ragflow`
   - 确保端口未被占用：`lsof -i :9380`

2. **数据库连接失败**
   - 检查 PostgreSQL 容器是否运行：`docker ps | grep postgres`
   - 验证连接配置：`docker exec -it ragflow-server-pg-test python3 -c "from rag.utils.pg_conn import PostgresqlConnection; print('连接成功' if PostgresqlConnection().connect() else '连接失败')"`

3. **依赖问题**
   - 检查 `requirements-pg-custom.txt` 是否包含所有必要依赖
   - 尝试手动安装缺失的依赖：`docker exec -it ragflow-server-pg-test pip install <package_name>`

### 日志查看

```bash
# 查看实时日志
docker-compose -f docker/docker-compose-postgres-custom.yml logs -f ragflow

# 查看 PostgreSQL 日志
docker-compose -f docker/docker-compose-postgres-custom.yml logs -f postgres
```

## 开发工作流

1. **进入容器**：
   ```bash
   docker exec -it ragflow-server-pg-test bash
   ```

2. **重启服务**：
   ```bash
   docker-compose -f docker/docker-compose-postgres-custom.yml restart ragflow
   ```

3. **停止环境**：
   ```bash
   docker-compose -f docker/docker-compose-postgres-custom.yml down
   ```

## 高级配置

### 自定义 PostgreSQL 配置

如需自定义 PostgreSQL 配置，可以修改 `docker-compose-postgres-custom.yml` 文件中的环境变量：

```yaml
postgres:
  environment:
    - POSTGRES_USER=custom_user
    - POSTGRES_PASSWORD=custom_password
    - POSTGRES_DB=custom_db
```

然后相应地更新 `service_conf_pg_test.yaml` 文件中的连接信息。

### 数据持久化

默认情况下，PostgreSQL 数据存储在 Docker 卷中。如需在主机上持久化数据，可以修改卷挂载配置：

```yaml
volumes:
  - ./postgres_data:/var/lib/postgresql/data
```

## 安全注意事项

1. **生产环境部署**：
   - 更改默认密码
   - 使用环境变量而非硬编码凭据
   - 配置适当的网络隔离

2. **API 安全**：
   - 考虑添加身份验证
   - 实施 HTTPS
   - 限制敏感端点访问

## 贡献

欢迎提交 Pull Request 和 Issue 以改进此部署配置。

## 许可证

Apache License 2.0
