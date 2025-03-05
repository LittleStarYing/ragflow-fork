#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import re
import json
import time
import os
import copy
import logging
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
import polars as pl

from rag import settings
from rag.settings import PAGERANK_FLD
from rag.utils import singleton
from api.utils.file_utils import get_project_base_directory

from rag.utils.doc_store_conn import (
    DocStoreConnection,
    MatchExpr,
    MatchTextExpr,
    MatchDenseExpr,
    FusionExpr,
    OrderByExpr,
)

logger = logging.getLogger('ragflow.pg_conn')

ATTEMPT_TIME = 2


@singleton
class PostgresConnection(DocStoreConnection):
    def __init__(self):
        """
        初始化PostgreSQL连接
        - 从配置文件读取连接参数
        - 建立数据库连接
        - 初始化所需扩展和表结构
        """
        self.dbname = settings.POSTGRES.get("db_name", "ragflow")
        self.host = "localhost"
        # self.host = settings.POSTGRES.get("host", "postgres")
        self.port = settings.POSTGRES.get("port", 5432)
        self.user = settings.POSTGRES.get("user", "postgres")
        self.password = settings.POSTGRES.get("password", "")
        self.pool_min_size = settings.POSTGRES.get("pool_min_size", 1)
        self.pool_max_size = settings.POSTGRES.get("pool_max_size", 10)

        logger.info(f"Use PostgreSQL {self.host}:{self.port} as the doc engine.")
        
        # 建立连接
        self._create_connection()
        
        # 初始化数据库结构
        self._init_database()
        
        logger.info(f"PostgreSQL {self.host}:{self.port} is healthy.")

    def _create_connection(self):
        """创建数据库连接，包含重试逻辑"""
        for _ in range(ATTEMPT_TIME):
            try:
                self.conn = psycopg2.connect(
                    database=self.dbname,
                    user=self.user,
                    password=self.password,
                    host=self.host,
                    port=self.port
                )
                if self.conn:
                    self.conn.autocommit = False
                    return
            except Exception as e:
                logger.warning(f"{str(e)}. Waiting PostgreSQL {self.host}:{self.port} to be healthy.")
                time.sleep(5)
        raise Exception(f"Could not connect to PostgreSQL at {self.host}:{self.port}")

    def _get_connection(self):
        """
        获取连接，如果连接关闭或者有问题则重新创建
        """
        try:
            if self.conn.closed:
                self._create_connection()
            # 简单测试连接是否有效
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
        except Exception:
            logger.warning("Connection lost, reconnecting...")
            self._create_connection()
        return self.conn

    def _init_database(self):
        """初始化数据库，包括创建扩展和基础表结构"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # 检查并创建 pgvector 扩展
                    cur.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');")
                    if not cur.fetchone()[0]:
                        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                        logger.info("Created pgvector extension")
                    conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise e

    def _ensure_pgvector_extension(self):
        """
        确保pgvector扩展已安装
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # 检查pgvector扩展是否已安装
                    cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
                    if not cur.fetchone():
                        logger.warning("pgvector extension not found, attempting to install...")
                        try:
                            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                            conn.commit()
                            logger.info("Successfully installed pgvector extension")
                        except Exception as e:
                            logger.error(f"Failed to install pgvector extension: {e}")
                            raise RuntimeError(f"pgvector extension is required but could not be installed: {e}")
                    else:
                        logger.debug("pgvector extension is already installed")
                    
                    # 检查pgvector版本
                    cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                    version = cur.fetchone()[0]
                    logger.info(f"Using pgvector version: {version}")
                    
                    return True
        except Exception as e:
            logger.error(f"Error checking pgvector extension: {e}")
            return False

    def execute_sql_file(self, file_path, cursor):
        """执行SQL文件内容"""
        try:
            with open(file_path, 'r') as file:
                sql_statements = [stmt.strip() for stmt in file.read().split(';') if stmt.strip()]
                for sql_statement in sql_statements:
                    if sql_statement.strip():
                        cursor.execute(sql_statement)
        except Exception as e:
            logger.error(f"Failed to execute SQL file {file_path}: {str(e)}")
            raise e

    def _sanitize_table_name(self, table_name):
        """安全处理表名，防止SQL注入"""
        return re.sub(r'[^a-zA-Z0-9_]', '', table_name)

    def _table_exists(self, table_name):
        """检查表是否存在"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT 1 FROM information_schema.tables 
                            WHERE table_name = %s
                        );
                    """, (table_name,))
                    return cur.fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to check if table exists: {str(e)}")
            return False

    def _get_table_columns(self, table_name, cur=None):
        """获取表的列名和类型"""
        columns = {}
        try:
            # 如果没有提供游标，创建一个新的连接和游标
            if cur is None:
                with self._get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            SELECT column_name, data_type, column_default 
                            FROM information_schema.columns 
                            WHERE table_name = %s
                        """, (table_name,))
                        return self._process_column_results(cur)
            else:
                # 使用提供的游标
                cur.execute("""
                    SELECT column_name, data_type, column_default 
                    FROM information_schema.columns 
                    WHERE table_name = %s
                """, (table_name,))
            # 处理结果
            return self._process_column_results(cur)
        except Exception as e:
            logger.error(f"Failed to get table columns: {str(e)}")
            return {}
            
    def _process_column_results(self, cur):
        """处理列查询结果"""
        columns = {}
        for row in cur.fetchall():
            columns[row[0]] = {'col_type': row[1], 'col_default': row[2]}
        return columns

    def _prepare_vector(self, vector_data):
        """
        准备向量数据以便与pgvector兼容
        """
        try:
            # 确保向量数据是浮点数列表
            if isinstance(vector_data, str):
                # 尝试将字符串转换为向量
                try:
                    # 处理可能的JSON字符串
                    import json
                    vector_data = json.loads(vector_data)
                except json.JSONDecodeError:
                    # 处理可能的字符串格式，例如 "[1.0, 2.0, 3.0]"
                    vector_data = vector_data.strip('[]').split(',')
            
            # 确保是列表或类似列表的对象
            if hasattr(vector_data, 'tolist'):  # numpy array或类似对象
                vector_data = vector_data.tolist()
            
            # 确保所有元素都是浮点数
            vector_data = [float(x) for x in vector_data]
            
            # 记录向量维度，用于调试
            vector_dim = len(vector_data)
            logger.debug(f"Prepared vector with dimension: {vector_dim}")
            
            return vector_data
        except Exception as e:
            logger.error(f"Error preparing vector data: {e}")
            raise ValueError(f"Invalid vector data format: {e}")
            
    def _prepare_jsonb(self, data):
        """
        准备JSONB数据，确保数据可以被正确地插入到PostgreSQL的JSONB字段中
        """
        try:
            import json
            
            # 如果已经是字符串，检查是否是有效的JSON
            if isinstance(data, str):
                try:
                    # 尝试解析以验证是否是有效的JSON
                    json.loads(data)
                    return data  # 如果是有效的JSON字符串，直接返回
                except json.JSONDecodeError:
                    # 如果不是有效的JSON，将其作为普通字符串处理
                    return json.dumps(data)
            
            # 如果是列表、字典或其他可序列化为JSON的对象
            return json.dumps(data)
            
        except Exception as e:
            logger.error(f"Error preparing JSONB data: {e}")
            # 如果出错，返回空的JSON数组
            return '[]'

    def _prepare_tsquery(self, query):
        """将查询字符串转换为tsquery格式"""
        # 替换特殊字符，分词并用&连接
        words = re.findall(r'\w+', query.lower())
        if not words:
            return ""
        return ' & '.join(words)

    """
        Database operations
    """

    def dbType(self) -> str:
        return "postgresql"

    def health(self) -> dict:
        """Return the health status of the database"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT version();")
                    version = cur.fetchone()[0]
                    return {
                        "type": "postgresql",
                        "version": version,
                        "status": "green" 
                    }
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "type": "postgresql",
                "status": "red",
                "error": str(e)
            }
    
    """
    Table operations
    """

    def createIdx(self, indexName: str, knowledgebaseId: str, vectorSize: int):
        """
        创建索引表
        
        Args:
            indexName: 索引名称
            knowledgebaseId: 知识库ID
            vectorSize: 向量维度
        
        Returns:
            bool: 是否成功创建索引
        """
        try:
            # 确保pgvector扩展已安装
            self._ensure_pgvector_extension()
            
            # 记录向量维度
            logger.info(f"Creating index with vector dimension: {vectorSize}")
            
            # 构建表名
            table_name = f"{indexName}_{knowledgebaseId}"
            
            # 检查表是否已存在
            if self.indexExist(indexName, knowledgebaseId):
                logger.info(f"Index {table_name} already exists")
                return True
            
            # 读取SQL文件内容
            sql_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'conf', 'postgres_vector_create.sql')
            
            with open(sql_file_path, 'r') as f:
                sql_template = f.read()
            
            # 替换SQL模板中的占位符
            sql = sql_template.replace('{table_name}', table_name)

            # 执行SQL
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    conn.commit()

                    cur.execute(f"""
                    ALTER TABLE "{table_name}" 
                    ADD COLUMN q_{vectorSize}_vec vector({vectorSize});
                    """)
                    conn.commit()
                    
                    # 创建全文搜索索引
                    cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{table_name}_content_fts ON "{table_name}" 
                    USING gin(to_tsvector('simple', content_with_weight));
                    """)
                    conn.commit()
                    
                    # 创建向量索引
                    # 使用HNSW索引，需要高精度和快速查询的场合，查询速度快，精度高，但占用内存大
                    # m：每个节点的最大连接数（默认 16）
                    # ef_construction：构建时的搜索宽度（默认 64）
                    cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{table_name}_embedding_ivfflat ON "{table_name}" 
                    USING hnsw (q_{vectorSize}_vec vector_l2_ops) WITH (m = 16, ef_construction = 50);
                    """)
                    
                    conn.commit()
                    
                    logger.info(f"Successfully created index {table_name} with vector dimension {vectorSize}")
                    return True
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False

    def deleteIdx(self, indexName: str, knowledgebaseId: str):
        """Delete table and its indices"""
        table_name = self._sanitize_table_name(f"{indexName}_{knowledgebaseId}")
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"DROP TABLE IF EXISTS \"{table_name}\" CASCADE;")
                conn.commit()
            logger.info(f"Successfully dropped table {table_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to drop table {table_name}: {str(e)}")
            return False

    def indexExist(self, indexName: str, knowledgebaseId: str) -> bool:
        """Check if table exists"""
        table_name = self._sanitize_table_name(f"{indexName}_{knowledgebaseId}")
        return self._table_exists(table_name)

    """
    CRUD operations
    """
    
    def search(
            self, selectFields: list[str],
            highlightFields: list[str],
            condition: dict,
            matchExprs: list[MatchExpr],
            orderBy: OrderByExpr,
            offset: int,
            limit: int,
            indexNames: str | list[str],
            knowledgebaseIds: list[str],
            aggFields: list[str] = [],
            rank_feature: dict | None = None
    ) -> tuple[pl.DataFrame, int]:
        """
        执行复合查询，支持文本搜索和向量搜索
        """
        logger.info(f"Search request: index={indexNames}, kb={knowledgebaseIds}, fields={selectFields}, condition={condition}")
        
        # 处理索引名称
        if isinstance(indexNames, str):
            indexNames = [indexNames]
            
        # 确保条件中包含知识库ID
        if knowledgebaseIds and 'kb_id' not in condition:
            condition['kb_id'] = knowledgebaseIds
            
        # 存储结果
        results = []
        total_count = 0

        # 如果selectFields中没有id，添加id
        if 'id' not in selectFields:
            selectFields.append('id')

        
        
        # 对每个索引执行查询
        for indexName in indexNames:
            for knowledgebaseId in knowledgebaseIds:
                table_name = f"{indexName}_{knowledgebaseId}"
                
                # 检查表是否存在
                if not self.indexExist(indexName, knowledgebaseId):
                    logger.warning(f"Table {table_name} does not exist, skipping")
                    continue
                    
                try:
                    # 构建查询
                    select_clause, from_clause, where_clause, order_clause, params, limit_clause = self._build_search_query(
                        table_name, selectFields, condition, matchExprs, orderBy, highlightFields, rank_feature
                    )
                    
                    # 组合完整查询
                    query = f"SELECT {select_clause} FROM {from_clause}"
                    if where_clause:
                        query += f" WHERE {where_clause}"
                    if order_clause:
                        query += f" ORDER BY {order_clause}"
                    
                    # 添加分页
                    if limit_clause is not None:
                        query += f" LIMIT {limit_clause}"
                    else:
                        query += f" LIMIT {limit} OFFSET {offset}"
                    
                    # 记录查询语句（不包含参数值）
                    logger.info(f"Executing query: {query}")
                    
                    # 记录参数数量，但不记录具体值以保护敏感数据
                    logger.info(f"Query parameters count: {len(params)}")
                    logger.info(f"Query parameters info: {params}")
                    
                    # 执行查询
                    with self._get_connection() as conn:
                        with conn.cursor() as cur:
                            # 执行查询
                            cur.execute(query, params)
                            rows = cur.fetchall()
                            
                            # 获取列名
                            columns = [desc[0] for desc in cur.description]
                            
                            # 计算总数
                            count_query = f"SELECT COUNT(*) FROM {from_clause}"
                            if where_clause:
                                count_query += f" WHERE {where_clause}"
                            cur.execute(count_query, params)
                            count = cur.fetchone()[0]
                            total_count += count
                            
                            # 转换结果为DataFrame
                            if rows:
                                # 创建字典列表
                                data = []
                                for row in rows:
                                    item = {}
                                    for i, col in enumerate(columns):
                                        value = row[i]
                                        # 处理 JSONB 格式的字段
                                        if col in ["position_int", "page_num_int", "top_int"] and isinstance(value, str):
                                            try:
                                                # 尝试解析 JSON 字符串
                                                value = json.loads(value)
                                            except (json.JSONDecodeError, TypeError):
                                                # 如果解析失败，保持原样
                                                pass
                                        item[col] = value
                                    data.append(item)
                                
                                # 创建DataFrame
                                df = pl.DataFrame(data)
                                results.append(df)
                                logger.info(f"Found {len(rows)} results in {table_name}")
                            else:
                                logger.info(f"No results found in {table_name}")
                except Exception as e:
                    logger.error(f"Error searching {table_name}: {str(e)}")
                    # raise

                    # 继续处理其他表
        
        # 合并结果
        if results:
            # 合并所有DataFrame
            combined_df = pl.concat(results)
            
            # 如果有排序，按_score重新排序
            if '_score' in combined_df.columns and any('_score' in field for field in orderBy.fields):
                combined_df = combined_df.sort('_score', descending=True)
                
            # 应用全局分页
            if offset > 0:
                combined_df = combined_df.slice(offset, len(combined_df))
            if limit > 0 and len(combined_df) > limit:
                combined_df = combined_df.slice(0, limit)
                
            logger.info(f"Total results: {total_count}, returned: {len(combined_df)}")
            return (combined_df, total_count)
        else:
            # 返回空结果
            logger.info("No results found in any table")
            return (self._create_empty_result_frame(selectFields), 0)
    
    def _build_search_query(self, table_name, selectFields, condition, matchExprs, orderBy, highlightFields, rank_feature):
        """构建搜索查询的各个部分"""
        # 准备查询部分
        select_parts = [f"\"{field}\"" for field in selectFields if field != "_score"]
        where_parts = []
        params = []
        if matchExprs:
            params = ["content_with_weight", "q_1536_vec"]

        limit_clause = None

        # 判断是什么得分方式
        
        # 处理条件
        filter_cond = None
        filter_fulltext = ""
        if condition:
            for key, value in condition.items():
                if key == "kb_id":
                    if isinstance(value, list):
                        placeholders = [f"%s"] * len(value)
                        params.extend(value)
                        where_parts.append(f"\"kb_id\" IN ({', '.join(placeholders)})")
                    else:
                        where_parts.append(f"\"kb_id\" = %s")
                        params.append(value)
                elif isinstance(value, list):
                    # 如果列表为空，跳过该条件，避免生成 "key IN ()" 这样的无效SQL
                    if not value:
                        continue
                    placeholders = [f"%s"] * len(value)
                    params.extend(value)
                    where_parts.append(f"\"{key}\" IN ({', '.join(placeholders)})")
                elif value is not None:
                    where_parts.append(f"\"{key}\" = %s")
                    params.append(value)
        
        # 处理匹配表达式
        has_text_match = False
        has_vector_match = False
        tsquery_expr = None
        order_parts = []
        
        text_where=[]
        vector_where=[]
        condition_where= where_parts.copy()
        
        for match_expr in matchExprs:
            if isinstance(match_expr, MatchTextExpr):
                has_text_match = True
                # 准备全文搜索查询
                tsquery = self._prepare_tsquery(match_expr.matching_text)
                tsquery_expr = f"to_tsquery('simple', %s)"
                params.append(tsquery)
                
                # 使用字段列表或默认的内容字段
                search_fields = "content_with_weight"
                weight_value = 1.0  # 默认权重
                
                if hasattr(match_expr, 'fields') and match_expr.fields:
                    # 处理可能带有权重的字段名，例如 title_tks^10
                    # TODO 实现
                    field = match_expr.fields[0]  # 简化处理，仅使用第一个字段
                    
                    # 检查是否带有权重标记
                    if '^' in field:
                        field_parts = field.split('^')
                        search_fields = field_parts[0]  # 实际字段名
                        try:
                            weight_value = float(field_parts[1])  # 权重值
                            logger.info(f"Using weight {weight_value} for field {search_fields}")
                        except (IndexError, ValueError):
                            logger.warning(f"Invalid weight format in {field}, using default weight")
                    else:
                        search_fields = field
                
                # 使用统一的tsvector表达式避免重复计算 - 确保字段名被正确引用
                tsvector_expr = f"to_tsvector('simple', \"{search_fields}\")"
                
                # 对内容字段应用全文搜索
                # 判断是否应用为硬性过滤条件
                text_filter_mode = match_expr.extra_options.get("text_filter_mode", "hard")  # 默认硬性过滤
                
                if text_filter_mode == "hard":
                    # 硬性过滤模式 - 必须匹配文本搜索条件
                    text_where.append(f"{tsvector_expr} @@ {tsquery_expr}")
                elif text_filter_mode == "soft" and has_vector_match:
                    # 软性过滤模式 - 不强制要求文本匹配，但影响评分
                    pass  # 不添加到 where_parts中
                else:
                    # 默认添加到过滤条件
                    text_where.append(f"{tsvector_expr} @@ {tsquery_expr}")
                    
                # 无论未决定硬性过滤与否，始终添加文本评分
                select_parts.append(f"ts_rank_cd({tsvector_expr}, {tsquery_expr}) * {weight_value} AS text_score")
                
                # 处理额外选项
                if hasattr(match_expr, 'extra_options') and match_expr.extra_options:
                    # 处理最小匹配参数等
                    if "minimum_should_match" in match_expr.extra_options:
                        # PostgreSQL没有直接等价物，可能需要调整查询逻辑
                        pass
            elif isinstance(match_expr, MatchDenseExpr):
                has_vector_match = True
                
                # 1. 获取向量数据并添加到参数
                vector_data = self._prepare_vector(match_expr.embedding_data)
                vector_clause = "%s::vector"
                params.append(vector_data)
                
                # 2. 获取向量列名 - 默认使用q_1536_vec而非embedding
                vector_column = getattr(match_expr, "vector_column_name", "q_1536_vec")
                
                # 3. 确定距离操作符 - 简化为单一逻辑
                distance_type = (getattr(match_expr, 'distance_type', None) or 
                                match_expr.extra_options.get("distance_type", "cosine")).lower()
                
                distance_operator = {
                    "l2": "<=>", 
                    "euclidean": "<=>",
                    "dot": "<#>", 
                    "inner_product": "<#>",
                    "cosine": "<->"
                }.get(distance_type, "<->")  # 默认使用余弦距离
                
                # 4. 构建相似度分数计算 - 使用变量避免重复
                similarity_expr = f"(1 - (\"{vector_column}\" {distance_operator} {vector_clause}))"
                select_parts.append(f"{similarity_expr} AS vector_score")
                
                # 5. 仅在明确请求时才应用相似度阈值（与ES保持一致）
                threshold = match_expr.extra_options.get("similarity")
                if threshold is not None and match_expr.extra_options.get("apply_threshold_filter", False):
                    logger.info(f"应用向量相似度硬性过滤，阈值: {threshold}")
                    vector_where.append(f"{similarity_expr} > {float(threshold)}")
                
                # 6. 设置排序和分页 - 默认用于排序而非过滤
                top_k = getattr(match_expr, 'topn', None) or match_expr.extra_options.get("top_k")
                if top_k:
                    # 使用距离值作为排序依据（升序，距离越小越相似）
                    order_parts = [f"\"{vector_column}\" {distance_operator} {vector_clause}"]
                    limit_clause = int(top_k)
        
        # 组合得分 - 增强版
        if has_text_match and has_vector_match:
            vector_weight = 0.5  # 默认向量权重
            text_weight = 0.5  # 默认纯文本权重
            hybrid_mode = "weighted_sum"  # 默认使用加权平均
            
            # 查看是否有 FusionExpr 调整权重和模式
            for match_expr in matchExprs:
                if isinstance(match_expr, FusionExpr):
                    # 获取融合模式
                    if match_expr.method in ["weighted_sum", "max_score", "min_score"]:
                        hybrid_mode = match_expr.method
                    
                    # 获取权重
                    if "weights" in match_expr.fusion_params:
                        weights = match_expr.fusion_params["weights"].split(",")
                        if len(weights) >= 2:
                            text_weight = float(weights[0])
                            vector_weight = float(weights[1])
            
            # 根据混合模式创建不同的组合得分
            if hybrid_mode == "weighted_sum":
                # 加权平均
                select_parts.append(f"((text_score * {text_weight}) + (vector_score * {vector_weight})) AS _score")
                logger.info(f"使用加权平均混合搜索: 文本={text_weight}, 向量={vector_weight}")
            elif hybrid_mode == "max_score":
                # 取最大分数
                select_parts.append(f"GREATEST(text_score, vector_score) AS _score")
                logger.info("使用最大分数混合搜索")
            elif hybrid_mode == "min_score":
                # 取最小分数
                select_parts.append(f"LEAST(text_score, vector_score) AS _score")
                logger.info("使用最小分数混合搜索")
            else:
                # 默认加权平均
                select_parts.append(f"((text_score * {text_weight}) + (vector_score * {vector_weight})) AS _score")
        elif has_text_match:
            select_parts.append("text_score AS _score")
            logger.info("仅使用文本搜索")
        elif has_vector_match:
            select_parts.append("vector_score AS _score")
            logger.info("仅使用向量搜索")
        
        # 处理排序
        if not order_parts:  # 如果没有设置向量距离排序
            if has_text_match or has_vector_match:
                order_parts.append("_score DESC")
            
            if orderBy and orderBy.fields:
                for field, order in orderBy.fields:
                    direction = "ASC" if order == 0 else "DESC"
                    order_parts.append(f"\"{field}\" {direction}")
        
        # 组合查询部分
        select_clause = ", ".join(select_parts)
        from_clause = f"\"{table_name}\""
        where_clause = ''

        condition_clause = " AND ".join(condition_where) if condition_where else ""
        text_clause = " AND ".join(text_where) if text_where else ""
        if vector_where:
            vector_clause = " AND ".join(vector_where) if vector_where else ""
            where_clause = condition_clause + ' AND ( (' + text_clause + ') OR (' + vector_clause + ') )'
        else:
            where_clause = condition_clause + ' AND ' + text_clause
        if where_clause.endswith(' AND '):
            where_clause = where_clause[:-5]
        # where_clause = " AND ".join(where_parts) if where_parts else ""
        
        order_clause = ", ".join(order_parts) if order_parts else ""
        # params.append('q_1536_vec')
        
        return select_clause, from_clause, where_clause, order_clause, params, limit_clause

    def _create_empty_result_frame(self, selectFields):
        """创建空的结果DataFrame"""
        schema = {field: [] for field in selectFields if field != "_score"}
        schema["_score"] = []
        return pl.DataFrame(schema)

    def get(self, chunkId: str, indexName: str, knowledgebaseIds: list[str]):
        """Get single document by chunk_id"""
        for kb_id in knowledgebaseIds:
            table_name = self._sanitize_table_name(f"{indexName}_{kb_id}")
            if not self.indexExist(indexName, kb_id):
                continue

            try:
                with self._get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(f"""
                            SELECT * FROM "{table_name}"
                            WHERE id = %s AND kb_id = %s
                            LIMIT 1
                        """, (chunkId, kb_id))
                        
                        row = cur.fetchone()
                        if row:
                            columns = [desc[0] for desc in cur.description]
                            return pl.DataFrame({
                                col: [row[i]]
                                for i, col in enumerate(columns)
                            })
            except Exception as e:
                logger.error(f"Error fetching document: {str(e)}")
                continue
                    
        return pl.DataFrame()

    def insert(self, documents: list[dict], indexName: str, knowledgebaseId: str = None) -> bool:
        """
        将文档插入到指定索引表中
        
        Args:
            documents: 文档列表，每个文档是一个字典
            indexName: 索引名称
            knowledgebaseId: 知识库ID
            
        Returns:
            bool: 是否成功插入所有文档
        """
        if not documents:
            logger.warning("No documents to insert")
            return True
            
        # 如果knowledgebaseId为None，使用默认值
        if knowledgebaseId is None:
            logger.warning("knowledgebaseId is None, using default value")
            knowledgebaseId = "default"
            
        # 构建表名
        table_name = f"{indexName}_{knowledgebaseId}"
        
        # 确保表存在
        if not self.indexExist(indexName, knowledgebaseId):
            logger.info(f"Table {table_name} does not exist, creating...")
            # 从文档中推断向量维度
            vector_size = 0
            patt = re.compile(r"q_(?P<vector_size>\d+)_vec")
            for k in documents[0].keys():
                m = patt.match(k)
                if m:
                    vector_size = int(m.group("vector_size"))
                    break
            if vector_size == 0:
                raise ValueError("Cannot infer vector size from documents")
            
            try:
                self.createIdx(indexName, knowledgebaseId, vector_size)
            except Exception:
                logger.error(f"Failed to createIdx {indexName, knowledgebaseId, vector_size}")
                return False
        
        # 记录插入文档数量
        logger.info(f"Inserting {len(documents)} documents into {table_name}")

        # 记录入参
        # with open(f"/ragflow/logs/insert_input.txt", "w") as f:
        #     f.write(f"documents: {json.dumps(documents, indent=4)}\n")
        #     f.write(f"indexName: {indexName}\n")
        #     f.write(f"knowledgebaseId: {knowledgebaseId}\n")
        
        # embedding fields can't have a default value....
        embedding_clmns = []
        clmns = self._get_table_columns(table_name)
        logger.info(f"_get_table_columns: {clmns}")
        for key, value in clmns.items():
            if value['col_type'] == 'USER-DEFINED':
                r = re.search(r"q_(\d+)_vec", key)
                if r:
                    embedding_clmns.append((key, int(r.group(1))))
        logger.info(f"embedding_clmns: {embedding_clmns}")
        
        try:

            # 批量插入文档
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    docs = copy.deepcopy(documents)
                    for d in docs:
                        assert "_id" not in d
                        assert "id" in d
                        for k, v in d.items():
                            if k in ["important_kwd", "question_kwd", "entities_kwd", "tag_kwd", "source_id"]:
                                assert isinstance(v, list)
                                d[k] = "###".join(v)
                            elif re.search(r"_feas$", k):
                                d[k] = json.dumps(v)
                            elif k == 'kb_id':
                                if isinstance(d[k], list):
                                    d[k] = d[k][0]  # since d[k] is a list, but we need a str
                            elif k == "position_int":
                                assert isinstance(v, list)
                                # 将元组转换为列表，然后转换为 JSON
                                if v and isinstance(v[0], tuple):
                                    # 将元组转换为列表
                                    position_list = [list(item) for item in v]
                                    # 将列表转换为 JSON字符串
                                    d[k] = json.dumps(position_list)
                                else:
                                    # 直接转换为 JSON字符串
                                    d[k] = json.dumps(v)
                            elif k in ["page_num_int", "top_int"]:
                                assert isinstance(v, list)
                                # 将列表转换为 JSON字符串
                                d[k] = json.dumps(v)
                        
                        for n, vs in embedding_clmns:
                            if n in d:
                                continue
                            d[n] = [0] * vs
                    
                    # 先删除重复数据
                    ids = ["'{}'".format(d["id"]) for d in docs]
                    str_ids = ", ".join(ids)
                    str_filter = f"id IN ({str_ids})"
                    delete_sql = f"delete from {table_name} where {str_filter}"
                    logger.info(delete_sql)
                    cur.execute(delete_sql)
                    conn.commit()

                    # 批量插入数据
                    cols = ", ".join('{}'.format(k) for k in docs[0].keys())
                    val_cols = ','.join('%s' for _ in docs[0].keys())
                    sql = "insert into {} (%s) values (%s)".format(table_name)
                    insert_sql = sql % (cols, val_cols)
                    cur.executemany(insert_sql, [tuple(d.values()) for d in docs])
                    conn.commit()

        except Exception as e:
            logger.error(f"Failed to insert document: {e}")
            # 继续处理其他文档

        return []

    def update(self, condition: dict, newValue: dict, indexName: str, knowledgebaseId: str):
        """Update documents matching condition"""
        table_name = self._sanitize_table_name(f"{indexName}_{knowledgebaseId}")
        if not self.indexExist(indexName, knowledgebaseId):
            return True

        set_clauses = []
        where_clauses = []
        params = []

        # Build SET clause
        for key, value in newValue.items():
            if key == 'metadata':
                set_clauses.append(f"{key} = %s::jsonb")
                params.append(json.dumps(value))
            else:
                set_clauses.append(f"{key} = %s")
                params.append(value)
        set_clauses.append("updated_at = CURRENT_TIMESTAMP")

        # Build WHERE clause
        for key, value in condition.items():
            if isinstance(value, list):
                where_clauses.append(f"{key} = ANY(%s)")
                params.append(value)
            else:
                where_clauses.append(f"{key} = %s")
                params.append(value)

        query = f"""
            UPDATE {table_name}
            SET {', '.join(set_clauses)}
            WHERE {' AND '.join(where_clauses)}
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, tuple(params))
            conn.commit()
        return True

    def delete(self, condition: dict, indexName: str, knowledgebaseId: str):
        """Delete documents matching condition"""
        table_name = self._sanitize_table_name(f"{indexName}_{knowledgebaseId}")
        if not self.indexExist(indexName, knowledgebaseId):
            return True

        where_clauses = []
        params = []

        for key, value in condition.items():
            if isinstance(value, list):
                where_clauses.append(f"{key} = ANY(%s)")
                params.append(value)
            else:
                where_clauses.append(f"{key} = %s")
                params.append(value)

        query = f"""
            DELETE FROM {table_name}
            WHERE {' AND '.join(where_clauses)}
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, tuple(params))
            conn.commit()
        return True

    def getTotal(self, res: tuple[pl.DataFrame, int] | pl.DataFrame) -> int:
        """Get total number of results"""
        if isinstance(res, tuple):
            return res[1]
        return len(res)
    
    def getChunkIds(self, res: tuple[pl.DataFrame, int] | pl.DataFrame) -> list[str]:
        logger.warning(f"getChunkIds res: {str(res)}.  --res[0]:{res[0]} -- type(res[0]):{type(res[0])}")
        if isinstance(res, tuple):
            res = res[0]
        return list(res["id"])

    def getFields(self, res: tuple[pl.DataFrame, int] | pl.DataFrame, fields: list[str]) -> dict[str, dict]:
        """Get specified fields from results"""
        df = res[0] if isinstance(res, tuple) else res
        if df.is_empty():
            return {}
        
        # 转换为字典格式，使用第一列作为键
        result = {}
        dicts = df.select(fields).to_dicts()
        for i, item in enumerate(dicts):
            # 使用索引作为键，因为可能没有唯一标识符
            id = item["id"]
            result[id] = item
        
        return result

    def getHighlight(self, res: tuple[pl.DataFrame, int] | pl.DataFrame, keywords: list[str], fieldnm: str) -> dict:
        """Get highlighted content - not implemented for PostgreSQL"""
        return {}

    def getAggregation(self, res: tuple[pl.DataFrame, int] | pl.DataFrame, fieldnm: str) -> dict:
        """获取聚合结果

        Args:
            res: 查询结果
            fieldnm: 字段名称

        Returns:
            dict: 聚合结果
        """
        # TODO: 实现聚合功能
        return {}

    def sql(self, sql: str, fetch_size: int = 1000, format: str = 'json') -> list:
        """执行原生 SQL 查询

        Args:
            sql: SQL 查询语句
            fetch_size: 每次获取的数据量
            format: 返回格式，支持 'json' 或 'dict'

        Returns:
            list: 查询结果
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    columns = [desc[0] for desc in cur.description]
                    results = []
                    
                    while True:
                        rows = cur.fetchmany(fetch_size)
                        if not rows:
                            break
                        
                        if format == 'json':
                            results.extend([
                                json.dumps(dict(zip(columns, row)))
                                for row in rows
                            ])
                        else:
                            results.extend([
                                dict(zip(columns, row))
                                for row in rows
                            ])
                        
                    return results
                
        except Exception as e:
            logger.error(f"Error executing SQL query: {str(e)}")
            raise

    def _check_connection(self) -> bool:
        """检查数据库连接状态，如果断开则重连

        Returns:
            bool: 连接是否正常
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
                return True
        except Exception as e:
            logger.warning(f"Database connection lost: {str(e)}. Attempting to reconnect...")
            try:
                self.conn = psycopg2.connect(
                    database=self.dbname,
                    user=self.user,
                    password=self.password,
                    host=self.host,
                    port=self.port
                )
                return True
            except Exception as e:
                logger.error(f"Failed to reconnect to database: {str(e)}")
                return False

    def __del__(self):
        """关闭数据库连接"""
        if hasattr(self, 'conn') and self.conn:
            try:
                self.conn.close()
            except Exception as e:
                logger.warning(f"Error closing database connection: {str(e)}")

    def health(self) -> dict:
        """检查健康状态"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
                return {
                    "type": "postgresql",
                    "status": "green",
                    "error": None
                }
        except Exception as e:
            return {
                "type": "postgresql",
                "status": "red",
                "error": str(e)
            }
