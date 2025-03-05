import unittest
import json
import os
import sys
from unittest.mock import patch, MagicMock

# 添加项目根目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from rag.utils.pg_conn import PostgresConnection


class TestPostgresConnectionInsert(unittest.TestCase):
    """测试 PostgresConnection 类的 insert 方法"""

    def setUp(self):
        """设置测试环境"""
        # 模拟 PostgresConnection 类的初始化
        with patch('rag.utils.pg_conn.PostgresConnection._create_connection'), \
             patch('rag.utils.pg_conn.PostgresConnection._init_database'):
            self.pg_conn = PostgresConnection()
        
        # 加载测试数据
        self.test_data_path = '/Users/chenzhengying/work_space/ragflow/docker/insert_input.txt'
        with open(self.test_data_path, 'r') as f:
            content = f.read()
            lines = content.strip().split('\n')
            documents_line = ' '.join(lines[:lines.index('indexName:')])
            index_name_line = lines[lines.index('indexName:')]
            kb_id_line = lines[lines.index('knowledgebaseId:')]
            
            self.documents = json.loads(documents_line.replace('documents: ', ''))
            self.index_name = index_name_line.replace('indexName: ', '').strip()
            self.kb_id = kb_id_line.replace('knowledgebaseId: ', '').strip()

    @patch('rag.utils.pg_conn.PostgresConnection._get_connection')
    def test_insert_with_complex_data_types(self, mock_get_connection):
        """测试插入包含复杂数据类型的文档"""
        # 模拟数据库连接和游标
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_get_connection.return_value = mock_conn
        
        # 模拟表列信息
        mock_cur.fetchall.return_value = [
            ('doc_id', 'character varying', None),
            ('kb_id', 'character varying', None),
            ('docnm_kwd', 'character varying', None),
            ('title_tks', 'character varying', None),
            ('title_sm_tks', 'character varying', None),
            ('page_num_int', 'integer[]', None),
            ('position_int', 'jsonb', None),
            ('top_int', 'integer[]', None),
            ('content_with_weight', 'text', None),
            ('content_ltks', 'text', None),
            ('content_sm_ltks', 'text', None),
            ('id', 'character varying', None),
            ('create_time', 'character varying', None),
            ('create_timestamp_flt', 'double precision', None),
            ('img_id', 'character varying', None),
            ('embedding', 'vector', None)
        ]
        
        # 执行插入操作
        result = self.pg_conn.insert(self.documents, self.index_name, self.kb_id)
        
        # 验证结果
        self.assertTrue(result)
        
        # 验证是否调用了正确的方法
        mock_get_connection.assert_called_once()
        mock_conn.cursor.assert_called_once()
        
        # 验证执行的 SQL 语句
        # 我们期望至少有一次 execute 调用
        self.assertTrue(mock_cur.execute.called)
        
        # 打印调用参数，用于调试
        print("Execute calls:")
        for call in mock_cur.execute.call_args_list:
            args, kwargs = call
            print(f"SQL: {args[0]}")
            if len(args) > 1:
                print(f"Params: {args[1]}")
        
        # 验证是否提交了事务
        mock_conn.commit.assert_called_once()

    @patch('rag.utils.pg_conn.PostgresConnection._get_connection')
    def test_prepare_vector(self, mock_get_connection):
        """测试 _prepare_vector 方法"""
        # 从测试数据中提取向量数据
        vector_data = self.documents[0]['q_1536_vec']
        
        # 调用方法
        result = self.pg_conn._prepare_vector(vector_data)
        
        # 验证结果
        self.assertEqual(len(result), len(vector_data))
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], float)

    @patch('rag.utils.pg_conn.PostgresConnection._get_connection')
    def test_prepare_jsonb(self, mock_get_connection):
        """测试 _prepare_jsonb 方法"""
        # 从测试数据中提取 JSONB 数据
        jsonb_data = self.documents[0]['position_int']
        
        # 调用方法
        result = self.pg_conn._prepare_jsonb(jsonb_data)
        
        # 验证结果
        self.assertIsInstance(result, str)
        # 确保结果是有效的 JSON
        parsed = json.loads(result)
        self.assertEqual(parsed, jsonb_data)

    @patch('rag.utils.pg_conn.PostgresConnection._get_connection')
    def test_insert_with_array_types(self, mock_get_connection):
        """测试插入包含数组类型的文档"""
        # 模拟数据库连接和游标
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_get_connection.return_value = mock_conn
        
        # 模拟表列信息
        mock_cur.fetchall.return_value = [
            ('page_num_int', 'integer[]', None),
            ('top_int', 'integer[]', None)
        ]
        
        # 创建测试文档
        test_doc = {
            'page_num_int': [1, 2, 3],
            'top_int': 5  # 非数组值，应该被转换为数组
        }
        
        # 执行插入操作
        result = self.pg_conn.insert([test_doc], self.index_name, self.kb_id)
        
        # 验证结果
        self.assertTrue(result)
        
        # 验证是否调用了正确的方法
        mock_get_connection.assert_called_once()
        
        # 验证执行的 SQL 语句
        self.assertTrue(mock_cur.execute.called)
        
        # 打印调用参数，用于调试
        print("Execute calls for array test:")
        for call in mock_cur.execute.call_args_list:
            args, kwargs = call
            print(f"SQL: {args[0]}")
            if len(args) > 1:
                print(f"Params: {args[1]}")
        
        # 验证是否提交了事务
        mock_conn.commit.assert_called_once()


if __name__ == '__main__':
    unittest.main()
