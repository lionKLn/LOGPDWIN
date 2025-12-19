#!/usr/bin/env python3
"""
Excel转JSON转换器 - 数据处理模块

将Excel文件转换为符合PredictionRequest格式的JSON数据
"""

import pandas as pd
import json
import argparse
import logging
from typing import Dict, List, Any, Optional
import os
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExcelToJsonConverter:
    """Excel转JSON转换器"""
    
    def __init__(self):
        self.required_fields = ['component', 'case_id', 'test_suite', 'rule', 'code_str']
        self.optional_fields = ['desc', 'func', 'case_spce', 'case_purpose']
    
    def extract_json_from_data_field(self, data_str: str) -> Dict[str, Any]:
        """从data字段字符串中提取JSON数据"""
        try:
            # 尝试直接解析JSON
            if isinstance(data_str, str) and data_str.strip().startswith('{'):
                return json.loads(data_str.strip())
            return {}
        except json.JSONDecodeError:
            logger.warning(f"无法解析JSON数据: {data_str[:100]}...")
            return {}
    
    def convert_excel_to_json(self, excel_path: str, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        将Excel文件转换为JSON格式数据
        
        Args:
            excel_path: Excel文件路径
            output_path: 输出JSON文件路径（可选）
            
        Returns:
            转换后的JSON数据列表
        """
        try:
            logger.info(f"开始读取Excel文件: {excel_path}")
            
            # 读取Excel文件
            df = pd.read_excel(excel_path)
            logger.info(f"Excel文件读取成功，共{len(df)}行数据")
            logger.info(f"Excel列名: {df.columns.tolist()}")
            
            # 检查必要的列是否存在
            if 'rule' not in df.columns:
                raise ValueError("Excel文件中缺少'rule'列")
            
            if 'data' not in df.columns:
                logger.warning("Excel文件中缺少'data'列，将尝试从其他列提取数据")
            
            converted_data = []
            
            for index, row in df.iterrows():
                try:
                    # 基础数据
                    prediction_data = {
                        'rule': str(row.get('rule', '')),
                    }
                    
                    # 从data字段提取JSON数据
                    if 'data' in df.columns and pd.notna(row.get('data')):
                        json_data = self.extract_json_from_data_field(str(row['data']))
                        
                        # 提取必需的字段（code_str为严格必选项）
                        missing_required = []
                        for field in self.required_fields:
                            if field == 'rule':  # rule字段已经处理
                                continue
                            if field in json_data:
                                prediction_data[field] = str(json_data[field])
                            elif field in row and pd.notna(row.get(field)):
                                prediction_data[field] = str(row[field])
                            else:
                                missing_required.append(field)
                        
                        # 检查code_str是否为严格必选项
                        if 'code_str' not in prediction_data or not prediction_data['code_str'].strip():
                            logger.error(f"第{index+1}行缺少严格必需字段code_str，跳过该记录")
                            continue
                        
                        if missing_required:
                            logger.warning(f"第{index+1}行缺少必需字段: {missing_required}")
                        
                        # 提取可选字段
                        for field in self.optional_fields:
                            if field in json_data:
                                prediction_data[field] = str(json_data[field])
                            elif field in row and pd.notna(row.get(field)):
                                prediction_data[field] = str(row[field])
                            else:
                                prediction_data[field] = ""
                    
                    else:
                        # 如果没有data字段，直接从行数据中提取
                        missing_required = []
                        for field in self.required_fields + self.optional_fields:
                            if field == 'rule':  # rule字段已经处理
                                continue
                            if field in row and pd.notna(row.get(field)):
                                prediction_data[field] = str(row[field])
                            else:
                                if field in self.required_fields:
                                    missing_required.append(field)
                        
                        # 检查code_str是否为严格必选项
                        if 'code_str' not in prediction_data or not prediction_data['code_str'].strip():
                            logger.error(f"第{index+1}行缺少严格必需字段code_str，跳过该记录")
                            continue
                        
                        if missing_required:
                            logger.warning(f"第{index+1}行缺少必需字段: {missing_required}")
                    
                    converted_data.append(prediction_data)
                    
                except Exception as e:
                    logger.error(f"处理第{index+1}行数据时出错: {e}")
                    continue
            
            logger.info(f"数据转换完成，共转换{len(converted_data)}条记录")
            
            # 保存为JSON文件
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(converted_data, f, ensure_ascii=False, indent=2)
                logger.info(f"JSON数据已保存到: {output_path}")
            
            return converted_data
            
        except Exception as e:
            logger.error(f"转换Excel文件时出错: {e}")
            raise
    
    def validate_data(self, data: List[Dict[str, Any]]) -> Dict[str, int]:
        """验证转换后的数据"""
        validation_stats = {
            'total': len(data),
            'valid': 0,
            'invalid': 0,
            'missing_required_fields': 0
        }
        
        for item in data:
            is_valid = True
            missing_fields = []
            
            # 检查必需字段
            for field in self.required_fields:
                if field not in item or not item[field].strip():
                    missing_fields.append(field)
                    is_valid = False
            
            if missing_fields:
                validation_stats['missing_required_fields'] += 1
                logger.warning(f"数据项缺少必需字段: {missing_fields}")
            
            if is_valid:
                validation_stats['valid'] += 1
            else:
                validation_stats['invalid'] += 1
        
        return validation_stats
    
    def generate_sample_data(self, output_path: str = None) -> List[Dict[str, Any]]:
        """生成示例数据用于测试"""
        sample_data = [
            {
                "component": "Authentication",
                "case_id": "AUTH_001",
                "test_suite": "LoginModule",
                "rule": "PasswordComplexity",
                "code_str": "def validate_password(password):\n    if len(password) < 8:\n        return False\n    return True",
                "desc": "验证密码复杂度规则",
                "func": "validate_password",
                "case_spce": "密码长度至少8位",
                "case_purpose": "确保用户密码安全性"
            },
            {
                "component": "Database",
                "case_id": "DB_002",
                "test_suite": "ConnectionPool",
                "rule": "ConnectionTimeout",
                "code_str": "def get_connection(timeout=30):\n    return create_connection(timeout=timeout)",
                "desc": "数据库连接超时处理",
                "func": "get_connection",
                "case_spce": "连接超时30秒",
                "case_purpose": "防止连接泄露"
            },
            {
                "component": "Frontend",
                "case_id": "UI_003",
                "test_suite": "FormValidation",
                "rule": "InputValidation",
                "code_str": "function validateInput(input) {\n    return input && input.length > 0;\n}",
                "desc": "输入验证规则",
                "func": "validateInput",
                "case_spce": "输入不能为空",
                "case_purpose": "确保数据完整性"
            }
        ]
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, ensure_ascii=False, indent=2)
            logger.info(f"示例数据已保存到: {output_path}")
        
        return sample_data

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Excel转JSON转换工具')
    parser.add_argument('-e', '--excel', type=str, help='输入Excel文件路径')
    parser.add_argument('-o', '--output', type=str, help='输出JSON文件路径（可选）')
    parser.add_argument('-v', '--validate', action='store_true', help='验证转换后的数据')
    parser.add_argument('-s', '--sample', action='store_true', help='生成示例数据')
    
    args = parser.parse_args()
    
    converter = ExcelToJsonConverter()
    
    try:
        if args.sample:
            # 生成示例数据
            sample_data = converter.generate_sample_data(args.output)
            print(f"✅ 生成{len(sample_data)}条示例数据")
            
            if args.validate:
                stats = converter.validate_data(sample_data)
                print(f"验证结果: {stats}")
        
        elif args.excel:
            # 转换Excel文件
            converted_data = converter.convert_excel_to_json(args.excel, args.output)
            print(f"✅ 成功转换{len(converted_data)}条记录")
            
            if args.validate:
                stats = converter.validate_data(converted_data)
                print(f"验证结果: {stats}")
        
        else:
            print("请提供输入文件路径或使用--sample生成示例数据")
            parser.print_help()
    
    except Exception as e:
        logger.error(f"转换过程出错: {e}")
        print(f"❌ 转换失败: {e}")

if __name__ == "__main__":
    main()