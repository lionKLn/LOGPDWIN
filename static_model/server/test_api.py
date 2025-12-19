#!/usr/bin/env python3
"""
模型推理API测试脚本
用于测试RESTful API服务器的各项功能
"""

import requests
import json
import time
import sys
from typing import Dict, List, Any

class APITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def test_health_check(self) -> bool:
        """测试健康检查接口"""
        print("🩺 测试健康检查接口...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 健康检查通过: {data}")
                return True
            else:
                print(f"❌ 健康检查失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 健康检查异常: {e}")
            return False
    
    def test_models_status(self) -> bool:
        """测试模型状态检查"""
        print("📊 测试模型状态检查...")
        try:
            response = self.session.get(f"{self.base_url}/models/status")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 模型状态: {json.dumps(data, indent=2)}")
                return True
            else:
                print(f"❌ 模型状态检查失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 模型状态检查异常: {e}")
            return False
    
    def test_single_prediction(self) -> bool:
        """测试单个样本预测"""
        print("🔮 测试单个样本预测...")
        
        test_data = {
            "component": "TestComponent",
            "case_id": "TEST_001",
            "test_suite": "SuiteA",
            "rule": "Rule1",
            "code_str": "def test_function():\n    return 'Hello World'",
            "Desc": "Test function description",
            "Func": "test_function",
            "case_spce": "Test specification",
            "case_purpose": "Test purpose"
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/static_predict",
                json=test_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 单个预测结果: {json.dumps(result, indent=2)}")
                return True
            else:
                print(f"❌ 单个预测失败: {response.status_code}")
                print(f"响应内容: {response.text}")
                return False
        except Exception as e:
            print(f"❌ 单个预测异常: {e}")
            return False
    
    def test_batch_prediction(self) -> bool:
        """测试批量预测"""
        print("📦 测试批量预测...")
        
        batch_data = {
            "batch": [
                {
                    "component": "CompA",
                    "case_id": "CASE_001",
                    "test_suite": "Suite1",
                    "rule": "RuleA",
                    "code_str": "print('Hello 1')",
                    "Desc": "Description 1",
                    "Func": "func1",
                    "case_spce": "Spec 1",
                    "case_purpose": "Purpose 1"
                },
                {
                    "component": "CompB", 
                    "case_id": "CASE_002",
                    "test_suite": "Suite2",
                    "rule": "RuleB",
                    "code_str": "print('Hello 2')",
                    "Desc": "Description 2",
                    "Func": "func2",
                    "case_spce": "Spec 2",
                    "case_purpose": "Purpose 2"
                }
            ]
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/static_predict_batch",
                json=batch_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 批量预测结果: {json.dumps(result, indent=2)}")
                return True
            else:
                print(f"❌ 批量预测失败: {response.status_code}")
                print(f"响应内容: {response.text}")
                return False
        except Exception as e:
            print(f"❌ 批量预测异常: {e}")
            return False
    
    def test_load_models(self) -> bool:
        """测试模型加载接口"""
        print("🔄 测试模型加载接口...")
        
        config = {
            "unsupervised_model_path": "logs/pdg/2025-05-20_14-30-00/best_pdg.pt",
            "classifier_model_path": "best_log_classifier.pt",
            "onehot_encoder_path": "onehot_encoder.pkl",
            "onehot_features_path": "onehot_feature_names.npy",
            "text_model_path": "./models/paraphrase-multilingual-MiniLM-L12-v2"
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/load_models",
                json=config,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 模型加载结果: {json.dumps(result, indent=2)}")
                return True
            else:
                print(f"❌ 模型加载失败: {response.status_code}")
                print(f"响应内容: {response.text}")
                return False
        except Exception as e:
            print(f"❌ 模型加载异常: {e}")
            return False
    
    def test_api_documentation(self) -> bool:
        """测试API文档接口"""
        print("📚 测试API文档接口...")
        try:
            # 测试Swagger文档
            response = self.session.get(f"{self.base_url}/docs")
            if response.status_code == 200:
                print("✅ Swagger文档可用")
            else:
                print(f"⚠️  Swagger文档返回状态码: {response.status_code}")
            
            # 测试ReDoc文档
            response = self.session.get(f"{self.base_url}/redoc")
            if response.status_code == 200:
                print("✅ ReDoc文档可用")
                return True
            else:
                print(f"⚠️  ReDoc文档返回状态码: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ API文档测试异常: {e}")
            return False
    
    def test_root_endpoint(self) -> bool:
        """测试根路径"""
        print("🏠 测试根路径...")
        try:
            response = self.session.get(f"{self.base_url}/")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 根路径响应: {data}")
                return True
            else:
                print(f"❌ 根路径测试失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 根路径测试异常: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """运行所有测试"""
        print("🚀 开始API测试...")
        print("=" * 50)
        
        results = {}
        
        results['root_endpoint'] = self.test_root_endpoint()
        time.sleep(1)
        
        results['api_documentation'] = self.test_api_documentation()
        time.sleep(1)
        
        results['health_check'] = self.test_health_check()
        time.sleep(1)
        
        results['models_status'] = self.test_models_status()
        time.sleep(1)
        
        results['single_prediction'] = self.test_single_prediction()
        time.sleep(1)
        
        results['batch_prediction'] = self.test_batch_prediction()
        time.sleep(1)
        
        results['load_models'] = self.test_load_models()
        
        print("=" * 50)
        print("📋 测试结果汇总:")
        
        all_passed = True
        for test_name, passed in results.items():
            status = "✅ 通过" if passed else "❌ 失败"
            print(f"{test_name}: {status}")
            if not passed:
                all_passed = False
        
        print("=" * 50)
        if all_passed:
            print("🎉 所有测试通过!")
        else:
            print("⚠️  部分测试失败，请检查服务器状态和日志")
        
        return results

def main():
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost:8000"
    
    print(f"连接到API服务器: {base_url}")
    print("请确保服务器已启动: python server.py")
    print()
    
    tester = APITester(base_url)
    
    try:
        results = tester.run_all_tests()
        
        failed_tests = [name for name, passed in results.items() if not passed]
        if failed_tests:
            print(f"\n❌ 失败的测试: {', '.join(failed_tests)}")
            sys.exit(1)
        else:
            print(f"\n✅ 所有测试通过!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n🛑 测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 测试过程中发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()