from tree_sitter import Language
import os

# 输出路径（使用绝对路径）
build_path = os.path.abspath("build/my-languages.so")

# 语言仓库路径（根据你的项目调整）
Language.build_library(
    build_path,
    [
        'vendor/tree-sitter-c',
        'vendor/tree-sitter-python',
        # 如果有其他语言，也可以继续加
    ]
)

print(f"✅ 编译完成: {build_path}")
