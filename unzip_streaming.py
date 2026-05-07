import os
import zipfile
import glob
from pathlib import Path

def unzip_realtime_benchmarks():
    """
    解压StreamingBench/下所有以Real-Time开头的.zip文件
    将所有文件保存到./StreamingBench/data/real目录中
    """
    
    # 定义源目录和目标目录
    source_dir = "./StreamingBench"
    target_dir = "./StreamingBench/data/real"
    
    # 创建目标目录（如果不存在）
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    print(f"目标目录: {target_dir}")
    
    # 查找所有以Real-Time开头的.zip文件
    pattern = os.path.join(source_dir, "Real-Time*.zip")
    zip_files = glob.glob(pattern)
    
    if not zip_files:
        print(f"在 {source_dir} 中未找到以Real-Time开头的.zip文件")
        return
    
    print(f"找到 {len(zip_files)} 个Real-Time开头的.zip文件")
    
    # 解压每个zip文件
    for zip_path in sorted(zip_files):
        zip_name = os.path.basename(zip_path)
        print(f"\n正在解压: {zip_name}")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            print(f"  ✓ 成功解压 {zip_name}")
            os.remove(zip_path)
            print(f"  ✓ 已删除原文件 {zip_name}")
        except Exception as e:
            print(f"  ✗ 解压 {zip_name} 失败: {e}")
    
    print(f"\n解压完成！所有文件已保存到 {target_dir}")
    
    # 打印目标目录的统计信息
    total_files = sum(len(files) for _, _, files in os.walk(target_dir))
    print(f"目标目录中共有 {total_files} 个文件")

if __name__ == "__main__":
    unzip_realtime_benchmarks()
