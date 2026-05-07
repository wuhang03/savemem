import os
import glob
import subprocess
from tqdm import tqdm


def unzip_ovo():
    """
    合并并解压 ovo/ 下 chunked_videos 开头的分卷 .tar 文件
    例如: chunked_videos.tar.partaa, chunked_videos.tar.partab, ...
    解压到 ovo/ 目录中，用 tqdm 显示字节级进度
    """
    source_dir = "./ovo"
    target_dir = "./ovo"

    # 只取 chunked_videos 开头的分卷文件
    all_parts = sorted(glob.glob(os.path.join(source_dir, "chunked_videos.tar.part*")))

    if not all_parts:
        print(f"在 {source_dir} 中未找到 chunked_videos.tar.part* 分卷文件")
        return

    total_size = sum(os.path.getsize(p) for p in all_parts)
    print(f"找到 {len(all_parts)} 个分卷，合计 {total_size / 1024**3:.2f} GB")

    tar_cmd = ["tar", "xf", "-", "-C", target_dir]
    chunk_size = 4 * 1024 * 1024  # 4 MB

    try:
        tar_proc = subprocess.Popen(tar_cmd, stdin=subprocess.PIPE)

        with tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024,
                  desc="解压 chunked_videos") as pbar:
            for part in all_parts:
                with open(part, "rb") as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        tar_proc.stdin.write(chunk)
                        pbar.update(len(chunk))

        tar_proc.stdin.close()
        tar_proc.wait()

        if tar_proc.returncode == 0:
            print("✓ 成功解压 chunked_videos")
            deleted_count = 0
            for part in all_parts:
                os.remove(part)
                deleted_count += 1
            print(f"✓ 已删除 {deleted_count} 个原始分卷文件")
        else:
            print(f"✗ 解压失败，tar 返回码: {tar_proc.returncode}")

    except Exception as e:
        print(f"✗ 处理时出错: {e}")


if __name__ == "__main__":
    unzip_ovo()
