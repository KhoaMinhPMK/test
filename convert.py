#%%
# Cell 1: Script để chuyển đổi file Python có quy ước #%% thành file .ipynb

import json
import sys
import re

def py_to_ipynb(py_filename, ipynb_filename):
    """
    Chuyển đổi một file Python (.py) với các cell được phân tách bằng #%%
    thành một file Jupyter test (.ipynb).

    Args:
        py_filename (str): Đường dẫn đến file .py đầu vào.
        ipynb_filename (str): Đường dẫn để lưu file .ipynb đầu ra.
    """
    try:
        with open(py_filename, 'r', encoding='utf-8') as f:
            script_content = f.read()
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{py_filename}'")
        return
    except Exception as e:
        print(f"Lỗi khi đọc file '{py_filename}': {e}")
        return

    # Phân tách nội dung script thành các cell dựa trên #%%
    # Cải tiến regex để bắt được cả #%% [markdown] và comment sau #%%
    cell_pattern = r'^\s*#\s*%%(?:\s*\[([^\]]+)\])?(?:\s+(.*))?$'
    cell_splits = re.split(cell_pattern, script_content, flags=re.MULTILINE)
    
    # Tổ chức lại kết quả split thành các tuple (content, cell_type, comment)
    cells = []
    i = 0
    # Cell đầu tiên (trước #%% đầu tiên)
    cells.append((cell_splits[0], "code", ""))  # Mặc định là code
    
    # Các cell còn lại
    while i + 3 <= len(cell_splits):
        cell_type = cell_splits[i+1] or "code"  # Mặc định là code nếu không chỉ định
        cell_comment = cell_splits[i+2] or ""
        cell_content = cell_splits[i+3]
        cells.append((cell_content, cell_type.lower(), cell_comment))
        i += 3
    
    test_cells = []

    for i, (cell_content, cell_type, cell_comment) in enumerate(cells):
        cell_content_stripped = cell_content.strip()
        if not cell_content_stripped:  # Bỏ qua cell rỗng
            continue
        
        source_lines = [line + '\n' for line in cell_content_stripped.splitlines()]
        # Xóa dòng cuối rỗng nếu có
        if source_lines and source_lines[-1].strip() == "":
            source_lines.pop()
        
        # Cell đầu tiên - xác định lại loại cell
        if i == 0 and all(line.strip().startswith('#') or not line.strip() for line in cell_content_stripped.splitlines()):
            cell_type = "markdown"
            # Xóa dấu # ở đầu mỗi dòng comment
            source_lines = [line.lstrip('# ').lstrip('#') + '\n' for line in cell_content_stripped.splitlines()]
        
        if cell_type == "markdown":
            # Nếu là markdown, xử lý nội dung đặc biệt
            if cell_content_stripped.startswith('"""') and cell_content_stripped.endswith('"""'):
                source_lines = [line + '\n' for line in cell_content_stripped.strip('"""').strip().splitlines()]
            elif cell_content_stripped.startswith("'''") and cell_content_stripped.endswith("'''"):
                source_lines = [line + '\n' for line in cell_content_stripped.strip("'''").strip().splitlines()]
            elif all(line.strip().startswith('#') for line in cell_content_stripped.splitlines() if line.strip()):
                # Xóa dấu # ở đầu mỗi dòng comment
                source_lines = [line.lstrip('# ').lstrip('#') + '\n' for line in cell_content_stripped.splitlines()]
            
            test_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": source_lines
            })
        else:  # Code cell
            test_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {
                    "tags": []
                },
                "outputs": [],
                "source": source_lines
            })

    # Cấu trúc tiêu chuẩn của file .ipynb
    test_json = {
        "cells": test_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    try:
        with open(ipynb_filename, 'w', encoding='utf-8') as f:
            json.dump(test_json, f, indent=2)
        print(f"Đã chuyển đổi thành công file '{py_filename}' sang '{ipynb_filename}'")
    except Exception as e:
        print(f"Lỗi khi ghi file '{ipynb_filename}': {e}")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        input_py_file = sys.argv[1]
        output_ipynb_file = sys.argv[2]
        py_to_ipynb(input_py_file, output_ipynb_file)
    else:
        print("Sử dụng: python convert.py <input_python_file.py> <output_test_file.ipynb>")
        print("Ví dụ: python convert.py test.py test.ipynb")
        # Chạy với file mặc định để demo
        print("\nĐang chạy demo với 'test.py' -> 'test.ipynb'")
        py_to_ipynb("test.py", "test.ipynb")
