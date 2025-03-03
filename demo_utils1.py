import os

# 更改全局临时目录
new_tmp_dir = "./demo/gradio_bg"
os.makedirs(new_tmp_dir, exist_ok=True)

os.environ['GRADIO_TEMP_DIR'] = new_tmp_dir


