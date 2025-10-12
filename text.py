import gradio as gr
import os

# ==========================================================
# !! 修改这里 !!
# 请将这个路径替换为你电脑上一个实际存在的视频文件的绝对路径
# 例如: "/home/ripemangobox/some_video.mp4"
# 或者在 Windows 上: "C:/Users/YourUser/Videos/some_video.mp4"
# ==========================================================
LOCAL_RENDER_DIR = "/home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D/HumanML3D/animations"
VIDEO_PATH = os.path.join(LOCAL_RENDER_DIR, "000016.mp4")

def play_local_video():
    # 检查文件是否存在，这是个好习惯
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"测试视频未找到，请检查路径: {VIDEO_PATH}")
    
    # Gradio 会为这个绝对路径创建一个可访问的 URL
    return VIDEO_PATH

with gr.Blocks() as demo:
    gr.Markdown("## 本地视频播放最小化测试")
    gr.Markdown(f"正在尝试播放: `{VIDEO_PATH}`")
    
    # 使用 gr.Video 组件，它会自动处理 /file= 路径
    video_player = gr.Video(label="本地视频")
    
    # 页面加载后自动调用函数
    demo.load(fn=play_local_video, inputs=None, outputs=video_player)

print("启动测试应用...")
# 在 launch() 函数中添加 allowed_paths 参数
# LOCAL_RENDER_DIR 就是您在代码开头定义的视频存放目录
demo.launch(share=True, allowed_paths=[LOCAL_RENDER_DIR])