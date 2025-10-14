import os
import gradio as gr

# ==============================================================================
# 1. 核心配置（请仔细检查此路径是否正确）
# ==============================================================================

# 视频文件所在的目录
VIDEO_DIR = "/sata/public/ripemangobox/Motion/datasets/finedance/animations/fd_checked_music_partation" 

# 我们要测试显示的具体视频文件名
TEST_VIDEO_FILENAME = "000001.mp4"

# Gradio 界面相关的静态信息
WEBSITE = """
<div class="embed_hidden">
<h1 style='text-align: center'>视频显示功能测试</h1>
<p>点击下方按钮，测试是否可以正常加载并显示指定的本地视频文件。</p>
</div>
"""

CSS = """
.retrieved_video {
    position: relative;
    margin: 0;
    box-shadow: var(--block-shadow);
    border-width: var(--block-border-width);
    border-color: #000000;
    border-radius: var(--block-radius);
    background: var(--block-background-fill);
    width: 100%;
    line-height: var(--line-sm);
}
"""

# ==============================================================================
# 2. 后端逻辑（用于生成视频的 HTML）
# ==============================================================================

def display_test_video():
    """
    一个简单的函数，用于构建测试视频的完整路径并生成HTML。
    """
    # 构造视频文件的绝对路径
    video_path = os.path.abspath(os.path.join(VIDEO_DIR, TEST_VIDEO_FILENAME))
    
    # 【关键调试步骤】检查文件是否存在
    if not os.path.exists(video_path):
        error_message = (
            f"<h2>错误：文件未找到！</h2>"
            f"<p>请检查以下路径是否正确，并且文件确实存在：</p>"
            f"<p><code>{video_path}</code></p>"
            f"<p>同时请确认启动脚本的当前工作目录。</p>"
        )
        print(f"ERROR: File not found at {video_path}")
        return error_message

    print(f"SUCCESS: Found video file at {video_path}")
    
    # 使用 Gradio 的文件服务语法构建 URL
    url = f"/file={video_path}"
    
    # 生成视频的 HTML 代码
    video_html = f"""
<video class="retrieved_video" width="700" height="700" preload="auto" muted playsinline
autoplay loop disablepictureinpicture title="测试视频">
  <source src="{url}" type="video/mp4">
  您的浏览器不支持 video 标签。
</video>
"""
    return video_html

# ==============================================================================
# 3. Gradio 界面布局
# ==============================================================================

theme = gr.themes.Default(primary_hue="blue", secondary_hue="gray")

with gr.Blocks(css=CSS, theme=theme) as demo:
    gr.Markdown(WEBSITE)
    
    with gr.Row():
        btn = gr.Button("显示测试视频", variant="primary")
        
    with gr.Row():
        # 用于显示视频或错误信息的 HTML 组件
        video_display_area = gr.HTML(value=None)

    # --- 事件监听器 ---
    btn.click(
        fn=display_test_video,
        inputs=None,
        outputs=video_display_area
    )

# ==============================================================================
# 4. 启动应用
# ==============================================================================

# 【关键配置】确保 VIDEO_DIR 被添加到了允许访问的路径列表中
demo.launch(
    share=True, 
    allowed_paths=[VIDEO_DIR]
)