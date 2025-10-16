import os
from functools import partial

import torch
import numpy as np
import pandas as pd
import gradio as gr

from demo.model1 import MDR_music_CLAP_encoder
from demo.load import load_unit_embeddings, load_splits, load_json

import argparse

# ==============================================================================
# 1. SETUP & CONFIGURATION
# ==============================================================================

# --- Paths (请确保路径正确)
METADATA_PATH = "fd_index.csv"
split_dir = "datasets/annotations/finedance/splits"
TRAIN_SPLIT_PATH = f"{split_dir}/train.txt"
VAL_SPLIT_PATH = f"{split_dir}/val.txt"
TEST_SPLIT_PATH = f"{split_dir}/test.txt"
VIDEO_DIR = "../datasets/finedance/animations/fd_checked_music_partation" 

MODEL_PATH = "RUN_DIR_fd"
DATASET = "finedance"
assert DATASET == "finedance"

WEBSITE = """
<div class="embed_hidden">
<h1 style='text-align: center'>Music-to-Dance Retrieval</h1>
<p>
选择一个舞蹈视频作为查询，系统将检索并展示与之音乐特征相似的其他舞蹈。
</p>
</div>
"""


# --- 修改后 ---
# --- 最终修改版 ---
CSS = """
/* 1. 全局宽度设置 */
#root, .gradio-container {
    width: 100% !important;
    max-width: 100% !important;
}

/* 2. .video-container (gr.Group) 的样式 */
.video-container {
    padding: 0 !important;
    min-width: 150px !important;
    /* 使用统一的浅灰背景，作为整个卡片的底色 */
    background-color: #f0f0f0;
    border: 1px solid #ddd; /* 可选：加个细边框更清晰 */
    border-radius: var(--block-radius) !important;
    overflow: hidden;
}

/* 3. 视频本身的样式 */
.video-container video {
    width: 100%;
    display: block;
    height: auto;
    aspect-ratio: 16 / 9;
    object-fit: cover;
}

/* 4. 信息条的样式 —— 无内框，仅文字 */
# .info-bar {
#     width: 100%;
#     /* ▼▼▼ 关键：背景透明，无内边距或极小 ▼▼▼ */
#     background-color: transparent;
#     color: black;
#     padding: 2px 4px; /* 保留少量上下内边距，避免文字贴边 */
#     font-size: 20px;
#     text-align: left;
#     # box-sizing: border-box;
#     # word-wrap: break-word;
#     margin: 0; /* 确保无外边距 */
# }

.info-bar {
    width: 100%;
    background-color: transparent;
    color: #333; /* 比黑色柔和，更易读 */
    font-size: 18px;
    line-height: 1.2; /* 让行高贴近字体，避免多余空间 */
    text-align: left;
    box-sizing: border-box;
    word-wrap: break-word;
    margin: 0;
    /* 可选：让容器不撑开，只包裹文字 */
    display: block;
}
"""


# --- 新增 JavaScript 代码 ---
# [MODIFIED] --- 替换整个 JavaScript 字符串 ---
js_functions = """
// 辅助函数：健壮地获取所有视频元素
function getAllVideos() {
    const videos = [];

    // 正确获取查询视频
    const queryContainer = document.getElementById('query_video');
    if (queryContainer) {
        const videoEl = queryContainer.querySelector('video');
        if (videoEl) videos.push(videoEl);
    }

    // 正确获取结果视频
    for (let i = 0; i < 16; i++) {
        const resultContainer = document.getElementById(`result_video_${i}`);
        if (resultContainer) {
            const videoEl = resultContainer.querySelector('video');
            if (videoEl) videos.push(videoEl);
        }
    }
    return videos;
}

// 后续函数无需修改
function playAllVideos() {
    getAllVideos().forEach(v => v.play());
    return [];
}

function pauseAllVideos() {
    getAllVideos().forEach(v => v.pause());
    return [];
}

function muteAllVideos() {
    getAllVideos().forEach(v => v.muted = true);
    return [];
}

function unmuteAllVideos() {
    getAllVideos().forEach(v => v.muted = false);
    return [];
}
"""

# ==============================================================================
# 2. DATA LOADING (与之前相同)
# ==============================================================================

def load_dance_data(metadata_path, train_path, val_path, test_path):
    try:
        df = pd.read_csv(metadata_path)
        df['new_name'] = df['new_name'].str.replace('.npy', '', regex=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"Metadata file not found at: {metadata_path}. Please check the METADATA_PATH variable.")

    with open(train_path, 'r') as f: train_files = {line.strip() for line in f}
    with open(val_path, 'r') as f: val_files = {line.strip() for line in f}
    with open(test_path, 'r') as f: test_files = {line.strip() for line in f}

    def get_split(new_name):
        if new_name in train_files: return 'train'
        elif new_name in val_files: return 'val'
        elif new_name in test_files: return 'test'
        return 'unknown'

    df['split'] = df['new_name'].apply(get_split)
    
    splits = ['All', 'train', 'val', 'test']
    names = sorted(df['name'].unique().tolist())
    style1s = sorted(df['style1'].unique().tolist())
    style2s = sorted(df['style2'].dropna().unique().tolist())

    print("Dance data loaded and processed successfully.")
    return df, splits, names, style1s, style2s


# ==============================================================================
# 3. CORE RETRIEVAL LOGIC (修改以返回绝对路径)
# ==============================================================================

def retrieve(
    *,
    model,
    unit_motion_embs,
    all_keyids,
    music_feat_path,  # 输入从 text 变为 music_feat_path
    keyids_index,
    split="test",
    nmax=8,
):
    keyids_in_split = [x for x in all_keyids[split] if x in keyids_index]
    index = [keyids_index[x] for x in keyids_in_split]

    unit_embs = unit_motion_embs[index]

    # 假设 model.compute_scores 接受的是音乐特征路径
    scores = model.compute_scores(music_feat_path, unit_embs=unit_embs)

    keyids_in_split = np.array(keyids_in_split)
    sorted_idxs = np.argsort(-scores)
    best_keyids = keyids_in_split[sorted_idxs]
    best_scores = scores[sorted_idxs]

    datas = []
    for keyid, score in zip(best_keyids, best_scores):
        if len(datas) == nmax:
            break
        
        video_path = os.path.join(VIDEO_DIR, f"{keyid}.mp4")
        if not os.path.exists(video_path):
            continue

        data = {
            ## --- 关键修改 ---
            # 我们需要的是绝对路径，供 gr.Video 使用
            "absolute_path": os.path.abspath(video_path),
            "score": round(float(score), 2),
            "text": h3d_index.get(keyid, {}).get("annotations", [{}])[0].get("seg_id", "N/A"),
            "keyid": keyid,
        }
        datas.append(data)
    return datas


# ==============================================================================
# 4. LOADING MODELS AND DATA
# ==============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load TMR Model and Embeddings
print("Loading TMR model...")
model = MDR_music_CLAP_encoder(MODEL_PATH).to(device)
print("Loading unit motion embeddings...")
unit_motion_embs, keyids_index, index_keyids = load_unit_embeddings(MODEL_PATH, DATASET, device)
all_keyids = load_splits(DATASET, splits=["test", "all"])
h3d_index = load_json(f"datasets/annotations/{DATASET}/annotations.json")

print("Loading FineDance metadata...")
dance_df, ui_splits, ui_names, ui_style1s, ui_style2s = load_dance_data(
    METADATA_PATH, TRAIN_SPLIT_PATH, VAL_SPLIT_PATH, TEST_SPLIT_PATH
)

retrieve_function = partial(
    retrieve,
    model=model,
    unit_motion_embs=unit_motion_embs,
    all_keyids=all_keyids,
    keyids_index=keyids_index,
)

# ==============================================================================
# 5. GRADIO UI AND INTERACTIVITY
# ==============================================================================

theme = gr.themes.Default(primary_hue="blue", secondary_hue="gray")

with gr.Blocks(css=CSS, theme=theme) as demo:
    gr.Markdown(WEBSITE)
    
    with gr.Row():
        with gr.Column(scale=2):
            split_dropdown = gr.Dropdown(choices=ui_splits, label="Split", value="All")
            name_dropdown = gr.Dropdown(choices=["All"] + ui_names, label="Name", value="All")
            style1_dropdown = gr.Dropdown(choices=["All"] + ui_style1s, label="Style 1", value="All")
            style2_dropdown = gr.Dropdown(choices=["All"] + ui_style2s, label="Style 2", value="All")
            file_dropdown = gr.Dropdown([], label="Select a File", info="File list updates based on filters above.")
            
            with gr.Row():
                btn = gr.Button("Retrieve Similar Dances", variant="primary")
                clear = gr.Button("Clear", variant="secondary")
        
        with gr.Column(scale=1):
            nvideo_slider = gr.Radio([4, 8, 12, 16], label="Videos to Retrieve", value=8)
            gr.Markdown("### Query Video")
            # 为查询视频创建一个容器，并放入Video和HTML组件
            with gr.Column(elem_classes=["video-container"]):
                query_video_display = gr.Video(label="Query", interactive=False, elem_id="query_video")
                query_html_display = gr.HTML()

    gr.Markdown("---")
    
    # # --- 新增：全局控制按钮 ---
    # with gr.Row():
    #     play_all_btn = gr.Button("▶️ Play All")
    #     pause_all_btn = gr.Button("⏸️ Pause All")
    #     mute_all_btn = gr.Button("🔇 Mute All")
    #     unmute_all_btn = gr.Button("🔊 Unmute All")
        
    gr.Markdown("### Retrieved Videos")

    result_video_outputs = []
    result_html_outputs = []
    for row in range(4):
        with gr.Row():
            for col in range(4):
                i = row * 4 + col
                with gr.Group(elem_classes=["video-container"]):
                    video_comp = gr.Video(interactive=False, show_label=False, elem_id=f"result_video_{i}")
                    html_comp = gr.HTML()
                    result_video_outputs.append(video_comp)
                    result_html_outputs.append(html_comp)
    
    # --- Backend Functions ---
    def update_file_list(split, name, style1, style2):
        df_filtered = dance_df.copy()
        if split != "All": df_filtered = df_filtered[df_filtered['split'] == split]
        if name != "All": df_filtered = df_filtered[df_filtered['name'] == name]
        if style1 != "All": df_filtered = df_filtered[df_filtered['style1'] == style1]
        if style2 != "All": df_filtered = df_filtered[df_filtered['style2'] == style2]
        files = sorted(df_filtered['new_name'].tolist())
        return gr.Dropdown(choices=files, value=None, interactive=True)

    def show_and_retrieve(file_name, nvids, gallery_split_choice):
        if not file_name:
            # 清空时，需要为所有组件返回None
            num_outputs = 2 + len(result_video_outputs) * 2
            return [None] * num_outputs

        # 1. 准备查询视频的绝对路径
        query_video_path = os.path.abspath(os.path.join(VIDEO_DIR, f"{file_name}.mp4"))
        
        # 2. 准备查询视频的HTML信息
        # query_info_title = f"<b>查询视频:</b><br>{file_name}" # 使用 <br> 换行
        # query_html_info = f'<div class="info-bar">{query_info_title}</div>'
        # 查询信息
        query_html_info = '<div class="info-bar"><b>Query:</b> ' + file_name + '</div>'
        
        # 4. 执行检索
        gallery_split = "test" if "Unseen" in gallery_split_choice else "all"
        # 假设 retrieve_function 返回包含 'absolute_path', 'score', 'keyid', 'text' 的字典列表
        retrieved_datas = retrieve_function(music_feat_path=query_video_path, split=gallery_split, nmax=nvids)
        
        # 5. 分别构建视频路径列表和HTML信息列表
        result_paths = []
        result_htmls = []
        for data in retrieved_datas:
            # 添加视频路径
            result_paths.append(data["absolute_path"])
            
            # 构建信息条的内容 (使用 <br> 进行换行)
            info_content = (
                f'<span style="margin-right: 24px;"><b>Score:</b> {data["score"]}</span>'
                f'<span style="margin-right: 24px;"><b>KeyID:</b> {data["keyid"]}</span>'
            )
            # 创建HTML覆盖层字符串
            result_htmls.append(f'<div class="info-bar">{info_content}</div>')

        # 6. 用None填充两个列表，以匹配UI组件的总数
        result_paths += [None] * (len(result_video_outputs) - len(result_paths))
        result_htmls += [None] * (len(result_html_outputs) - len(result_htmls))
        
        # 7. 返回组合后的最终列表
        return [query_video_path, query_html_info] + result_paths + result_htmls
        
    def clear_all():
        # 计算需要清空的UI组件总数：查询(video+html) + 结果(video+html)
        num_ui_to_clear = 2 + len(result_video_outputs) * 2
        # 返回对应数量的None，再加上下拉菜单的重置值
        return [None] * num_ui_to_clear + [gr.Dropdown(value="All"), gr.Dropdown(value="All"), gr.Dropdown(value="All"), gr.Dropdown(value="All"), gr.Dropdown(choices=[], value=None)]
    
    # --- Event Listeners ---
    for dropdown in [split_dropdown, name_dropdown, style1_dropdown, style2_dropdown]:
        dropdown.change(
            fn=update_file_list,
            inputs=[split_dropdown, name_dropdown, style1_dropdown, style2_dropdown],
            outputs=[file_dropdown]
        )

    btn.click(
        fn=show_and_retrieve,
        inputs=[file_dropdown, nvideo_slider, split_dropdown],
        outputs=[query_video_display, query_html_display] + result_video_outputs + result_html_outputs
    )

    # 修复 clear.click
    clear.click(
        fn=clear_all,
        # 这个列表必须包含所有需要清空的UI组件
        outputs=[query_video_display, query_html_display] + result_video_outputs + result_html_outputs + [split_dropdown, name_dropdown, style1_dropdown, style2_dropdown, file_dropdown]
    )
    
    # 当应用加载时，使用下拉框的默认值 ("All") 来初始化文件列表
    demo.load(
        fn=update_file_list,
        inputs=[split_dropdown, name_dropdown, style1_dropdown, style2_dropdown],
        outputs=[file_dropdown]
    )

## --- 关键修改 ---
# 确保在 launch 时提供绝对路径
absolute_video_dir = os.path.abspath(VIDEO_DIR)
print(f"Gradio is allowed to serve files from: {absolute_video_dir}")

demo.launch(
    share=True, 
    allowed_paths=[absolute_video_dir]
)