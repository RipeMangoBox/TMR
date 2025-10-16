import os
from functools import partial

import torch
import numpy as np
import pandas as pd
import gradio as gr
import hydra  # <-- ADDED
from omegaconf import DictConfig # <-- ADDED
from demo.model import MDR_Music_Encoder
from demo.load import load_unit_embeddings, load_splits, load_json

import argparse

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
# Modified to accept video_dir and h3d_index as arguments
def retrieve(
    *,
    model,
    unit_motion_embs,
    all_keyids,
    music_feat_path,
    keyids_index,
    video_dir,          # <-- MODIFIED: Passed in
    h3d_index,          # <-- MODIFIED: Passed in
    split="test",
    nmax=8,
):
    keyids_in_split = [x for x in all_keyids[split] if x in keyids_index]
    index = [keyids_index[x] for x in keyids_in_split]
    unit_embs = unit_motion_embs[index]
    scores = model.compute_scores(music_feat_path, unit_embs=unit_embs)
    keyids_in_split = np.array(keyids_in_split)
    sorted_idxs = np.argsort(-scores)
    best_keyids = keyids_in_split[sorted_idxs]
    best_scores = scores[sorted_idxs]

    datas = []
    for keyid, score in zip(best_keyids, best_scores):
        if len(datas) == nmax:
            break
        
        video_path = os.path.join(video_dir, f"{keyid}.mp4") # <-- MODIFIED: Use passed-in var
        if not os.path.exists(video_path):
            continue

        data = {
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
# The decorator that enables Hydra
@hydra.main(version_base=None, config_path="configs", config_name="app_fd")
def retrieval_app(cfg: DictConfig) -> None:
    # ==============================================================================
    # 1. SETUP & CONFIGURATION (from Hydra's cfg object)
    # ==============================================================================
    
    # Assert that the dataset is correct
    assert cfg.dataset.name == "finedance", "This app is configured for the finedance dataset."
    
    # --- UI Strings and Styles (can also be moved to config if desired)
    WEBSITE = f"""
    <div class="embed_hidden">
    <h1 style='text-align: center'>{cfg.ui.title}</h1>
    <p>{cfg.ui.description}</p>
    </div>
    """

    # ==============================================================================
    # 2. LOADING MODELS AND DATA (Using paths from cfg)
    # ==============================================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading TMR model...")
    model = MDR_Music_Encoder(cfg.paths.model_dir, music_encoder_type='clap').to(device)
    
    print("Loading unit motion embeddings...")
    unit_motion_embs, keyids_index, index_keyids = load_unit_embeddings(cfg.paths.model_dir, cfg.dataset.name, device)
    
    all_keyids = load_splits(cfg.dataset.name, splits=["test", "val", "train", "all"])
    h3d_index = load_json(cfg.paths.annotations_json)

    print("Loading FineDance metadata...")
    dance_df, ui_splits, ui_names, ui_style1s, ui_style2s = load_dance_data(
        cfg.paths.metadata, cfg.paths.train_split, cfg.paths.val_split, cfg.paths.test_split
    )

    retrieve_function = partial(
        retrieve,
        model=model,
        unit_motion_embs=unit_motion_embs,
        all_keyids=all_keyids,
        keyids_index=keyids_index,
        video_dir=cfg.paths.video_dir,  # <-- Bind necessary args from cfg
        h3d_index=h3d_index             # <-- Bind necessary args from cfg
    )

    # ==============================================================================
    # 3. GRADIO UI AND INTERACTIVITY
    # ==============================================================================
    
    theme = gr.themes.Default(primary_hue="blue", secondary_hue="gray")

    # All Gradio logic is now inside the main function
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
                # Use values from the config file
                nvideo_slider = gr.Radio(cfg.ui.retrieval_n_choices, label="Videos to Retrieve", value=cfg.ui.retrieval_n_default)
                gr.Markdown("### Query Video")
                with gr.Column(elem_classes=["video-container"]):
                    query_video_display = gr.Video(label="Query", interactive=False, elem_id="query_video")
                    query_html_display = gr.HTML()

        gr.Markdown("---")
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
        # These are now nested functions, they can access `dance_df`, `cfg`, etc. from the outer scope
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
                num_outputs = 2 + len(result_video_outputs) * 2
                return [None] * num_outputs

            # Use cfg for the video directory path
            query_video_path = os.path.abspath(os.path.join(cfg.paths.video_dir, f"{file_name}.mp4"))
            query_html_info = '<div class="info-bar"><b>Query:</b> ' + file_name + '</div>'
            
            gallery_split = "test" if "Unseen" in gallery_split_choice else "all"
            retrieved_datas = retrieve_function(music_feat_path=query_video_path, split=gallery_split, nmax=nvids)
            
            result_paths = [data["absolute_path"] for data in retrieved_datas]
            result_htmls = []
            for data in retrieved_datas:
                info_content = (
                    f'<span style="margin-right: 24px;"><b>Score:</b> {data["score"]}</span>'
                    f'<span style="margin-right: 24px;"><b>KeyID:</b> {data["keyid"]}</span>'
                )
                result_htmls.append(f'<div class="info-bar">{info_content}</div>')

            result_paths += [None] * (len(result_video_outputs) - len(result_paths))
            result_htmls += [None] * (len(result_html_outputs) - len(result_htmls))
            
            return [query_video_path, query_html_info] + result_paths + result_htmls
            
        def clear_all():
            num_ui_to_clear = 2 + len(result_video_outputs) * 2
            return [None] * num_ui_to_clear + [gr.Dropdown(value="All"), gr.Dropdown(value="All"), gr.Dropdown(value="All"), gr.Dropdown(value="All"), gr.Dropdown(choices=[], value=None)]
        
        # --- Event Listeners (No changes needed here) ---
        for dropdown in [split_dropdown, name_dropdown, style1_dropdown, style2_dropdown]:
            dropdown.change(fn=update_file_list, inputs=[split_dropdown, name_dropdown, style1_dropdown, style2_dropdown], outputs=[file_dropdown])
        btn.click(fn=show_and_retrieve, inputs=[file_dropdown, nvideo_slider, split_dropdown], outputs=[query_video_display, query_html_display] + result_video_outputs + result_html_outputs)
        clear.click(fn=clear_all, outputs=[query_video_display, query_html_display] + result_video_outputs + result_html_outputs + [split_dropdown, name_dropdown, style1_dropdown, style2_dropdown, file_dropdown])
        demo.load(fn=update_file_list, inputs=[split_dropdown, name_dropdown, style1_dropdown, style2_dropdown], outputs=[file_dropdown])

    # --- Launch the server using settings from the config ---
    absolute_video_dir = os.path.abspath(cfg.paths.video_dir)
    print(f"Gradio is allowed to serve files from: {absolute_video_dir}")

    demo.launch(
        share=cfg.server.share,
        server_port=cfg.server.port, 
        allowed_paths=[absolute_video_dir]
    )

# --- Standard Hydra entry point ---
if __name__ == "__main__":
    retrieval_app()