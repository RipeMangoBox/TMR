import os
from functools import partial

import torch
import numpy as np
import pandas as pd
import gradio as gr

from demo.model import MDR_music_CLAP_encoder
from demo.load import load_unit_embeddings, load_splits, load_json

import argparse

# ==============================================================================
# 1. SETUP & CONFIGURATION
# ==============================================================================

# --- Paths (IMPORTANT: Please update these paths to match your project structure)
METADATA_PATH = "fd_index.csv"
split_dir = "datasets/annotations/finedance/splits"
TRAIN_SPLIT_PATH = f"{split_dir}/train.txt"
VAL_SPLIT_PATH = f"{split_dir}/val.txt"
TEST_SPLIT_PATH = f"{split_dir}/test.txt"
# This is the directory where your actual MP4 files are stored.
# The path is constructed as: os.path.join(VIDEO_DIR, new_name + ".mp4")
VIDEO_DIR = "/sata/public/ripemangobox/Motion/datasets/finedance/animations/fd_checked_music_partation"
# 通过 realpath 对其进行规范化，解析所有符号链接等
VIDEO_DIR = os.path.realpath(VIDEO_DIR)

# --- Original Model/Demo Setup
# parser for the model
# Using default values directly instead of parsing for simplicity in this script
# parser = argparse.ArgumentParser()
# parser.add_argument("--run_dir", default="RUN_DIR/MDR")
# args = parser.parse_args()
# MODEL_PATH = args.run_dir
MODEL_PATH = "RUN_DIR_fd" # Set your model path directly

DATASET = "finedance"
assert DATASET == "finedance"

WEBSITE = """
<div class="embed_hidden">
<h1 style='text-align: center'>TMR: Text-to-Motion Retrieval Using Contrastive 3D Human Motion Synthesis </h1>
<h3 style="text-align:center;">Music-to-Dance Adaptation</h3>
<p>
This space illustrates <a href='https://mathis.petrovich.fr/tmr/' target='_blank'><b>TMR</b></a>, adapted for music-to-dance retrieval. Select a dance video using the filters below. The system will use the video's text description to find other dances with similar motions from the FineDance gallery.
</p>
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
# 2. NEW DATA LOADING FUNCTION FOR FINEDANCE
# ==============================================================================

def load_dance_data(metadata_path, train_path, val_path, test_path):
    """
    Loads dance metadata from CSV and merges it with train/val/test splits.
    """
    try:
        df = pd.read_csv(metadata_path)
        # --- MODIFICATION START ---
        # Remove the .npy extension from the 'new_name' column
        df['new_name'] = df['new_name'].str.replace('.npy', '', regex=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"Metadata file not found at: {metadata_path}. Please check the METADATA_PATH variable.")

    # Read split files
    with open(train_path, 'r') as f:
        train_files = {line.strip() for line in f}
    with open(val_path, 'r') as f:
        val_files = {line.strip() for line in f}
    with open(test_path, 'r') as f:
        test_files = {line.strip() for line in f}

    # Assign split to each row
    def get_split(new_name):
        if new_name in train_files:
            return 'train'
        elif new_name in val_files:
            return 'val'
        elif new_name in test_files:
            return 'test'
        return 'unknown'

    df['split'] = df['new_name'].apply(get_split)
    
    # Pre-calculate unique values for filters
    splits = ['All', 'train', 'val', 'test']
    names = sorted(df['name'].unique().tolist())
    style1s = sorted(df['style1'].unique().tolist())
    style2s = sorted(df['style2'].dropna().unique().tolist()) # dropna for optional styles

    print("Dance data loaded and processed successfully.")
    return df, splits, names, style1s, style2s


# ==============================================================================
# 3. CORE RETRIEVAL LOGIC (Mostly Unchanged, with modifications for local videos)
# ==============================================================================

def get_video_html_from_data(data, width=700, height=700):
    """Generates HTML for a retrieved video."""
    url = data["url"]
    score = data["score"]
    text = data["text"]
    keyid = data["keyid"]
    
    title = f"""Score = {score}
Corresponding text: {text}
FineDance keyid: {keyid}"""

    # Gradio serves local files using the /file=... path
    # The original #t=start,end is not reliable for local files, so it's removed.
    video_html = f"""
<video class="retrieved_video" width="{width}" height="{height}" preload="auto" muted playsinline onpause="this.load()"
autoplay loop disablepictureinpicture title="{title}">
  <source src="{url}" type="video/mp4">
  Your browser does not support the video tag.
</video>
"""
    return video_html

def get_html_for_local_video(file_path, title="Query Video", width=700, height=700):
    """Generates HTML for a single local video, like the query video."""
    if not os.path.exists(file_path):
        return f"<p>Video not found at: {file_path}</p>"
        
    url = f"/file={os.path.abspath(file_path)}"
    
    video_html = f"""
<video class="retrieved_video" width="{width}" height="{height}" preload="auto" muted playsinline onpause="this.load()"
autoplay loop disablepictureinpicture title="{title}">
  <source src="{url}" type="video/mp4">
  Your browser does not support the video tag.
</video>
"""
    return video_html


def retrieve(
    *,
    model,
    unit_motion_embs,
    all_keyids,
    text,
    keyids_index,
    split="test",
    nmax=8,
):
    # This function is the core TMR model inference logic. It remains largely the same.
    # Note: The keyid mapping to video files is now different for finedance,
    # but the retrieval logic based on embeddings is the same.
    keyids_in_split = [x for x in all_keyids[split] if x in keyids_index]
    index = [keyids_index[x] for x in keyids_in_split]

    unit_embs = unit_motion_embs[index]

    scores = model.compute_scores(text, unit_embs=unit_embs, device=device)

    keyids_in_split = np.array(keyids_in_split)
    sorted_idxs = np.argsort(-scores)
    best_keyids = keyids_in_split[sorted_idxs]
    best_scores = scores[sorted_idxs]

    datas = []
    # This part needs to be adapted to your data structure.
    # Assuming `h3d_index` maps keyid to text and other info.
    for keyid, score in zip(best_keyids, best_scores):
        if len(datas) == nmax:
            break
        
        # We need a way to get from a TMR keyid back to a video file.
        # This is the most complex part of the adaptation. For now, let's assume
        # the keyid is the `new_name` without extension, or we can map it.
        # Let's assume `keyid` is the filename for simplicity here.
        video_path = os.path.join(VIDEO_DIR, f"{keyid}.mp4")
        if not os.path.exists(video_path):
            continue

        data = {
            "url": f"/file={os.path.abspath(video_path)}",
            "score": round(float(score), 2),
            "text": h3d_index.get(keyid, {}).get("annotations", [{}])[0].get("text", "N/A"),
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
unit_motion_embs, keyids_index, index_keyids = load_unit_embeddings(
    MODEL_PATH, DATASET, device
)
all_keyids = load_splits(DATASET, splits=["test", "all"])
# This index maps TMR's internal keyid to annotations like text
h3d_index = load_json(f"datasets/annotations/{DATASET}/annotations.json")

# --- Load Custom Dance Data
print("Loading FineDance metadata...")
dance_df, ui_splits, ui_names, ui_style1s, ui_style2s = load_dance_data(
    METADATA_PATH, TRAIN_SPLIT_PATH, VAL_SPLIT_PATH, TEST_SPLIT_PATH
)

# --- Create the main retrieval function with loaded models
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
    
    # --- UI for Filtering and Selection
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
            nvideo_slider = gr.Radio(
                [4, 8, 12, 16],
                label="Videos to Retrieve",
                value=8,
                info="Number of similar videos to display.",
            )
            gr.Markdown("### Query Video")
            query_video_display = gr.HTML(value=None)


    # --- UI for Displaying Results
    gr.Markdown("---")
    gr.Markdown("### Retrieved Videos")
    result_videos = []
    i = -1
    for _ in range(4): # 4 rows
        with gr.Row():
            for _ in range(4): # 4 columns
                i += 1
                video = gr.HTML()
                result_videos.append(video)

    # --- Backend Functions for UI Interactivity
    
    def update_filter_options(split):
        """Updates name/style dropdowns based on the selected split."""
        if split == "All":
            df_filtered = dance_df
        else:
            df_filtered = dance_df[dance_df['split'] == split]
        
        names = ["All"] + sorted(df_filtered['name'].unique().tolist())
        style1s = ["All"] + sorted(df_filtered['style1'].unique().tolist())
        style2s = ["All"] + sorted(df_filtered['style2'].dropna().unique().tolist())
        
        # When updating filters, also clear the file list
        return (
            gr.Dropdown(choices=names, value="All", interactive=True),
            gr.Dropdown(choices=style1s, value="All", interactive=True),
            gr.Dropdown(choices=style2s, value="All", interactive=True),
            gr.Dropdown(choices=[], value=None, interactive=True) # Clear file list
        )

    def update_file_list(split, name, style1, style2):
        """Updates the file dropdown based on all active filters."""
        df_filtered = dance_df.copy()
        
        if split != "All":
            df_filtered = df_filtered[df_filtered['split'] == split]
        if name != "All":
            df_filtered = df_filtered[df_filtered['name'] == name]
        if style1 != "All":
            df_filtered = df_filtered[df_filtered['style1'] == style1]
        if style2 != "All":
            df_filtered = df_filtered[df_filtered['style2'] == style2]
            
        files = sorted(df_filtered['new_name'].tolist())
        return gr.Dropdown(choices=files, value=None, interactive=True)

    def show_and_retrieve(file_name, nvids, gallery_split_choice):
        """Main function: displays query video and retrieves similar ones."""
        if not file_name:
            # Clear everything if no file is selected
            return [None] * (1 + len(result_videos))

        # --- 1. Display the selected query video
        # Construct path by adding .mp4, as file_name is now extension-less.
        query_video_path = os.path.join(VIDEO_DIR, f"{file_name}.mp4")
        query_html = get_html_for_local_video(query_video_path, title=f"Query: {file_name}")

        # --- 2. Find the text description for the selected file
        # --- NO CHANGE NEEDED HERE, but note the logic: ---
        # The file_name from the dropdown now matches the clean 'new_name' in the DataFrame.
        file_info = dance_df[dance_df['new_name'] == file_name].iloc[0]
        # 'query_keyid' is now correctly assigned the extension-less filename.
        # query_keyid = file_name
        # query_text = h3d_index.get(query_keyid, {}).get("annotations", [{}])[0].get("seg_id", "")
        
        # if not query_text:
        #     print(f"Warning: No text description found for {file_name}. Retrieval may not be meaningful.")
        #     # Still show the query video but can't retrieve
        #     return [query_html] + [None] * len(result_videos)

        # --- 3. Retrieve similar motions using the text
        # The gallery can be "all" or "test" (unseen)
        gallery_split = "test" if "Unseen" in gallery_split_choice else "all"
        
        retrieved_datas = retrieve_function(text=query_video_path, split=gallery_split, nmax=nvids)
        
        # --- 4. Generate HTML for retrieved videos
        result_htmls = [get_video_html_from_data(data) for data in retrieved_datas]
        # Pad with Nones if fewer videos are found than requested
        result_htmls += [None] * (len(result_videos) - len(result_htmls))
        
        return [query_html] + result_htmls
        
    def clear_all():
        """Clears all inputs and outputs."""
        return (
            [None] * (1 + len(result_videos)) + # Clear query video + result videos
            [gr.Dropdown(value="All"), gr.Dropdown(value="All"), gr.Dropdown(value="All"), gr.Dropdown(value="All"), gr.Dropdown(choices=[], value=None)]
        )

    # --- Event Listeners
    split_dropdown.change(
        fn=update_filter_options,
        inputs=[split_dropdown],
        outputs=[name_dropdown, style1_dropdown, style2_dropdown, file_dropdown]
    )
    
    for dropdown in [split_dropdown, name_dropdown, style1_dropdown, style2_dropdown]:
        dropdown.change(
            fn=update_file_list,
            inputs=[split_dropdown, name_dropdown, style1_dropdown, style2_dropdown],
            outputs=[file_dropdown]
        )

    btn.click(
        fn=show_and_retrieve,
        inputs=[file_dropdown, nvideo_slider, split_dropdown], # Using split_dropdown to decide gallery
        outputs=[query_video_display] + result_videos
    )
    
    clear.click(
        fn=clear_all,
        outputs=[query_video_display] + result_videos + [split_dropdown, name_dropdown, style1_dropdown, style2_dropdown, file_dropdown]
    )


demo.launch(
    share=True, 
    allowed_paths=[
        "/sata/public/ripemangobox/Motion/datasets/finedance/animations/fd_checked_music_partation",
    ]
)