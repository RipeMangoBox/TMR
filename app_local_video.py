from functools import partial
import os
import torch
import numpy as np
import gradio as gr

from demo.model import TMR_text_encoder
from demo.load import load_unit_embeddings, load_splits, load_json

import argparse

# ======================
# 新增：是否使用本地视频
# ======================
use_local_video = True  # <-- 设置为 True 使用本地视频
LOCAL_RENDER_DIR = "/home/ripemangobox/Coding/Github/Motion/datasets/FineDance/FineDance/animations"  # 本地视频存放目录（可修改）
# parser for the model
parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", default="models/tmr_FineDance_guoh3dfeats")
args = parser.parse_args()
MODEL_PATH = args.run_dir

# For now, only compatible with the finadance dataset
DATASET = "finadance"
assert DATASET == "finadance"

WEBSITE = """
<div class="embed_hidden">
<h1 style='text-align: center'>TMR: Text-to-Motion Retrieval Using Contrastive 3D Human Motion Synthesis </h1>

<h2 style='text-align: center'>
<a href="https://mathis.petrovich.fr" target="_blank"><nobr>Mathis Petrovich</nobr></a> &emsp;
<a href="https://ps.is.mpg.de/~black" target="_blank"><nobr>Michael J. Black</nobr></a> &emsp;
<a href="https://imagine.enpc.fr/~varolg" target="_blank"><nobr>Güll Varol</nobr></a>
</h2>

<h2 style='text-align: center'>
<nobr>ICCV 2023</nobr>
</h2>

<h3 style="text-align:center;">
<a target="_blank" href="https://arxiv.org/abs/2305.00976"> <button type="button" class="btn btn-primary btn-lg"> Paper </button></a>
<a target="_blank" href="https://github.com/Mathux/TMR"> <button type="button" class="btn btn-primary btn-lg"> Code </button></a>
<a target="_blank" href="https://mathis.petrovich.fr/tmr"> <button type="button" class="btn btn-primary btn-lg"> Webpage </button></a>
<a target="_blank" href="https://mathis.petrovich.fr/tmr/tmr.bib"> <button type="button" class="btn btn-primary btn-lg"> BibTex </button></a>
</h3>

<h3> Description </h3>
<p>
This space illustrates <a href='https://mathis.petrovich.fr/tmr/' target='_blank'><b>TMR</b></a>, a method for text-to-motion retrieval. Given a gallery of 3D human motions (which can be unseen during training) and a text query, the goal is to search for motions which are close to the text query.
</p>
</div>
"""

EXAMPLES = [
    "A person is walking slowly",
    "A person is walking in a circle",
    "A person is jumping rope",
    "Someone is doing a backflip",
    "A person is doing a moonwalk",
    "A person walks forward and then turns back",
    "Picking up an object",
    "A person is swimming in the sea",
    "A human is squatting",
    "Someone is jumping with one foot",
    "A person is chopping vegetables",
    "Someone walks backward",
    "Somebody is ascending a staircase",
    "A person is sitting down",
    "A person is taking the stairs",
    "Someone is doing jumping jacks",
    "The person walked forward and is picking up his toolbox",
    "The person angrily punching the air",
]

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

.contour_video {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    z-index: var(--layer-5);
    border-radius: var(--block-radius);
    background: var(--background-fill-primary);
    padding: 0 var(--size-6);
    max-height: var(--size-screen-h);
    overflow: hidden;
}
"""

DEFAULT_TEXT = "A person is "

# ==============================================================================
# 修改后的函数：支持本地视频
# ==============================================================================
def FineDance_keyid_to_babel_rendered_url(h3d_index, amass_to_babel, keyid, use_local_video=False):
    global LOCAL_RENDER_DIR

    # 如果使用本地视频，跳过镜像和 HumanAct12 的过滤（可选保留）
    if not use_local_video:
        # 原逻辑：跳过镜像
        if "M" in keyid:
            return None

        dico = h3d_index[keyid]
        path = dico["path"]

        # 跳过 HumanAct12
        if "humanact12" in path:
            return None

        if path not in amass_to_babel:
            return None

        babel_id = amass_to_babel[path].zfill(6)
        url = f"https://babel-renders.s3.eu-central-1.amazonaws.com/{babel_id}.mp4"

        ann = dico["annotations"][0]
        start = ann["start"]
        end = ann["end"]
        text = ann["text"]

    else:
        # 本地视频逻辑
        dico = h3d_index[keyid]
        path = dico["path"]

        if path not in amass_to_babel:
            return None

        babel_id = amass_to_babel[path].zfill(6)
        local_path = os.path.join(LOCAL_RENDER_DIR, f"{babel_id}.mp4")

        if not os.path.exists(local_path):
            print(f"Warning: Local video not found: {local_path}")
            return None

        # Gradio 静态文件服务语法
        url = f"/file={os.path.abspath(local_path)}"

        ann = dico["annotations"][0]
        start = ann["start"]
        end = ann["end"]
        text = ann["text"]

    data = {
        "url": url,
        "start": start,
        "end": end,
        "text": text,
        "keyid": keyid,
        "babel_id": babel_id,
        "path": path,
    }

    return data


def retrieve(
    *,
    model,
    keyid_to_url,
    unit_motion_embs,
    all_keyids,
    text,
    keyids_index,
    index_keyids,
    split="test",
    nmax=8,
):
    keyids = [x for x in all_keyids[split] if x in keyids_index]
    index = [keyids_index[x] for x in keyids]

    unit_embs = unit_motion_embs[index]

    scores = model.compute_scores(text, unit_embs=unit_embs)

    keyids = np.array(keyids)
    sorted_idxs = np.argsort(-scores)
    best_keyids = keyids[sorted_idxs]
    best_scores = scores[sorted_idxs]

    datas = []
    for keyid, score in zip(best_keyids, best_scores):
        if len(datas) == nmax:
            break

        data = keyid_to_url(keyid)
        if data is None:
            continue
        data["score"] = round(float(score), 2)
        datas.append(data)
    return datas


def get_video_html(data, video_id, width=700, height=700):
    url = data["url"]
    start = data["start"]
    end = data["end"]
    score = data["score"]
    text = data["text"]
    keyid = data["keyid"]
    babel_id = data["babel_id"]
    path = data["path"]

    # 注意：本地视频无法可靠使用 #t=start,end，但保留以兼容远程
    trim = f"#t={start},{end}"
    title = f"""Score = {score}

Corresponding text: {text}

FineDance keyid: {keyid}

BABEL keyid: {babel_id}

AMASS path: {path}"""

    video_html = f"""
<video class="retrieved_video" width="{width}" height="{height}" preload="auto" muted playsinline onpause="this.load()"
autoplay loop disablepictureinpicture id="{video_id}" title="{title}">
  <source src="{url}{trim}" type="video/mp4">
  Your browser does not support the video tag.
</video>
"""
    return video_html


def retrieve_component(retrieve_function, text, splits_choice, nvids, n_component=24):
    if text == DEFAULT_TEXT or text == "" or text is None:
        return [None for _ in range(n_component)]

    nvids = min(nvids, n_component)

    if "Unseen" in splits_choice:
        split = "test"
    else:
        split = "all"

    datas = retrieve_function(text=text, split=split, nmax=nvids)
    htmls = [get_video_html(data, idx) for idx, data in enumerate(datas)]
    htmls = htmls + [None for _ in range(max(0, n_component - nvids))]
    return htmls


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOADING
model = TMR_text_encoder(MODEL_PATH).to(device)

unit_motion_embs, keyids_index, index_keyids = load_unit_embeddings(
    MODEL_PATH, DATASET, device
)

all_keyids = load_splits(DATASET, splits=["test", "all"])

h3d_index = load_json(f"datasets/annotations/{DATASET}/annotations.json")
amass_to_babel = load_json("demo/amass_to_babel.json")

# ================================
# 修改：传入 use_local_video 参数
# ================================
keyid_to_url = partial(FineDance_keyid_to_babel_rendered_url, h3d_index, amass_to_babel, use_local_video=use_local_video)
retrieve_function = partial(
    retrieve,
    model=model,
    keyid_to_url=keyid_to_url,
    unit_motion_embs=unit_motion_embs,
    all_keyids=all_keyids,
    keyids_index=keyids_index,
    index_keyids=index_keyids,
)

# DEMO
theme = gr.themes.Default(primary_hue="blue", secondary_hue="gray")
retrieve_and_show = partial(retrieve_component, retrieve_function)

with gr.Blocks(css=CSS, theme=theme) as demo:
    gr.Markdown(WEBSITE)
    videos = []

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Column(scale=2):
                text = gr.Textbox(
                    placeholder="Type the motion you want to search with a sentence",
                    show_label=True,
                    label="Text prompt",
                    value=DEFAULT_TEXT,
                )
            with gr.Column(scale=1):
                btn = gr.Button("Retrieve", variant="primary")
                clear = gr.Button("Clear", variant="secondary")

            with gr.Row():
                with gr.Column(scale=1):
                    splits_choice = gr.Radio(
                        ["All motions", "Unseen motions"],
                        label="Gallery of motion",
                        value="All motions",
                        info="The motion gallery is coming from FineDance",
                    )

                with gr.Column(scale=1):
                    nvideo_slider = gr.Radio(
                        [4, 8, 12, 16, 24],
                        label="Videos",
                        value=8,
                        info="Number of videos to display",
                    )

        with gr.Column(scale=2):
            def retrieve_example(text, splits_choice, nvideo_slider):
                return retrieve_and_show(text, splits_choice, nvideo_slider)

            examples = gr.Examples(
                examples=[[x, None, None] for x in EXAMPLES],
                inputs=[text, splits_choice, nvideo_slider],
                examples_per_page=20,
                run_on_click=False,
                cache_examples=False,
                fn=retrieve_example,
                outputs=[],
            )

    i = -1
    for _ in range(6):
        with gr.Row():
            for _ in range(4):
                i += 1
                video = gr.HTML()
                videos.append(video)

    examples.outputs = videos

    def load_example(example_id):
        processed_example = examples.non_none_processed_examples[example_id]
        return gr.utils.resolve_singleton(processed_example)

    examples.dataset.click(
        load_example,
        inputs=[examples.dataset],
        outputs=examples.inputs_with_examples,
        show_progress=False,
        postprocess=False,
        queue=False,
    ).then(fn=retrieve_example, inputs=examples.inputs, outputs=videos)

    btn.click(
        fn=retrieve_and_show,
        inputs=[text, splits_choice, nvideo_slider],
        outputs=videos,
    )
    text.submit(
        fn=retrieve_and_show,
        inputs=[text, splits_choice, nvideo_slider],
        outputs=videos,
    )
    splits_choice.change(
        fn=retrieve_and_show,
        inputs=[text, splits_choice, nvideo_slider],
        outputs=videos,
    )
    nvideo_slider.change(
        fn=retrieve_and_show,
        inputs=[text, splits_choice, nvideo_slider],
        outputs=videos,
    )

    def clear_videos():
        return [None for _ in range(24)] + [DEFAULT_TEXT]

    clear.click(fn=clear_videos, outputs=videos + [text])

demo.launch(share=True)