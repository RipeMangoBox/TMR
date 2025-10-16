# Text model + TMR text encoder only

from typing import List
import torch.nn as nn
import os

import torch, torchaudio
import numpy as np
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import normalize
from einops import repeat
import json
import warnings
import laion_clap
import logging
from tqdm import tqdm

logger = logging.getLogger("torch.distributed.nn.jit.instantiator")
logger.setLevel(logging.ERROR)


warnings.filterwarnings(
    "ignore", "The PyTorch API of nested tensors is in prototype stage*"
)

warnings.filterwarnings("ignore", "Converting mask without torch.bool dtype to bool*")

torch.set_float32_matmul_precision("high")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False) -> None:
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
        else:
            x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


def read_config(run_dir: str):
    path = os.path.join(run_dir, "config.json")
    with open(path, "r") as f:
        config = json.load(f)
    return config


class TMR_text_encoder(nn.Module):
    def __init__(self, run_dir: str) -> None:
        config = read_config(run_dir)
        modelpath = config["data"]["text_to_token_emb"]["modelname"]

        text_encoder_conf = config["model"]["text_encoder"]

        vae = text_encoder_conf["vae"]
        latent_dim = text_encoder_conf["latent_dim"]
        ff_size = text_encoder_conf["ff_size"]
        num_layers = text_encoder_conf["num_layers"]
        num_heads = text_encoder_conf["num_heads"]
        activation = text_encoder_conf["activation"]
        nfeats = text_encoder_conf["nfeats"]

        super().__init__()

        # Projection of the text-outputs into the latent space
        self.projection = nn.Linear(nfeats, latent_dim)
        self.vae = vae
        self.nbtokens = 2 if vae else 1

        self.tokens = nn.Parameter(torch.randn(self.nbtokens, latent_dim))
        self.sequence_pos_encoding = PositionalEncoding(
            latent_dim, dropout=0.0, batch_first=True
        )

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=0.0,
            activation=activation,
            batch_first=True,
        )

        self.seqTransEncoder = nn.TransformerEncoder(
            seq_trans_encoder_layer, num_layers=num_layers
        )

        text_encoder_pt_path = os.path.join(run_dir, "last_weights/text_encoder.pt")
        state_dict = torch.load(text_encoder_pt_path)
        self.load_state_dict(state_dict)

        from transformers import logging

        # load text model
        logging.set_verbosity_error()

        # Tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)

        # Text model
        self.text_model = AutoModel.from_pretrained(modelpath)
        # Then configure the model
        self.text_encoded_dim = self.text_model.config.hidden_size
        self.eval()

    def get_last_hidden_state(self, texts: List[str], return_mask: bool = False):
        encoded_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        output = self.text_model(**encoded_inputs.to(self.text_model.device))
        if not return_mask:
            return output.last_hidden_state
        return output.last_hidden_state, encoded_inputs.attention_mask.to(dtype=bool)

    def forward(self, texts: List[str]) -> Tensor:
        text_encoded, mask = self.get_last_hidden_state(texts, return_mask=True)

        x = self.projection(text_encoded)

        device = x.device
        bs = len(x)

        tokens = repeat(self.tokens, "nbtoken dim -> bs nbtoken dim", bs=bs)
        xseq = torch.cat((tokens, x), 1)

        token_mask = torch.ones((bs, self.nbtokens), dtype=bool, device=device)
        aug_mask = torch.cat((token_mask, mask), 1)

        # add positional encoding
        xseq = self.sequence_pos_encoding(xseq)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)
        return final[:, 0]

    # compute score for retrieval
    def compute_scores(self, texts, unit_embs=None, embs=None):
        # not both empty
        assert not (unit_embs is None and embs is None)
        # not both filled
        assert not (unit_embs is not None and embs is not None)

        output_str = False
        # if one input, squeeze the output
        if isinstance(texts, str):
            texts = [texts]
            output_str = True

        # compute unit_embs from embs if not given
        if embs is not None:
            unit_embs = normalize(embs)

        with torch.no_grad():
            latent_unit_texts = normalize(self.forward(texts))
            # compute cosine similarity between 0 and 1
            scores = (unit_embs @ latent_unit_texts.T).T / 2 + 0.5
            scores = scores.cpu().numpy()

        if output_str:
            scores = scores[0]

        return scores

class MDR_music_CLAP_encoder(nn.Module):
    def __init__(self, run_dir: str) -> None:
        super().__init__()
        self.BATCH_SIZE_TOKEN = 256
        self.BATCH_SIZE_SENT = 32
        self.TARGET_SR = 48000
        

        config = read_config(run_dir)
        text_encoder_conf = config["model"]["text_encoder"]

        vae = text_encoder_conf["vae"]
        latent_dim = text_encoder_conf["latent_dim"]
        ff_size = text_encoder_conf["ff_size"]
        num_layers = text_encoder_conf["num_layers"]
        num_heads = text_encoder_conf["num_heads"]
        activation = text_encoder_conf["activation"]
        nfeats = text_encoder_conf["nfeats"]

        super().__init__()

        # Projection of the text-outputs into the latent space
        self.projection = nn.Linear(nfeats, latent_dim)
        self.vae = vae
        self.nbtokens = 2 if vae else 1

        self.tokens = nn.Parameter(torch.randn(self.nbtokens, latent_dim))
        self.sequence_pos_encoding = PositionalEncoding(
            latent_dim, dropout=0.0, batch_first=True
        )

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=0.0,
            activation=activation,
            batch_first=True,
        )

        self.seqTransEncoder = nn.TransformerEncoder(
            seq_trans_encoder_layer, num_layers=num_layers
        )

        text_encoder_pt_path = os.path.join(run_dir, "last_weights/text_encoder.pt")
        state_dict = torch.load(text_encoder_pt_path)
        self.load_state_dict(state_dict)

        # ---------- CLAP -----------------
        modelpath = '/sata/public/ripemangobox/Motion/LODGE/CLAP/checkpoints/music_audioset_epoch_15_esc_90.14.pt'
        print("🔌 Initializing CLAP model...")
        self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        # Make sure you provide the correct path to your checkpoint file
        self.model.load_ckpt(modelpath)
        print(f"✅ Model loaded on {device}.")
        
        from transformers import logging

        # load text model
        logging.set_verbosity_error()

        # Tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.eval()

    def get_last_hidden_state(self, texts: List[str], return_mask: bool = False, device = 'cuda'):
        output = self.process_files_with_stride(texts, device=device)
        output = output.unsqueeze(0).to(device)
        if not return_mask:
            return output
        return output, torch.ones((1, output.shape[1]), device=device).to(dtype=bool)

    def forward(self, texts: List[str], device = 'cuda') -> Tensor:
        text_encoded, mask = self.get_last_hidden_state(texts, return_mask=True, device=device)

        x = self.projection(text_encoded)

        device = x.device
        bs = len(x)

        tokens = repeat(self.tokens, "nbtoken dim -> bs nbtoken dim", bs=bs)
        xseq = torch.cat((tokens, x), 1)

        token_mask = torch.ones((bs, self.nbtokens), dtype=bool, device=device)
        aug_mask = torch.cat((token_mask, mask), 1)

        # add positional encoding
        xseq = self.sequence_pos_encoding(xseq)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)
        return final[:, 0]

    # compute score for retrieval
    def compute_scores(self, texts, unit_embs=None, embs=None, device='cuda'):
        # not both empty
        assert not (unit_embs is None and embs is None)
        # not both filled
        assert not (unit_embs is not None and embs is not None)

        output_str = False
        # if one input, squeeze the output
        if isinstance(texts, str):
            texts = [texts]
            output_str = True

        # compute unit_embs from embs if not given
        if embs is not None:
            unit_embs = normalize(embs)

        with torch.no_grad():
            latent_unit_texts = normalize(self(texts, device=device))
            # compute cosine similarity between 0 and 1
            scores = (unit_embs @ latent_unit_texts.T).T / 2 + 0.5
            scores = scores.cpu().numpy()

        if output_str:
            scores = scores[0]

        return scores

    def process_files_with_stride(self, audio_files: list, window_sec: float = 1.0, stride_sec: float = 0.5, device = 'cuda'):
        """
        Processes audio files by splitting them into chunks with a given window and stride.
        This version adaptively handles different original sample rates.
        """
        window_samples = int(window_sec * self.TARGET_SR)
        stride_samples = int(stride_sec * self.TARGET_SR)
        resampler_cache = {}
        # --- MODIFICATION END ---

        ret_embs = []
        if isinstance(audio_files, str):
            audio_files = [audio_files]
        for audio_path in tqdm(audio_files, desc=f"Processing with stride={stride_sec}s"):
            waveform, original_sr = torchaudio.load(audio_path)
            waveform = waveform.to(device)

            # Resample only if the audio's sample rate is not the target rate
            if original_sr != self.TARGET_SR:
                # --- MODIFICATION START ---
                # Check if a resampler for this specific original_sr already exists
                if original_sr not in resampler_cache:
                    print(f"\nINFO: Creating new resampler for {original_sr} Hz -> {self.TARGET_SR} Hz")
                    # Create and cache a new resampler
                    resampler_cache[original_sr] = torchaudio.transforms.Resample(
                        orig_freq=original_sr, new_freq=self.TARGET_SR
                    ).to(device)
                
                # Use the appropriate resampler from the cache
                waveform = resampler_cache[original_sr](waveform)
                # --- MODIFICATION END ---

            # Ensure audio is mono for processing
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            assert waveform.shape[0] == 1
            
            total_samples = waveform.shape[1]
            
            if total_samples < window_samples:
                print(f"⚠️ Skipping {os.path.basename(audio_path)}: shorter than window size after resampling.")
                continue

            chunks = []
            for start_sample in range(0, total_samples - window_samples + 1, stride_samples):
                end_sample = start_sample + window_samples
                chunks.append(waveform[:, start_sample:end_sample])
            
            if not chunks:
                continue

            all_embeds = []
            for i in range(0, len(chunks), self.BATCH_SIZE_TOKEN):
                batch_chunks = chunks[i:i + self.BATCH_SIZE_TOKEN]
                
                stacked_tensor = torch.stack(batch_chunks)
                batch_tensor = stacked_tensor.squeeze(1)
                
                with torch.no_grad():
                    batch_embeds = self.model.get_audio_embedding_from_data(x=batch_tensor, use_tensor=True)
                all_embeds.append(batch_embeds.cpu())

            final_embeds = torch.cat(all_embeds, dim=0)
            ret_embs.append(final_embeds)
            
        return torch.cat(ret_embs, dim=0)
            
    def process_full_files(self, audio_files: list, device = 'cuda'):
        """
        Processes audio files by getting a single embedding for the entire file.
        NOTE: The underlying `get_audio_embedding_from_filelist` function handles
        resampling automatically, so no changes are needed here.
        """
        
        all_embeds = []
        for i in tqdm(range(0, len(audio_files), self.BATCH_SIZE_SENT), desc="Processing full files"):
            batch_files = audio_files[i:i + self.BATCH_SIZE_SENT]
            
            with torch.no_grad():
                # This function loads each file, checks its SR, and resamples internally.
                batch_embeds = self.model.get_audio_embedding_from_filelist(x=batch_files, use_tensor=True)
                all_embeds.append(batch_embeds)
        final_embeds = torch.cat(all_embeds, dim=0)
        return batch_embeds

        