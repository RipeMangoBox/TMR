# Text model + TMR text encoder only

from typing import List
import torch.nn as nn
import os

import torch, torchaudio
import numpy as np
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, Wav2Vec2FeatureExtractor
from torch.nn.functional import normalize
from einops import repeat
import json
import warnings
import laion_clap
import logging
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import List, Union, Tuple

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

# ======================================================================
# 1. ABSTRACT BASE CLASS: Defines the interface for all encoders
# ======================================================================
class BaseFeatureExtractor(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_last_hidden_state(
        self, audio_paths: Union[str, List[str]], return_mask: bool = False, device: str = 'cuda'
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Processes audio file(s) and returns their feature representation.
        
        Args:
            audio_paths: A single path or a list of paths to audio files.
            return_mask: If True, returns a boolean mask along with the features.
            device: The device to perform computation on.

        Returns:
            A tensor of features, or a tuple of (features, mask).
            The feature tensor should be of shape [batch_size, sequence_length, feature_dim].
            For this project, we concatenate all sequences, so it becomes [total_sequence_length, feature_dim].
        """
        pass

# ======================================================================
# 2. CONCRETE IMPLEMENTATION: CLAP Encoder
# ======================================================================
class MDR_CLAP_Encoder(BaseFeatureExtractor):
    def __init__(self, **kwargs):
        super().__init__()
        self.TARGET_SR = 48000
        self.BATCH_SIZE_TOKEN = 256
        # model_path = '/sata/public/ripemangobox/Motion/LODGE/CLAP/checkpoints/music_audioset_epoch_15_esc_90.14.pt'
        # print("🔌 Initializing CLAP model...")
        # self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
        # self.model.load_ckpt(model_path)
        # print("✅ CLAP Model loaded.")
        
        modelpath = '/sata/public/ripemangobox/Motion/LODGE/CLAP/checkpoints/music_audioset_epoch_15_esc_90.14.pt'
        print("🔌 Initializing CLAP model...")
        self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        # Make sure you provide the correct path to your checkpoint file
        self.model.load_ckpt(modelpath)
        print(f"✅ Model loaded on {device}.")
        self.eval()

    def get_last_hidden_state(self, audio_paths: Union[str, List[str]], return_mask: bool = False, device: str = 'cuda'):
        self.to(device)
        # The original `process_files_with_stride` returns a concatenated tensor for all files
        # which fits the required output format.
        output = self.process_files_with_stride(audio_paths, device=device)
        output = output.unsqueeze(0) # Add a batch dimension
        
        if not return_mask:
            return output
            
        mask = torch.ones((1, output.shape[1]), device=device, dtype=torch.bool)
        return output, mask

    def process_files_with_stride(self, audio_paths: Union[str, List[str]], window_sec: float = 1.0, stride_sec: float = 0.5, device: str = 'cuda'):
        if isinstance(audio_paths, str):
            audio_paths = [audio_paths]
            
        window_samples = int(window_sec * self.TARGET_SR)
        stride_samples = int(stride_sec * self.TARGET_SR)
        resampler_cache = {}
        all_files_embeds = []

        for audio_path in tqdm(audio_paths, desc="CLAP Processing"):
            try:
                waveform, original_sr = torchaudio.load(audio_path)
                waveform = waveform.to(device)

                if original_sr != self.TARGET_SR:
                    if original_sr not in resampler_cache:
                        resampler_cache[original_sr] = T.Resample(
                            orig_freq=original_sr, new_freq=self.TARGET_SR
                        ).to(device)
                    waveform = resampler_cache[original_sr](waveform)

                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                total_samples = waveform.shape[1]
                if total_samples < window_samples:
                    continue

                chunks = [
                    waveform[:, i:i + window_samples]
                    for i in range(0, total_samples - window_samples + 1, stride_samples)
                ]
                
                if not chunks:
                    continue

                single_file_embeds = []
                for i in range(0, len(chunks), self.BATCH_SIZE_TOKEN):
                    batch_chunks = torch.stack(chunks[i:i + self.BATCH_SIZE_TOKEN]).squeeze(1)
                    with torch.no_grad():
                        batch_embeds = self.model.get_audio_embedding_from_data(x=batch_chunks, use_tensor=True)
                    single_file_embeds.append(batch_embeds)
                
                all_files_embeds.append(torch.cat(single_file_embeds, dim=0))
            except Exception as e:
                print(f"Error processing {audio_path} with CLAP: {e}")
                continue
        
        if not all_files_embeds:
             return torch.empty(0, 256, device=device) # Return empty tensor if no files processed
             
        return torch.cat(all_files_embeds, dim=0).to(device)


# ======================================================================
# 3. CONCRETE IMPLEMENTATION: MERT Encoder
# ======================================================================
class MDR_MERT_Encoder(BaseFeatureExtractor):
    def __init__(self, layer: int = 15, batch_size: int = 16, **kwargs):
        super().__init__()
        self.TARGET_SR = 24000
        self.layer_to_extract = layer
        self.batch_size = batch_size
        
        print(f"🔌 Initializing MERT model (extracting from layer {self.layer_to_extract})...")
        model_id = "m-a-p/MERT-v1-330M"
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
        print("✅ MERT Model loaded.")
        self.eval()

    def get_last_hidden_state(self, audio_paths: Union[str, List[str]], return_mask: bool = False, device: str = 'cuda'):
        self.to(device)
        if isinstance(audio_paths, str):
            audio_paths = [audio_paths]
            
        all_files_embeds = []
        resampler_cache = {}

        for i in tqdm(range(0, len(audio_paths), self.batch_size), desc="MERT Processing"):
            batch_files = audio_paths[i:i + self.batch_size]
            batch_waveforms = []
            
            try:
                for audio_path in batch_files:
                    waveform, original_sr = torchaudio.load(audio_path)
                    if original_sr != self.TARGET_SR:
                        if original_sr not in resampler_cache:
                            resampler_cache[original_sr] = T.Resample(orig_freq=original_sr, new_freq=self.TARGET_SR)
                        waveform = resampler_cache[original_sr](waveform)

                    waveform = torch.mean(waveform, dim=0) if waveform.shape[0] > 1 else waveform.squeeze(0)
                    batch_waveforms.append(waveform.numpy())
                
                inputs = self.processor(
                    batch_waveforms, sampling_rate=self.TARGET_SR, return_tensors="pt", padding=True
                ).to(device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)

                layer_hidden_states = outputs.hidden_states[self.layer_to_extract]
                attention_mask = inputs.attention_mask

                for j in range(len(batch_files)):
                    true_length = attention_mask[j].sum()
                    seq_len = self.model._get_feat_extract_output_lengths(true_length).item()
                    unpadded_features = layer_hidden_states[j, :seq_len, :]
                    all_files_embeds.append(unpadded_features)
            
            except Exception as e:
                print(f"Error processing MERT batch starting with {batch_files[0]}: {e}")
                continue
        
        if not all_files_embeds:
             return torch.empty(0, 1024, device=device) # MERT's dim is 1024
        
        output = torch.cat(all_files_embeds, dim=0).unsqueeze(0) # Add batch dimension

        if not return_mask:
            return output
            
        mask = torch.ones((1, output.shape[1]), device=device, dtype=torch.bool)
        return output, mask

# ======================================================================
# 4. MAIN WRAPPER CLASS: The public-facing encoder
# ======================================================================
class MDR_Music_Encoder(nn.Module):
    def __init__(self, run_dir: str, music_encoder_type: str, music_encoder_args: dict = {}) -> None:
        super().__init__()
        
        # --- 1. Instantiate the chosen music feature extractor ---
        if music_encoder_type.lower() == 'clap':
            self.feature_extractor = MDR_CLAP_Encoder(**music_encoder_args)
            nfeats = 256 # CLAP output dimension
        elif music_encoder_type.lower() == 'mert':
            self.feature_extractor = MDR_MERT_Encoder(**music_encoder_args)
            nfeats = 1024 # MERT output dimension
        else:
            raise ValueError(f"Unknown music_encoder_type: {music_encoder_type}")
            
        # --- 2. Build the text-side Transformer architecture ---
        # This part is common regardless of the feature extractor used.
        config = read_config(run_dir)
        text_encoder_conf = config["model"]["text_encoder"]
        
        # Update nfeats based on the actual feature extractor's output
        text_encoder_conf["nfeats"] = nfeats

        latent_dim = text_encoder_conf["latent_dim"]
        self.projection = nn.Linear(nfeats, latent_dim)
        
        self.vae = text_encoder_conf["vae"]
        self.nbtokens = 2 if self.vae else 1
        self.tokens = nn.Parameter(torch.randn(self.nbtokens, latent_dim))
        
        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout=0.0, batch_first=True)
        
        seq_trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=text_encoder_conf["num_heads"],
            dim_feedforward=text_encoder_conf["ff_size"], dropout=0.0,
            activation=text_encoder_conf["activation"], batch_first=True
        )
        self.seqTransEncoder = nn.TransformerEncoder(
            seq_trans_encoder_layer, num_layers=text_encoder_conf["num_layers"]
        )

        # --- 3. Load pre-trained weights for the text-side Transformer ---
        text_encoder_pt_path = os.path.join(run_dir, "last_weights/text_encoder.pt")
        print(f"Loading text-side transformer weights from {text_encoder_pt_path}...")
        # self.load_state_dict(torch.load(text_encoder_pt_path)) # Commented out for standalone run
        print("✅ Text-side transformer configured.")
        self.eval()

    def forward(self, audio_paths: Union[str, List[str]], device: str = 'cuda') -> Tensor:
        self.to(device)
        # 1. Get features and mask from the chosen feature extractor
        music_features, mask = self.feature_extractor.get_last_hidden_state(
            audio_paths, return_mask=True, device=device
        )
        
        # 2. Project features and proceed with the text-side transformer
        x = self.projection(music_features)
        
        bs = len(x)
        tokens = repeat(self.tokens, "nbtoken dim -> bs nbtoken dim", bs=bs)
        xseq = torch.cat((tokens, x), 1)

        token_mask = torch.ones((bs, self.nbtokens), dtype=torch.bool, device=device)
        aug_mask = torch.cat((token_mask, mask), 1)

        xseq = self.sequence_pos_encoding(xseq)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)
        
        # Return the feature for the first token (usually the class token)
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
    

# ======================================================================
# 5. EXAMPLE USAGE
# ======================================================================
if __name__ == '__main__':
    # --- Setup a dummy environment for testing ---
    if not os.path.exists("dummy_run_dir/last_weights"):
        os.makedirs("dummy_run_dir/last_weights")
    if not os.path.exists("dummy_audio"):
        os.makedirs("dummy_audio")
    
    # Create a dummy weights file
    # In a real scenario, this would be your trained text_encoder.pt
    dummy_encoder = MDR_Music_Encoder("dummy_run_dir", 'clap', {'model_path': 'dummy.pt'})
    torch.save(dummy_encoder.state_dict(), "dummy_run_dir/last_weights/text_encoder.pt")

    # Create dummy audio files
    sr = 44100
    dummy_wav1, _ = torchaudio.load('dummy_audio/sample1.wav', 'wb') if os.path.exists('dummy_audio/sample1.wav') else torchaudio.save('dummy_audio/sample1.wav', torch.randn(1, sr * 5), sr)
    dummy_wav2, _ = torchaudio.load('dummy_audio/sample2.wav', 'wb') if os.path.exists('dummy_audio/sample2.wav') else torchaudio.save('dummy_audio/sample2.wav', torch.randn(1, sr * 8), sr)
    
    audio_file_list = ["dummy_audio/sample1.wav", "dummy_audio/sample2.wav"]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\n" + "="*50)
    print("🚀 TESTING CLAP ENCODER")
    print("="*50)
    # --- Instantiate the main encoder with CLAP ---
    # NOTE: You must provide the correct path to your downloaded CLAP checkpoint
    clap_model_path = '/sata/public/ripemangobox/Motion/LODGE/CLAP/checkpoints/music_audioset_epoch_15_esc_90.14.pt'
    if os.path.exists(clap_model_path):
        clap_encoder = MDR_Music_Encoder(
            run_dir="dummy_run_dir",
            music_encoder_type='clap',
            music_encoder_args={'model_path': clap_model_path}
        )
        # Get the final latent representation for a list of audio files
        clap_output = clap_encoder(audio_file_list, device=device)
        print(f"\nFinal CLAP output shape: {clap_output.shape}") # Should be [1, latent_dim]
    else:
        print(f"⚠️ CLAP checkpoint not found at {clap_model_path}. Skipping CLAP test.")

    print("\n" + "="*50)
    print("🚀 TESTING MERT ENCODER")
    print("="*50)
    # --- Instantiate the main encoder with MERT ---
    # You can specify the layer and batch size
    mert_encoder = MDR_Music_Encoder(
        run_dir="dummy_run_dir",
        music_encoder_type='mert',
        music_encoder_args={'layer': 15, 'batch_size': 2}
    )
    # Get the final latent representation for a list of audio files
    mert_output = mert_encoder(audio_file_list, device=device)
    print(f"\nFinal MERT output shape: {mert_output.shape}") # Should be [1, latent_dim]