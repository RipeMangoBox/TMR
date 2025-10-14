"""
Contrastive Language-Audio Pretraining Model from LAION
--------------------------------------------------------
Paper: https://arxiv.org/abs/2211.06687
Authors (equal contributions): Ke Chen, Yusong Wu, Tianyu Zhang, Yuchen Hui
Support: LAION
"""

import numpy as np
import librosa
import torch
import laion_clap

# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype('float32')


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype('int16')

model = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
model.load_ckpt('CLAP/checkpoints/music_audioset_epoch_15_esc_90.14.pt')

# Directly get audio embeddings from audio files
audio_file = [
    '/sata/public/ripemangobox/Motion/LODGE/data/finedance/music_wav/001.wav',
    '/sata/public/ripemangobox/Motion/LODGE/data/finedance/music_wav/002.wav'
]
audio_embed = model.get_audio_embedding_from_filelist(x = audio_file, use_tensor=False)
print(audio_embed[:,-20:])
print(audio_embed.shape)

# Get audio embeddings from audio data
audio_data, _ = librosa.load('/sata/public/ripemangobox/Motion/LODGE/data/finedance/music_wav/001.wav', sr=48000) # sample rate should be 48000
audio_data = audio_data.reshape(1, -1) # Make it (1,T) or (N,T)
audio_embed = model.get_audio_embedding_from_data(x = audio_data, use_tensor=False)
print(audio_embed[:,-20:])
print(audio_embed.shape)

# Directly get audio embeddings from audio files, but return torch tensor
audio_file = [
    '/sata/public/ripemangobox/Motion/LODGE/data/finedance/music_wav/001.wav',
    '/sata/public/ripemangobox/Motion/LODGE/data/finedance/music_wav/002.wav'
]
audio_embed = model.get_audio_embedding_from_filelist(x = audio_file, use_tensor=True)
print(audio_embed[:,-20:])
print(audio_embed.shape)

# Get audio embeddings from audio data
audio_data, _ = librosa.load('/sata/public/ripemangobox/Motion/LODGE/data/finedance/music_wav/001.wav', sr=48000) # sample rate should be 48000
audio_data = audio_data.reshape(1, -1) # Make it (1,T) or (N,T)
audio_data = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float() # quantize before send it in to the model
audio_embed = model.get_audio_embedding_from_data(x = audio_data, use_tensor=True)
print(audio_embed[:,-20:])
print(audio_embed.shape)

# Get text embedings from texts:
text_data = ["I love the contrastive learning", "I love the pretrain model"] 
text_embed = model.get_text_embedding(text_data)
print(text_embed)
print(text_embed.shape)

# Get text embedings from texts, but return torch tensor:
text_data = ["I love the contrastive learning", "I love the pretrain model"] 
text_embed = model.get_text_embedding(text_data, use_tensor=True)
print(text_embed)
print(text_embed.shape)

