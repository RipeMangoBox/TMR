import os
import orjson
import json
import logging
import torch
from torch import Tensor
import numpy as np
from tqdm import tqdm

from abc import ABC, abstractmethod

from src.model import TextToEmb


logger = logging.getLogger(__name__)


class TextEmbeddings(ABC):
    name = ...

    def __init__(
        self,
        modelname: str,
        path: str = "",
        device: str = "cpu",
        preload: bool = True,
        disable: bool = False,
    ):
        self.modelname = modelname
        self.embeddings_folder = os.path.join(path, self.name)
        self.cache = {}
        self.device = device
        self.disable = disable

        if preload and not disable:
            try:
                self.load_embeddings()
            except FileNotFoundError as exc:
                logger.warning(
                    "Precomputed text embeddings not found for path=%s model=%s; "
                    "falling back to on-the-fly encoding. Missing file: %s",
                    path,
                    modelname,
                    exc,
                )
                self.embeddings_index = {}
        else:
            self.embeddings_index = {}

    @abstractmethod
    def load_model(self) -> None:
        ...

    @abstractmethod
    def load_embeddings(self) -> None:
        ...

    @abstractmethod
    def get_embedding(self, text: str) -> Tensor:
        ...

    def __contains__(self, text):
        return text in self.embeddings_index

    def get_model(self):
        model = getattr(self, "model", None)
        if model is None:
            model = self.load_model()
        return model

    def __call__(self, texts):
        if self.disable:
            return texts

        squeeze = False
        if isinstance(texts, str):
            texts = [texts]
            squeeze = True

        x_dict_lst = []
        # one at a time here
        for text in texts:
            # Precomputed in advance
            if text in self:
                x_dict = self.get_embedding(text)
            # Already computed during the session
            elif text in self.cache:
                x_dict = self.cache[text]
            # Load the text model (if not already loaded) + compute on the fly
            else:
                model = self.get_model()
                x_dict = model(text)
                self.cache[text] = x_dict
            x_dict_lst.append(x_dict)

        if squeeze:
            return x_dict_lst[0]
        return x_dict_lst


class TokenEmbeddings(TextEmbeddings):
    name = "token_embeddings"

    def load_model(self):
        self.model = TextToEmb(self.modelname, mean_pooling=False, device=self.device)
        return self.model

    def load_embeddings(self):
        self.embeddings_big = torch.from_numpy(
            np.load(os.path.join(self.embeddings_folder, self.modelname + ".npy"))
        ).to(dtype=torch.float, device=self.device)
        self.embeddings_slice = np.load(
            os.path.join(self.embeddings_folder, self.modelname + "_slice.npy")
        )
        self.embeddings_index = load_json(
            os.path.join(self.embeddings_folder, self.modelname + "_index.json")
        )
        self.text_dim = self.embeddings_big.shape[-1]

    def get_embedding(self, text):
        # Precomputed in advance
        index = self.embeddings_index[text]
        begin, end = self.embeddings_slice[index]
        embedding = self.embeddings_big[begin:end]
        x_dict = {"x": embedding, "length": len(embedding)}
        return x_dict


class SentenceEmbeddings(TextEmbeddings):
    name = "sent_embeddings"

    def load_model(self):
        self.model = TextToEmb(self.modelname, mean_pooling=True, device=self.device)
        return self.model

    def load_embeddings(self):
        self.embeddings = torch.from_numpy(
            np.load(os.path.join(self.embeddings_folder, self.modelname + ".npy"))
        ).to(dtype=torch.float, device=self.device)
        self.embeddings_index = load_json(
            os.path.join(self.embeddings_folder, self.modelname + "_index.json")
        )
        assert len(self.embeddings_index) == len(self.embeddings)

        self.text_dim = self.embeddings.shape[-1]

    def get_embedding(self, text):
        index = self.embeddings_index[text]
        embedding = self.embeddings[index]
        return embedding.to(self.device)


def load_json(json_path):
    with open(json_path, "rb") as ff:
        return orjson.loads(ff.read())


def load_annotations(path, name="annotations.json"):
    json_path = os.path.join(path, name)
    return load_json(json_path)


def write_json(data, path):
    with open(path, "w") as ff:
        ff.write(json.dumps(data, indent=4))


def _deduplicate_texts(texts):
    ordered = {}
    for text in texts:
        if text is None:
            continue
        cleaned = str(text).strip()
        if not cleaned or cleaned in ordered:
            continue
        ordered[cleaned] = None
    return list(ordered.keys())


def _split_text_batches(texts, batch_size=None):
    if not texts:
        return []
    if batch_size is not None and batch_size > 0:
        return [texts[idx : idx + batch_size] for idx in range(0, len(texts), batch_size)]

    num_splits = max(1, min(100, len(texts)))
    return [batch.tolist() for batch in np.array_split(np.asarray(texts, dtype=object), num_splits)]


def save_token_embeddings_from_texts(
    path,
    texts,
    modelname="sentence-transformers/all-mpnet-base-v2",
    device="cuda",
    batch_size=None,
    progress_desc="Encoding token embeddings",
):
    model = TextToEmb(modelname, device=device)

    path = os.path.join(path, TokenEmbeddings.name)
    ptpath = os.path.join(path, f"{modelname}.npy")
    slicepath = os.path.join(path, f"{modelname}_slice.npy")
    jsonpath = os.path.join(path, f"{modelname}_index.json")

    path = os.path.split(ptpath)[0]
    os.makedirs(path, exist_ok=True)

    all_texts = _deduplicate_texts(texts)
    if not all_texts:
        raise ValueError("No valid texts found for token embedding export.")

    logger.info("Saving token embeddings for %s unique texts", len(all_texts))
    all_texts_batched = _split_text_batches(all_texts, batch_size=batch_size)

    nb_tokens_so_far = 0
    big_tensor = []
    index = []
    for all_texts_batch in tqdm(
        all_texts_batched,
        desc=progress_desc,
        unit="batch",
    ):
        x_dict = model(list(all_texts_batch))

        tensor = x_dict["x"]
        nb_tokens = x_dict["length"]

        tensor_no_padding = [x[:n].cpu() for x, n in zip(tensor, nb_tokens)]
        tensor_concat = torch.cat(tensor_no_padding)

        big_tensor.append(tensor_concat)
        ends = torch.cumsum(nb_tokens, 0)
        begins = torch.cat((0 * ends[[0]], ends[:-1]))

        ends += nb_tokens_so_far
        begins += nb_tokens_so_far
        nb_tokens_so_far += len(tensor_concat)

        index.append(torch.stack((begins, ends)).T)

    big_tensor = torch.cat(big_tensor).cpu().numpy()
    index = torch.cat(index).cpu().numpy()

    np.save(ptpath, big_tensor)
    np.save(slicepath, index)
    print(f"{ptpath} written")
    print(f"{slicepath} written")

    dico = {txt: i for i, txt in enumerate(all_texts)}
    write_json(dico, jsonpath)
    print(f"{jsonpath} written")


def save_sent_embeddings_from_texts(
    path,
    texts,
    modelname="sentence-transformers/all-mpnet-base-v2",
    device="cuda",
    batch_size=None,
    progress_desc="Encoding sentence embeddings",
):
    model = TextToEmb(modelname, mean_pooling=True, device=device)

    path = os.path.join(path, SentenceEmbeddings.name)
    ptpath = os.path.join(path, f"{modelname}.npy")
    jsonpath = os.path.join(path, f"{modelname}_index.json")

    path = os.path.split(ptpath)[0]
    os.makedirs(path, exist_ok=True)

    all_texts = _deduplicate_texts(texts)
    if not all_texts:
        raise ValueError("No valid texts found for sentence embedding export.")

    logger.info("Saving sentence embeddings for %s unique texts", len(all_texts))
    all_texts_batched = _split_text_batches(all_texts, batch_size=batch_size)

    embeddings = []
    for all_texts_batch in tqdm(
        all_texts_batched,
        desc=progress_desc,
        unit="batch",
    ):
        embedding = model(list(all_texts_batch)).cpu()
        embeddings.append(embedding)

    embeddings = torch.cat(embeddings).numpy()
    np.save(ptpath, embeddings)
    print(f"{ptpath} written")

    dico = {txt: i for i, txt in enumerate(all_texts)}
    write_json(dico, jsonpath)
    print(f"{jsonpath} written")


def save_token_embeddings(
    path, modelname="sentence-transformers/all-mpnet-base-v2", device="cuda"
):
    annotations = load_annotations(path)

    # fetch all the texts
    all_texts = []
    for dico in annotations.values():
        for lst in dico["annotations"]:
            all_texts.append(lst["text"])
    save_token_embeddings_from_texts(
        path,
        all_texts,
        modelname=modelname,
        device=device,
        progress_desc="Encoding token embeddings",
    )


def save_sent_embeddings(
    path, modelname="sentence-transformers/all-mpnet-base-v2", device="cuda"
):
    annotations = load_annotations(path)

    # fetch all the texts
    all_texts = []
    for dico in annotations.values():
        for lst in dico["annotations"]:
            all_texts.append(lst["text"])
    save_sent_embeddings_from_texts(
        path,
        all_texts,
        modelname=modelname,
        device=device,
        progress_desc="Encoding sentence embeddings",
    )
