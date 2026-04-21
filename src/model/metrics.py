import numpy as np


def print_latex_metrics(metrics):
    vals = [str(x).zfill(2) for x in [1, 2, 3, 5, 10]]
    t2m_keys = [f"t2m/R{i}" for i in vals] + ["t2m/MedR"]
    m2t_keys = [f"m2t/R{i}" for i in vals] + ["m2t/MedR"]

    keys = t2m_keys + m2t_keys

    def ff(val_):
        val = str(val_).ljust(5, "0")
        # make decimal fine when only one digit
        if val[1] == ".":
            val = str(val_).ljust(4, "0")
        return val

    str_ = "& " + " & ".join([ff(metrics[key]) for key in keys]) + r" \\"
    dico = {key: ff(metrics[key]) for key in keys}
    print(dico)
    print("Number of samples: {}".format(int(metrics["t2m/len"])))
    print(str_)


def all_contrastive_metrics(
    sims, emb=None, threshold=None, rounding=2, return_cols=False
):
    text_selfsim = None
    if emb is not None:
        text_selfsim = emb @ emb.T

    t2m_m, t2m_cols = contrastive_metrics(
        sims, text_selfsim, threshold, return_cols=True, rounding=rounding
    )
    m2t_m, m2t_cols = contrastive_metrics(
        sims.T, text_selfsim, threshold, return_cols=True, rounding=rounding
    )

    all_m = {}
    for key in t2m_m:
        all_m[f"t2m/{key}"] = t2m_m[key]
        all_m[f"m2t/{key}"] = m2t_m[key]

    all_m["t2m/len"] = float(len(sims))
    all_m["m2t/len"] = float(len(sims[0]))
    if return_cols:
        return all_m, t2m_cols, m2t_cols
    return all_m


def contrastive_metrics(
    sims,
    text_selfsim=None,
    threshold=None,
    return_cols=False,
    rounding=2,
    break_ties="averaging",
):
    n, m = sims.shape
    assert n == m
    num_queries = n

    dists = -sims
    sorted_dists = np.sort(dists, axis=1)
    # GT is in the diagonal
    gt_dists = np.diag(dists)[:, None]

    if text_selfsim is not None and threshold is not None:
        real_threshold = 2 * threshold - 1
        valid_matches = text_selfsim > real_threshold
        np.fill_diagonal(valid_matches, True)
        masked_dists = np.where(valid_matches, dists, np.nan)
        gt_dists = np.nanmin(masked_dists, axis=1, keepdims=True)

        if not np.all(np.isfinite(gt_dists)):
            fallback_gt = np.diag(dists)[:, None]
            gt_dists = np.where(np.isfinite(gt_dists), gt_dists, fallback_gt)

    if not np.all(np.isfinite(gt_dists)):
        raise ValueError("Non-finite GT distances encountered during metric evaluation.")

    if break_ties == "optimistically":
        cols = np.sum(sorted_dists < gt_dists, axis=1)
    elif break_ties == "averaging":
        lower = np.sum(sorted_dists < gt_dists, axis=1)
        upper = np.sum(sorted_dists <= gt_dists, axis=1) - 1
        cols = (lower + upper) / 2
    else:
        raise ValueError(f"Unsupported tie-breaking strategy: {break_ties}")

    if return_cols:
        return cols2metrics(cols, num_queries, rounding=rounding), cols
    return cols2metrics(cols, num_queries, rounding=rounding)


def cols2metrics(cols, num_queries, rounding=2):
    metrics = {}
    vals = [str(x).zfill(2) for x in [1, 2, 3, 5, 10]]
    for val in vals:
        metrics[f"R{val}"] = 100 * float(np.sum(cols < int(val))) / num_queries

    metrics["MedR"] = float(np.median(cols) + 1)

    if rounding is not None:
        for key in metrics:
            metrics[key] = round(metrics[key], rounding)
    return metrics
