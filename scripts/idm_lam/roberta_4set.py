"""Supervised RoBERTa (online, strict causal) on the held-out 4 sets, scored with the SAME
scorers as the IDM experiments (dts_scoring for tiage/dialseg/superseg; ami_scoring for AMI).

Checkpoint/data were wiped -> retrain roberta-base token-classifier on superseg_train (via
dts_common, end-turn labels), then online inference (segment.py logic), then score.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from dts_common import load_split, ENCODER_NAME  # noqa: E402  (load_split gives turns+boundaries)
from gate2_heldout import load_ami  # noqa: E402
from hi_ontop.dts_scoring import score_dialogues  # noqa: E402
from hi_ontop.ami_scoring import score_meetings  # noqa: E402
from transformers import AutoTokenizer, RobertaForTokenClassification  # noqa: E402

DEV = "cuda"
IGNORE = -100
CFG = dict(backbone="roberta-base", max_utt_len=25, max_seq_len=512, sliding_window=20,
           lr=1e-5, batch=16, epochs=10, seed=42)


def endturn_labels(turns, boundaries):
    """label[t]=1 if turn t is LAST of a segment = (t+1) starts new seg. last=0."""
    n = len(turns)
    y = [1 if (t + 1) in boundaries else 0 for t in range(n)]
    if y:
        y[-1] = 0
    return y


def encode_window(tok, utts, labels, mode, max_utt_len, max_seq_len):
    toks = [tok.cls_token]; cmask = [0]
    for u in utts:
        ut = tok.tokenize(u)[:max_utt_len - 2]
        toks += ut + ["</s>", "</s>"]; cmask += [0] * len(ut) + [1, 0]
    toks = toks[:-1]; cmask = cmask[:-1]
    ids = tok.convert_tokens_to_ids(toks)[:max_seq_len]; cmask = cmask[:max_seq_len]
    lab = list(labels)
    if mode == "train" and lab:
        lab[-1] = IGNORE
    tlab = [IGNORE] * len(ids); cls_pos = []; i = 0
    for j, m in enumerate(cmask):
        if m:
            if i < len(lab):
                tlab[j] = lab[i]
            cls_pos.append(j); i += 1
    return ids, tlab, cls_pos


class TrainDS(torch.utils.data.Dataset):
    def __init__(self, dialogs, tok):
        self.d, self.tok = dialogs, tok

    def __len__(self):
        return len(self.d)

    def __getitem__(self, idx):
        utts, yt = self.d[idx]; W = CFG["sliding_window"]
        if len(utts) > W:
            s = random.randint(0, len(utts) - W)
            utts, yt = utts[s:s + W - 1], yt[s:s + W - 1]
        ids, lab, _ = encode_window(self.tok, utts, yt, "train", CFG["max_utt_len"], CFG["max_seq_len"])
        return torch.tensor(ids), torch.tensor(lab)


def collate(pad):
    def f(b):
        ids, labs = zip(*b)
        ids = pad_sequence(ids, batch_first=True, padding_value=pad)
        labs = pad_sequence(labs, batch_first=True, padding_value=IGNORE)
        return ids, ids.ne(pad).long(), labs
    return f


def train_roberta(dialogs, tok):
    model = RobertaForTokenClassification.from_pretrained(CFG["backbone"], num_labels=2).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=1e-3)
    dl = torch.utils.data.DataLoader(TrainDS(dialogs, tok), batch_size=CFG["batch"], shuffle=True,
                                     collate_fn=collate(tok.pad_token_id), num_workers=0)
    for ep in range(CFG["epochs"]):
        model.train(); tot = 0.0
        for ids, am, labs in dl:
            ids, am, labs = ids.to(DEV), am.to(DEV), labs.to(DEV)
            out = model(input_ids=ids, attention_mask=am, labels=labs)
            opt.zero_grad(); out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += out.loss.item()
        print(f"  ep{ep} loss {tot/len(dl):.4f}", flush=True)
    return model


@torch.no_grad()
def online_preds(model, tok, list_utts):
    """Strict-causal per-utterance end-turn preds (segment.py logic)."""
    model.eval(); W, pad = CFG["sliding_window"], tok.pad_token_id
    preds = []
    for utts in list_utts:
        n = len(utts); yp = [0] * n; win_ids, meta = [], []
        for t in range(1, n):
            ws = max(0, t - W + 1); wu = utts[ws:t + 1]
            ids, _, cls_pos = encode_window(tok, wu, [0] * len(wu), "test",
                                            CFG["max_utt_len"], CFG["max_seq_len"])
            local = (t - 1) - ws
            win_ids.append(torch.tensor(ids)); meta.append((t, cls_pos[local]))
        for i in range(0, len(win_ids), 256):
            chunk = win_ids[i:i + 256]
            batch = pad_sequence(chunk, batch_first=True, padding_value=pad).to(DEV)
            logits = model(input_ids=batch, attention_mask=batch.ne(pad).long()).logits.cpu().numpy()
            for w, (t, j) in enumerate(meta[i:i + 256]):
                yp[t - 1] = int(logits[w, j].argmax())
        if yp:
            yp[-1] = 0
        preds.append(yp)
    return preds


def main():
    random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
    tok = AutoTokenizer.from_pretrained(CFG["backbone"])

    tr = load_split("superseg_train")
    dialogs = [(d["turns"], endturn_labels(d["turns"], d["boundaries"])) for d in tr]
    print(f"[train] {len(dialogs)} dialogs, superseg_train", flush=True)
    model = train_roberta(dialogs, tok)

    print("\n=== RoBERTa online — IDM과 동일 채점기 ===", flush=True)
    # DTS 3 sets
    for ds in ["tiage_test", "dialseg711_test", "superseg_test"]:
        data = load_split(ds)
        utts = [d["turns"] for d in data]
        golds = [endturn_labels(d["turns"], d["boundaries"]) for d in data]
        preds = online_preds(model, tok, utts)
        r = score_dialogues(golds, preds)
        npred = sum(sum(p) for p in preds); ng = sum(sum(g) for g in golds)
        print(f"{ds:16s} F1={r['f1']:.3f} Pk={r['pk']:.3f} WD={r['wd']:.3f} Score={r['score']:.3f}  "
              f"n_pred={npred} gold={ng}", flush=True)
    # AMI (end-turn pred -> start-turn for ami_scoring)
    from sentence_transformers import SentenceTransformer  # only to satisfy load_ami enc arg
    enc = SentenceTransformer(ENCODER_NAME, device="cuda")
    ami = load_ami("eval", enc)
    utts = [d["turns"] for d in ami]
    preds_end = online_preds(model, tok, utts)
    golds = []; preds_start = []
    for d, pe in zip(ami, preds_end):
        n = len(d["turns"]); bt = list(d["bnd_top"]); bt[-1] = 0; golds.append(bt)
        ps = [0] * n
        for t in range(n):
            if pe[t] == 1 and t + 1 < n:
                ps[t + 1] = 1
        ps[-1] = 0; preds_start.append(ps)
    r = score_meetings(golds, preds_start)
    npred = sum(sum(p) for p in preds_start); ng = sum(sum(g) for g in golds)
    print(f"{'ami_eval':16s} F1={r['f1']:.3f} Pk={r['pk']:.3f} WD={r['wd']:.3f} Score={r['score']:.3f}  "
          f"n_pred={npred} gold={ng}", flush=True)


if __name__ == "__main__":
    main()
