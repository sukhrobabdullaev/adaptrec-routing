# -*- coding: utf-8 -*-
"""MiniLM cross-encoder'ning numpy'dagi oldinga yurishi.

Nega torch emas: PyPI'dagi torch g'ildiragi CUDA bog'liqliklari bilan
bir necha gigabayt tortadi, bizga esa faqat CPU'dagi inferens kerak.
6 qatlamli BERT'ni numpy'da yozish ~150 qator va u narxni ham toza
o'lchash imkonini beradi (freymvork ustamasi yo'q).

Model: cross-encoder/ms-marco-MiniLM-L6-v2
  qatlam 6, yashirin 384, bosh 12, oraliq 1536, lug'at 30522
"""
import json, os, numpy as np


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def layer_norm(x, w, b, eps=1e-12):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


class MiniLMCrossEncoder:
    """Faqat inferens. Kirish: (B, L) token id, chiqish: (B,) relevantlik bahosi."""

    def __init__(self, katalog):
        from safetensors.numpy import load_file
        self.cfg = json.load(open(os.path.join(katalog, "config.json")))
        W = load_file(os.path.join(katalog, "model.safetensors"))
        self.W = {k: v.astype(np.float32) for k, v in W.items()}
        self.L = self.cfg["num_hidden_layers"]
        self.H = self.cfg["num_attention_heads"]
        self.d = self.cfg["hidden_size"]
        self.dh = self.d // self.H
        self._p = self._tayyorla()

    def _g(self, *nomlar):
        for n in nomlar:
            if n in self.W: return self.W[n]
        raise KeyError(nomlar)

    def _tayyorla(self):
        """Og'irliklarni bir marta transpozitsiya qilib keshlash."""
        g = self._g
        p = {
            "we": g("bert.embeddings.word_embeddings.weight"),
            "pe": g("bert.embeddings.position_embeddings.weight"),
            "te": g("bert.embeddings.token_type_embeddings.weight"),
            "eln_w": g("bert.embeddings.LayerNorm.weight", "bert.embeddings.LayerNorm.gamma"),
            "eln_b": g("bert.embeddings.LayerNorm.bias", "bert.embeddings.LayerNorm.beta"),
            "layers": [],
        }
        for i in range(self.L):
            b = "bert.encoder.layer.%d." % i
            p["layers"].append({
                "q_w": g(b + "attention.self.query.weight").T, "q_b": g(b + "attention.self.query.bias"),
                "k_w": g(b + "attention.self.key.weight").T,   "k_b": g(b + "attention.self.key.bias"),
                "v_w": g(b + "attention.self.value.weight").T, "v_b": g(b + "attention.self.value.bias"),
                "o_w": g(b + "attention.output.dense.weight").T, "o_b": g(b + "attention.output.dense.bias"),
                "ln1_w": g(b + "attention.output.LayerNorm.weight", b + "attention.output.LayerNorm.gamma"),
                "ln1_b": g(b + "attention.output.LayerNorm.bias", b + "attention.output.LayerNorm.beta"),
                "i_w": g(b + "intermediate.dense.weight").T, "i_b": g(b + "intermediate.dense.bias"),
                "d_w": g(b + "output.dense.weight").T, "d_b": g(b + "output.dense.bias"),
                "ln2_w": g(b + "output.LayerNorm.weight", b + "output.LayerNorm.gamma"),
                "ln2_b": g(b + "output.LayerNorm.bias", b + "output.LayerNorm.beta"),
            })
        p["pool_w"] = g("bert.pooler.dense.weight").T
        p["pool_b"] = g("bert.pooler.dense.bias")
        p["cls_w"] = g("classifier.weight").T
        p["cls_b"] = g("classifier.bias")
        return p

    def __call__(self, ids, type_ids=None, mask=None):
        """ids: (B, L) int32. Qaytadi: (B,) float32 baho."""
        p = self._p
        B, L = ids.shape
        if type_ids is None: type_ids = np.zeros((B, L), np.int32)
        if mask is None: mask = (ids != 0).astype(np.float32)

        x = p["we"][ids] + p["pe"][:L][None] + p["te"][type_ids]
        x = layer_norm(x, p["eln_w"], p["eln_b"])
        # (B, 1, 1, L) additiv niqob
        add = (1.0 - mask)[:, None, None, :] * -10000.0

        for lay in p["layers"]:
            q = (x @ lay["q_w"] + lay["q_b"]).reshape(B, L, self.H, self.dh).transpose(0, 2, 1, 3)
            k = (x @ lay["k_w"] + lay["k_b"]).reshape(B, L, self.H, self.dh).transpose(0, 2, 1, 3)
            v = (x @ lay["v_w"] + lay["v_b"]).reshape(B, L, self.H, self.dh).transpose(0, 2, 1, 3)
            s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(self.dh) + add
            s -= s.max(-1, keepdims=True)
            np.exp(s, out=s)
            s /= s.sum(-1, keepdims=True)
            c = (s @ v).transpose(0, 2, 1, 3).reshape(B, L, self.d)
            x = layer_norm(x + (c @ lay["o_w"] + lay["o_b"]), lay["ln1_w"], lay["ln1_b"])
            h = gelu(x @ lay["i_w"] + lay["i_b"])
            x = layer_norm(x + (h @ lay["d_w"] + lay["d_b"]), lay["ln2_w"], lay["ln2_b"])

        pooled = np.tanh(x[:, 0] @ p["pool_w"] + p["pool_b"])
        return (pooled @ p["cls_w"] + p["cls_b"]).ravel()


class WordPiece:
    """BERT uncased WordPiece tokenizatori (vocab.txt dan)."""

    def __init__(self, vocab_yoli, max_input_chars_per_word=100):
        self.v = {}
        with open(vocab_yoli, encoding="utf-8") as f:
            for i, line in enumerate(f):
                self.v[line.rstrip("\n")] = i
        self.unk = self.v["[UNK]"]; self.cls = self.v["[CLS]"]
        self.sep = self.v["[SEP]"]; self.pad = self.v["[PAD]"]
        self.mx = max_input_chars_per_word

    @staticmethod
    def _bosqich1(text):
        """Kichik harf, aksentlarni tashlash, punktuatsiyani ajratish."""
        import unicodedata
        text = unicodedata.normalize("NFD", text.lower())
        out = []
        for ch in text:
            if unicodedata.category(ch) == "Mn": continue
            if ch.isspace(): out.append(" ")
            elif ch.isalnum(): out.append(ch)
            else: out.append(" " + ch + " ")
        return "".join(out).split()

    def _wordpiece(self, so_z):
        if len(so_z) > self.mx: return [self.unk]
        out, start = [], 0
        while start < len(so_z):
            end = len(so_z); cur = None
            while start < end:
                sub = so_z[start:end]
                if start > 0: sub = "##" + sub
                if sub in self.v: cur = self.v[sub]; break
                end -= 1
            if cur is None: return [self.unk]
            out.append(cur); start = end
        return out

    def kodla(self, a, b, max_len=64):
        """(a, b) juftligini [CLS] a [SEP] b [SEP] ko'rinishida kodlaydi."""
        ta = [t for w in self._bosqich1(a) for t in self._wordpiece(w)]
        tb = [t for w in self._bosqich1(b) for t in self._wordpiece(w)]
        bo_sh = max_len - 3
        while len(ta) + len(tb) > bo_sh:
            if len(ta) > len(tb): ta.pop()
            else: tb.pop()
        ids = [self.cls] + ta + [self.sep] + tb + [self.sep]
        tt = [0] * (len(ta) + 2) + [1] * (len(tb) + 1)
        return ids, tt

    def paket(self, juftliklar, max_len=64):
        """Juftliklar ro'yxatidan (ids, type_ids, mask) paketini yasaydi."""
        kod = [self.kodla(a, b, max_len) for a, b in juftliklar]
        L = max(len(i) for i, _ in kod)
        B = len(kod)
        ids = np.zeros((B, L), np.int32)
        tt = np.zeros((B, L), np.int32)
        mk = np.zeros((B, L), np.float32)
        for r, (i, t) in enumerate(kod):
            ids[r, :len(i)] = i; tt[r, :len(t)] = t; mk[r, :len(i)] = 1.0
        return ids, tt, mk
