# -*- coding: utf-8 -*-
"""Real tavsiya komponentlari, TO'LIQ reyting rejimida.

Har bir komponent butun katalog bo'yicha (n_item,) ball vektorini
qaytaradi. Namuna olingan negativlar yo'q: bu maqolaning o'zi va'da
qilgan "full unsampled ranking" protokoli va u siyrak tuzilmalar
tufayli nomzodli yondashuvdan tezroq ham chiqadi.

Interfeys:
    yangilash(u, i, t)  -> onlayn holat yangilash
    baholash(u, hist)   -> (n_item,) float32
"""
import numpy as np, collections, math


class Komponent:
    nom = "?"
    def yangilash(self, u, i, t): pass
    def baholash(self, u, hist): raise NotImplementedError


class Ommaboplik(Komponent):
    """Yaqin oynadagi ommaboplik (asos chizig'i va zaxira signal)."""
    nom = "m0_pop"

    def __init__(self, n_item, oyna=50000):
        self.cnt = np.zeros(n_item, np.float32)
        self.bufer = collections.deque(maxlen=oyna)

    def yangilash(self, u, i, t):
        if len(self.bufer) == self.bufer.maxlen:
            self.cnt[self.bufer[0]] -= 1.0
        self.bufer.append(i); self.cnt[i] += 1.0

    def baholash(self, u, hist):
        return self.cnt


class ItemKNN(Komponent):
    """m1: birgalikda uchrash kosinusi. Qatorlar siyrak, keshlanadi."""
    nom = "m1_itemknn"

    def __init__(self, n_item, max_hist=10):
        self.n_item = n_item
        self.co = collections.defaultdict(collections.Counter)
        self.cnt = np.zeros(n_item, np.float32)
        self.max_hist = max_hist
        self._kesh = {}

    def yangilash(self, u, i, t, oldingi=()):
        self.cnt[i] += 1.0
        self._kesh.pop(i, None)
        for h in oldingi[-self.max_hist:]:
            if h == i: continue
            self.co[h][i] += 1; self.co[i][h] += 1
            self._kesh.pop(h, None)

    def _qator(self, h):
        r = self._kesh.get(h)
        if r is None:
            row = self.co.get(h)
            if not row: return None
            idx = np.fromiter(row.keys(), np.int32, len(row))
            val = np.fromiter(row.values(), np.float32, len(row))
            r = (idx, val); self._kesh[h] = r
        return r

    def baholash(self, u, hist):
        s = np.zeros(self.n_item, np.float32)
        if not hist: return s
        inv = 1.0 / np.sqrt(self.cnt + 1.0)
        for h in hist[-self.max_hist:]:
            r = self._qator(h)
            if r is None: continue
            idx, val = r
            np.add.at(s, idx, val * inv[idx] / math.sqrt(self.cnt[h] + 1.0))
        return s


class Kontent(Komponent):
    """m2: item matni ustida TF-IDF + LSA, profil bilan kosinus."""
    nom = "m2_kontent"

    def __init__(self, matnlar, dim=128, seed=0):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize
        tf = TfidfVectorizer(max_features=60000, ngram_range=(1, 2),
                             min_df=2, sublinear_tf=True, strip_accents="unicode")
        X = tf.fit_transform(matnlar)
        svd = TruncatedSVD(n_components=dim, random_state=seed)
        self.E = normalize(svd.fit_transform(X)).astype(np.float32)
        self.n_item = self.E.shape[0]

    def baholash(self, u, hist):
        if not hist: return np.zeros(self.n_item, np.float32)
        p = self.E[hist[-20:]].mean(0)
        n = np.linalg.norm(p)
        if n == 0: return np.zeros(self.n_item, np.float32)
        return self.E @ (p / n)


class P3Alpha(Komponent):
    """m3: item -> foydalanuvchi -> item uch qadamli sayr.

    Qimmat komponent: tarixdagi har bir item uchun uni ko'rgan
    foydalanuvchilar bo'ylab yuriladi. Ommabop itemlarda bu ro'yxat
    uzun, shuning uchun `cap` bilan tekis namuna olinadi (real
    tizimlarda ham shunday qilinadi).
    """
    nom = "m3_graf"

    def __init__(self, n_item, alpha=0.6, cap=120, max_hist=6):
        self.n_item = n_item
        self.item_users = collections.defaultdict(list)
        self.user_items = collections.defaultdict(list)
        self.alpha, self.cap, self.max_hist = alpha, cap, max_hist

    def yangilash(self, u, i, t):
        iu = self.item_users[i]
        if len(iu) < 4000: iu.append(u)
        self.user_items[u].append(i)

    def baholash(self, u, hist):
        s = np.zeros(self.n_item, np.float32)
        if not hist: return s
        for h in hist[-self.max_hist:]:
            users = self.item_users.get(h)
            if not users: continue
            if len(users) > self.cap:
                users = users[::max(1, len(users) // self.cap)][:self.cap]
            w_h = 1.0 / (len(users) ** self.alpha)
            for v in users:
                its = self.user_items.get(v)
                if not its: continue
                np.add.at(s, its, w_h / (len(its) ** self.alpha))
        return s


class Ketmaket(Komponent):
    """m4: oxirgi itemlardan o'tish ehtimoli (Markov, susayuvchi og'irlik)."""
    nom = "m4_ketmaket"

    def __init__(self, n_item):
        self.n_item = n_item
        self.tr = collections.defaultdict(collections.Counter)
        self.out = collections.Counter()
        self._kesh = {}

    def yangilash(self, u, i, t, oxirgi=None):
        if oxirgi is not None and oxirgi != i:
            self.tr[oxirgi][i] += 1; self.out[oxirgi] += 1
            self._kesh.pop(oxirgi, None)

    def baholash(self, u, hist):
        s = np.zeros(self.n_item, np.float32)
        if not hist: return s
        for d, h in enumerate(reversed(hist[-3:])):
            r = self._kesh.get(h)
            if r is None:
                row = self.tr.get(h)
                if not row: continue
                idx = np.fromiter(row.keys(), np.int32, len(row))
                val = np.fromiter(row.values(), np.float32, len(row))
                r = (idx, val / max(1, self.out[h])); self._kesh[h] = r
            idx, val = r
            np.add.at(s, idx, (0.6 ** d) * val)
        return s


class CrossEncoderRerank(Komponent):
    """m5: MiniLM cross-encoder bilan qayta tartiblash (qimmat komponent).

    To'liq katalogni transformer bilan baholash mumkin emas, shuning uchun
    bu komponent asosiy ballning top-K sini oladi va faqat o'shalarni
    qayta tartiblaydi. Bu maqolaning 8 bosqichli pipeline'idagi
    re-ranking bosqichiga aynan mos keladi va aynan shu sababdan
    komponentning narxi so'rov boshiga o'lchanadi.
    """
    nom = "m5_crossenc"

    def __init__(self, n_item, sarlavhalar, model_katalogi, top_k=20,
                 max_len=32, tarix_uzunligi=3, backend="onnx", threads=2):
        """backend: "onnx" (rasmiy int8 eksporti, production'ga yaqin) yoki
        "numpy" (mos yozuvlar implementatsiyasi, ~20x sekinroq)."""
        from .minilm import WordPiece
        self.n_item = n_item
        self.titles = sarlavhalar
        self.tok = WordPiece(model_katalogi + "/vocab.txt")
        self.top_k, self.max_len, self.hl = top_k, max_len, tarix_uzunligi
        self.backend = backend
        if backend == "onnx":
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.sess = ort.InferenceSession(model_katalogi + "/model_qint8.onnx",
                                             so, providers=["CPUExecutionProvider"])
        else:
            from .minilm import MiniLMCrossEncoder
            self.m = MiniLMCrossEncoder(model_katalogi)

    def _ball(self, ids, tt, mk):
        if self.backend == "onnx":
            return self.sess.run(None, {"input_ids": ids.astype(np.int64),
                                        "attention_mask": mk.astype(np.int64),
                                        "token_type_ids": tt.astype(np.int64)})[0].ravel()
        return self.m(ids, tt, mk)

    def baholash(self, u, hist, asos=None):
        s = np.zeros(self.n_item, np.float32)
        if not hist or asos is None: return s
        k = min(self.top_k, self.n_item)
        top = np.argpartition(-asos, k - 1)[:k]
        so_rov = " ".join(self.titles[h] for h in hist[-self.hl:])[:220]
        juft = [(so_rov, self.titles[int(c)][:120]) for c in top]
        ids, tt, mk = self.tok.paket(juft, max_len=self.max_len)
        sc = self._ball(ids, tt, mk)
        s[top] = sc
        return s


class Fuzya(Komponent):
    """Arzon komponentlarning normallashtirilgan yig'indisi.

    m5 aynan shu ballning top-K sini qayta tartiblaydi, shuning uchun
    m5 ni baholashda taqqoslash asosi aynan shu bo'lishi kerak.
    """
    nom = "f_asos"

    def __init__(self, n_item, manbalar):
        self.n_item = n_item
        self.manbalar = manbalar          # Komponent obyektlari ro'yxati

    def baholash(self, u, hist, oldindan=None):
        s = np.zeros(self.n_item, np.float32)
        for c in self.manbalar:
            x = oldindan.get(c.nom) if oldindan else None
            if x is None: x = c.baholash(u, hist)
            m = x.max()
            if m > 0: s += x / m
        return s
