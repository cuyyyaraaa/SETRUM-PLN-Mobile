"""
Pemuat hierarki klasifikasi resmi PLN Mobile (5 layer).
Menyediakan daftar opsi L4/L5 yang dipersempit per PATH PARENT LENGKAP
(L1 -> L2 -> L3 -> L4 -> L5), bukan cuma per leaf label -> dipakai LLM zero-shot.

PENTING (fix ambiguitas): beberapa label L3/L4 di hierarki resmi PLN muncul
di lebih dari satu cabang (mis. L3="Pengaduan" ada di bawah LAYANAN dan di
bawah APLIKASI; L4="Pendaftaran Baru" ada di bawah 4 L3 berbeda). Karena itu
opsi L4/L5 WAJIB di-index dengan path lengkap (l1,l2,l3) dan (l1,l2,l3,l4),
bukan hanya (l3) atau (l4) saja -- kalau tidak, opsi yang ditawarkan ke LLM
zero-shot jadi gabungan lintas cabang yang tidak valid.

Sumber: Hirarki Ulasan dan Indeks Klasifikasi Ulasan PLN Mobile
        (Direktorat Retail dan Niaga, November 2024)
"""
import json, os
from collections import defaultdict

_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'hierarchy_pln_clean.json')


class HierarchyPLN:
    def __init__(self, path=None):
        self.path = path or _PATH
        self._loaded = False

        self.l3_by_l1l2   = defaultdict(set)          # (l1,l2)          -> {l3}
        self.l4_by_path3  = defaultdict(set)          # (l1,l2,l3)       -> {l4}
        self.l5_by_path4  = defaultdict(set)          # (l1,l2,l3,l4)    -> {l5}
        self.param_by_path5 = {}                      # (l1,l2,l3,l4,l5) -> param

        # Index global (leaf-only) DIPERTAHANKAN hanya sebagai fallback
        # darurat, dan dipakai dengan warning eksplisit -- bukan jalur utama.
        self._l4_by_l3_any  = defaultdict(set)
        self._l5_by_l4_any  = defaultdict(set)
        self._param_by_l5_any = {}

        self.all_l1, self.all_l2 = set(), set()
        self.ambiguous_l3 = set()   # l3 label yg muncul di >1 (l1,l2)
        self.ambiguous_l4 = set()   # l4 label yg muncul di >1 (l1,l2,l3)
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            print(f"[Hierarchy] file tidak ditemukan: {self.path}")
            return
        with open(self.path, encoding='utf-8') as f:
            data = json.load(f)

        l3_parents = defaultdict(set)
        l4_parents = defaultdict(set)

        for d in data:
            l1, l2, l3 = d.get('l1'), d.get('l2'), d.get('l3')
            l4, l5, pr = d.get('l4'), d.get('l5'), d.get('param')

            if l1: self.all_l1.add(l1)
            if l2: self.all_l2.add(l2)

            if l1 and l2 and l3:
                self.l3_by_l1l2[(l1, l2)].add(l3)
                l3_parents[l3].add((l1, l2))

            if l1 and l2 and l3 and l4:
                self.l4_by_path3[(l1, l2, l3)].add(l4)
                self._l4_by_l3_any[l3].add(l4)
                l4_parents[l4].add((l1, l2, l3))

            if l1 and l2 and l3 and l4 and l5:
                self.l5_by_path4[(l1, l2, l3, l4)].add(l5)
                self._l5_by_l4_any[l4].add(l5)

            if l1 and l2 and l3 and l4 and l5 and pr:
                self.param_by_path5[(l1, l2, l3, l4, l5)] = pr
                self._param_by_l5_any[l5] = pr

        self.ambiguous_l3 = {k for k, v in l3_parents.items() if len(v) > 1}
        self.ambiguous_l4 = {k for k, v in l4_parents.items() if len(v) > 1}
        self._loaded = True
        print(f"[Hierarchy] {len(self.l3_by_l1l2)} kombinasi L1-L2 dimuat. "
              f"{len(self.ambiguous_l3)} label L3 ambigu, "
              f"{len(self.ambiguous_l4)} label L4 ambigu (di-scope per path).")

    def get_l3_options(self, l1, l2):
        return sorted(self.l3_by_l1l2.get((l1, l2), set()))

    def get_l4_options(self, l1, l2, l3):
        """Opsi L4 yang valid HANYA untuk path (l1,l2,l3) ini."""
        opts = self.l4_by_path3.get((l1, l2, l3))
        if opts:
            return sorted(opts)
        # Fallback darurat: path tidak ditemukan persis (mis. l1/l2 tidak
        # cocok dgn hierarki resmi). Pakai leaf-only index tapi WARN, karena
        # ini bisa menggabung opsi lintas cabang untuk l3 yang ambigu.
        fallback = self._l4_by_l3_any.get(l3, set())
        if fallback:
            tag = " (AMBIGU lintas cabang)" if l3 in self.ambiguous_l3 else ""
            print(f"[Hierarchy] WARN: path ({l1},{l2},{l3}) tidak persis ada di "
                  f"hierarki -> fallback leaf-only untuk L3='{l3}'{tag}")
        return sorted(fallback)

    def get_l5_options(self, l1, l2, l3, l4):
        """Opsi L5 yang valid HANYA untuk path (l1,l2,l3,l4) ini."""
        opts = self.l5_by_path4.get((l1, l2, l3, l4))
        if opts:
            return sorted(opts)
        fallback = self._l5_by_l4_any.get(l4, set())
        if fallback:
            tag = " (AMBIGU lintas cabang)" if l4 in self.ambiguous_l4 else ""
            print(f"[Hierarchy] WARN: path ({l1},{l2},{l3},{l4}) tidak persis ada di "
                  f"hierarki -> fallback leaf-only untuk L4='{l4}'{tag}")
        return sorted(fallback)

    def get_parameter(self, l1, l2, l3, l4, l5):
        pr = self.param_by_path5.get((l1, l2, l3, l4, l5))
        if pr:
            return pr
        return self._param_by_l5_any.get(l5, '')

    def is_loaded(self):
        return self._loaded

    # ── Util untuk rekonsiliasi label (dipakai training/reconcile_labels.py) ──
    def all_l3_under(self, l1, l2):
        return self.get_l3_options(l1, l2)

    def all_l4_leaves(self):
        return sorted(self._l4_by_l3_any.keys())

    def all_l5_leaves(self):
        return sorted(self._l5_by_l4_any.keys())


_inst = None
def get_hierarchy():
    global _inst
    if _inst is None:
        _inst = HierarchyPLN()
    return _inst


if __name__ == '__main__':
    h = get_hierarchy()
    print("Ambiguous L3:", sorted(h.ambiguous_l3))
    print("Ambiguous L4:", sorted(h.ambiguous_l4))
    print(h.get_l4_options('LAYANAN', 'KETENAGALISTRIKAN', 'Pengaduan')[:5])
    print(h.get_l4_options('APLIKASI', 'KETENAGALISTRIKAN', 'Pengaduan')[:5])
