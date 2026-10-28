"""Algoritma Genetika dasar (ditulis dari nol, tanpa DEAP/PyGAD).

Representasi kromosom (encoding integer):
    kromosom = array panjang N (N = jumlah sesi kuliah)
    gen i    = slot * R + ruang   dengan 0 <= gen < S * R
    slot     = gen // R  (0..S-1, S = jumlah hari x slot per hari)
    ruang    = gen %  R  (0..R-1)
Satu gen langsung memuat pasangan (slot waktu, ruang) untuk satu sesi.
"""
from dataclasses import dataclass, field
import numpy as np

# ----------------------------------------------------------------------
# Model masalah dan fungsi fitness
# ----------------------------------------------------------------------
class Masalah:
    """`data` = modul data (data.py atau data_contoh.py)."""

    def __init__(self, data):
        self.hari = data.HARI
        self.nama_slot = data.SLOT
        self.ruang_def = data.RUANG
        self.preferensi = getattr(data, "PREFERENSI_DOSEN_HINDARI_HARI", {})
        self.sesi = data.bangun_sesi()
        self.N = len(self.sesi)
        self.R = len(self.ruang_def)
        self.n_hari = len(self.hari)
        self.n_slot_hari = len(self.nama_slot)
        self.S = self.n_hari * self.n_slot_hari
        self.n_gen_nilai = self.S * self.R           # ruang nilai tiap gen

        self.daftar_kelas = sorted({s["kelas"] for s in self.sesi})
        self.daftar_dosen = sorted({d for s in self.sesi for d in s["dosen"]})
        self.n_kelas = len(self.daftar_kelas)
        self.n_dosen = len(self.daftar_dosen)
        self.kelas_id = np.array([self.daftar_kelas.index(s["kelas"]) for s in self.sesi])
        self.mhs = np.array([s["mhs"] for s in self.sesi])
        self.kap = np.array([k for _, k in self.ruang_def])

        # satu sesi bisa diajar >1 dosen: simpan pasangan (sesi, dosen)
        ps, pd_ = [], []
        for i, s in enumerate(self.sesi):
            for d in s["dosen"]:
                ps.append(i); pd_.append(self.daftar_dosen.index(d))
        self.pair_sesi = np.array(ps)
        self.pair_dosen = np.array(pd_)

    # --- helper decode ---
    def slot_ruang(self, kromosom):
        return kromosom // self.R, kromosom % self.R

    # --- kendala keras ---
    def rincian_pelanggaran(self, kromosom):
        slot, ruang = self.slot_ruang(kromosom)

        def bentrok(kunci, ukuran):
            c = np.bincount(kunci, minlength=ukuran)
            return int(np.sum(np.maximum(c - 1, 0)))

        v_dosen = bentrok(self.pair_dosen * self.S + slot[self.pair_sesi],
                          self.n_dosen * self.S)
        v_ruang = bentrok(kromosom, self.S * self.R)            # slot+ruang sama
        v_kelas = bentrok(self.kelas_id * self.S + slot, self.n_kelas * self.S)
        v_kapasitas = int(np.sum(self.mhs > self.kap[ruang]))
        return {"dosen": v_dosen, "ruang": v_ruang,
                "kelas": v_kelas, "kapasitas": v_kapasitas}

    def pelanggaran_keras(self, kromosom):
        return sum(self.rincian_pelanggaran(kromosom).values())

    def fitness(self, kromosom):
        """fitness = 1 / (1 + total pelanggaran kendala keras)."""
        return 1.0 / (1.0 + self.pelanggaran_keras(kromosom))

    # --- kendala lunak (didefinisikan & dilaporkan, belum dioptimalkan) ---
    def penalti_lunak(self, kromosom):
        slot, _ = self.slot_ruang(kromosom)
        hari = slot // self.n_slot_hari
        jam = slot % self.n_slot_hari

        # (a) dosen mengajar pada hari yang dihindari
        pref = 0
        for i, s in enumerate(self.sesi):
            for d in s["dosen"]:
                if hari[i] in self.preferensi.get(d, set()):
                    pref += 1

        # (b) jeda (slot kosong) di antara kuliah pada hari yang sama untuk tiap kelas
        jeda = 0
        for k in range(self.n_kelas):
            for h in range(self.n_hari):
                j = np.sort(jam[(self.kelas_id == k) & (hari == h)])
                if len(j) > 1:
                    jeda += int(j[-1] - j[0] + 1 - len(j))
        return {"preferensi_dosen": pref, "jeda_mahasiswa": jeda}


# ----------------------------------------------------------------------
# Operator genetika
# ----------------------------------------------------------------------
def inisialisasi(rng, ukuran_pop, masalah):
    return rng.integers(0, masalah.n_gen_nilai, size=(ukuran_pop, masalah.N))


def seleksi_roulette(rng, pop, fit, n):
    p = fit / fit.sum()
    idx = rng.choice(len(pop), size=n, p=p)
    return pop[idx]


def seleksi_turnamen(rng, pop, fit, n, k=3):
    kandidat = rng.integers(0, len(pop), size=(n, k))
    menang = kandidat[np.arange(n), np.argmax(fit[kandidat], axis=1)]
    return pop[menang]


def crossover_satu_titik(rng, a, b):
    t = rng.integers(1, len(a))
    return (np.concatenate([a[:t], b[t:]]),
            np.concatenate([b[:t], a[t:]]))


def crossover_uniform(rng, a, b):
    mask = rng.random(len(a)) < 0.5
    return np.where(mask, a, b), np.where(mask, b, a)


def mutasi_acak(rng, k, pm, n_nilai):
    k = k.copy()
    m = rng.random(len(k)) < pm
    k[m] = rng.integers(0, n_nilai, size=int(m.sum()))
    return k


SELEKSI = {"roulette": seleksi_roulette, "tournament": seleksi_turnamen}
CROSSOVER = {"onepoint": crossover_satu_titik, "uniform": crossover_uniform}


# ----------------------------------------------------------------------
# Loop evolusi
# ----------------------------------------------------------------------
@dataclass
class Parameter:
    ukuran_pop: int = 100
    generasi: int = 200
    pc: float = 0.8            # probabilitas crossover
    pm: float = 0.03           # probabilitas mutasi per gen
    elitisme: int = 2
    seleksi: str = "tournament"
    crossover: str = "uniform"
    ukuran_turnamen: int = 3


@dataclass
class HasilRun:
    seed: int
    terbaik: np.ndarray
    fitness_terbaik: float
    fitness_rata_akhir: float
    generasi_valid_pertama: int           # -1 jika tidak pernah valid
    riwayat_terbaik: list = field(default_factory=list)
    riwayat_rata: list = field(default_factory=list)


def jalankan_ga(masalah, par, seed):
    rng = np.random.default_rng(seed)
    pop = inisialisasi(rng, par.ukuran_pop, masalah)
    pilih = SELEKSI[par.seleksi]
    silang = CROSSOVER[par.crossover]

    riwayat_terbaik, riwayat_rata = [], []
    valid_pertama = -1
    terbaik_global, fit_terbaik_global = None, -1.0

    for gen in range(par.generasi + 1):
        fit = np.array([masalah.fitness(k) for k in pop])
        i_best = int(np.argmax(fit))
        if fit[i_best] > fit_terbaik_global:
            fit_terbaik_global, terbaik_global = fit[i_best], pop[i_best].copy()
        riwayat_terbaik.append(float(fit[i_best]))
        riwayat_rata.append(float(fit.mean()))
        if valid_pertama < 0 and fit[i_best] == 1.0:
            valid_pertama = gen
        if gen == par.generasi:
            break

        # elitisme: salin individu terbaik apa adanya
        urut = np.argsort(-fit)
        elit = pop[urut[:par.elitisme]].copy()

        # seleksi orang tua
        n_anak = par.ukuran_pop - par.elitisme
        n_ortu = n_anak + (n_anak % 2)
        if par.seleksi == "tournament":
            ortu = pilih(rng, pop, fit, n_ortu, par.ukuran_turnamen)
        else:
            ortu = pilih(rng, pop, fit, n_ortu)

        # crossover + mutasi
        anak = []
        for i in range(0, n_ortu, 2):
            a, b = ortu[i], ortu[i + 1]
            if rng.random() < par.pc:
                c1, c2 = silang(rng, a, b)
            else:
                c1, c2 = a.copy(), b.copy()
            anak.append(mutasi_acak(rng, c1, par.pm, masalah.n_gen_nilai))
            anak.append(mutasi_acak(rng, c2, par.pm, masalah.n_gen_nilai))
        pop = np.vstack([elit, np.array(anak[:n_anak])])

    return HasilRun(seed, terbaik_global, float(fit_terbaik_global),
                    riwayat_rata[-1], valid_pertama,
                    riwayat_terbaik, riwayat_rata)
