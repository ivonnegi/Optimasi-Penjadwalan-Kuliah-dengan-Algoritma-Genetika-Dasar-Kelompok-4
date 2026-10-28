"""Menjalankan eksperimen UTS: AG dasar untuk penjadwalan kuliah.

Contoh:
    python main.py                       # 10 run, tournament + uniform
    python main.py --compare             # bandingkan 4 kombinasi seleksi x crossover
    python main.py --selection roulette --crossover onepoint --runs 10
"""
import argparse
import importlib
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ga import Masalah, Parameter, jalankan_ga


def tabel_jadwal(masalah, kromosom):
    """Tabel jadwal (baris = slot waktu, kolom = hari)."""
    slot, ruang = masalah.slot_ruang(kromosom)
    sel = {(h, j): [] for h in range(masalah.n_hari) for j in range(masalah.n_slot_hari)}
    for i, s in enumerate(masalah.sesi):
        h, j = slot[i] // masalah.n_slot_hari, slot[i] % masalah.n_slot_hari
        dosen = ", ".join(s["dosen"])
        sel[(h, j)].append(f'{s["kode"]} {s["nama"]} [{s["kelas"]}; {dosen}; {masalah.ruang_def[ruang[i]][0]}]')
    data = {masalah.hari[h]: [" | ".join(sel[(h, j)]) for j in range(masalah.n_slot_hari)]
            for h in range(masalah.n_hari)}
    return pd.DataFrame(data, index=masalah.nama_slot)


def daftar_sesi(masalah, kromosom):
    slot, ruang = masalah.slot_ruang(kromosom)
    baris = []
    for i, s in enumerate(masalah.sesi):
        h, j = int(slot[i] // masalah.n_slot_hari), int(slot[i] % masalah.n_slot_hari)
        baris.append({
            "Kode": s["kode"], "Mata Kuliah": s["nama"], "Sesi": s["sesi_ke"],
            "Kelas": s["kelas"], "Dosen": "; ".join(s["dosen"]),
            "Hari": masalah.hari[h], "Jam": masalah.nama_slot[j],
            "Ruang": masalah.ruang_def[ruang[i]][0], "_h": h, "_j": j,
        })
    df = pd.DataFrame(baris).sort_values(["_h", "_j", "Ruang"]).drop(columns=["_h", "_j"])
    return df.reset_index(drop=True)


def jalankan_konfigurasi(masalah, par, n_run, seed0):
    hasil = [jalankan_ga(masalah, par, seed0 + r) for r in range(n_run)]
    return hasil


def ringkas(hasil):
    baris = []
    for h in hasil:
        baris.append({
            "Seed": h.seed,
            "Fitness terbaik": round(h.fitness_terbaik, 4),
            "Fitness rata-rata (populasi akhir)": round(h.fitness_rata_akhir, 4),
            "Generasi valid pertama": h.generasi_valid_pertama if h.generasi_valid_pertama >= 0 else "-",
            "Pelanggaran keras": int(round(1 / h.fitness_terbaik - 1)),
        })
    return pd.DataFrame(baris)


def plot_konvergensi(hasil, judul, path):
    terbaik = np.array([h.riwayat_terbaik for h in hasil])
    rata = np.array([h.riwayat_rata for h in hasil])
    g = np.arange(terbaik.shape[1])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r in terbaik:
        ax.plot(g, r, color="tab:blue", alpha=0.15, lw=0.8)
    ax.plot(g, terbaik.mean(0), color="tab:blue", lw=2, label="Fitness terbaik (rata-rata semua run)")
    ax.plot(g, rata.mean(0), color="tab:orange", lw=2, label="Fitness rata-rata populasi (rata-rata semua run)")
    ax.set_xlabel("Generasi"); ax.set_ylabel("Fitness"); ax.set_ylim(0, 1.05)
    ax.set_title(judul); ax.grid(alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--gen", type=int, default=200)
    ap.add_argument("--pc", type=float, default=0.8)
    ap.add_argument("--pm", type=float, default=0.03)
    ap.add_argument("--elite", type=int, default=2)
    ap.add_argument("--selection", choices=["roulette", "tournament"], default="tournament")
    ap.add_argument("--crossover", choices=["onepoint", "uniform"], default="uniform")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--out", default="output")
    ap.add_argument("--data", choices=["prodi", "contoh"], default="prodi",
                    help="prodi = data kelas B (data.py); contoh = data dummy 20 matkul (data_contoh.py)")
    ap.add_argument("--compare", action="store_true",
                    help="bandingkan semua kombinasi seleksi x crossover")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    modul = importlib.import_module("data" if a.data == "prodi" else "data_contoh")
    masalah = Masalah(modul)
    print(f"Data: {modul.__name__}.py")
    print(f"Masalah: {len(set(s['kode'] for s in masalah.sesi))} mata kuliah, "
          f"{masalah.N} sesi, {masalah.n_dosen} dosen, "
          f"{masalah.R} ruang, {masalah.n_hari} hari x {masalah.n_slot_hari} slot")

    def par_dari(sel, cr):
        return Parameter(a.pop, a.gen, a.pc, a.pm, a.elite, sel, cr)

    if a.compare:
        kombinasi = [(s, c) for s in ["roulette", "tournament"] for c in ["onepoint", "uniform"]]
    else:
        kombinasi = [(a.selection, a.crossover)]

    ringkasan = []
    hasil_utama, par_utama = None, None
    for sel, cr in kombinasi:
        par = par_dari(sel, cr)
        hasil = jalankan_konfigurasi(masalah, par, a.runs, a.seed0)
        df = ringkas(hasil)
        valid = [h.generasi_valid_pertama for h in hasil if h.generasi_valid_pertama >= 0]
        ringkasan.append({
            "Seleksi": sel, "Crossover": cr,
            "Run valid": f"{len(valid)}/{len(hasil)}",
            "Fitness terbaik (rata-rata)": round(float(np.mean([h.fitness_terbaik for h in hasil])), 4),
            "Fitness rata-rata akhir": round(float(np.mean([h.fitness_rata_akhir for h in hasil])), 4),
            "Gen. valid pertama (rata-rata)": round(float(np.mean(valid)), 1) if valid else "-",
        })
        nama = f"{sel}_{cr}"
        df.to_csv(os.path.join(a.out, f"hasil_run_{nama}.csv"), index=False)
        plot_konvergensi(hasil, f"Konvergensi AG ({sel} + {cr}, {a.runs} run)",
                         os.path.join(a.out, f"konvergensi_{nama}.png"))
        print(f"\n=== {sel} + {cr} ===")
        print(df.to_string(index=False))
        if hasil_utama is None or (sel, cr) == (a.selection, a.crossover):
            hasil_utama, par_utama = hasil, par

    if a.compare:
        print("\n=== Ringkasan perbandingan ===")
        rdf = pd.DataFrame(ringkasan)
        print(rdf.to_string(index=False))
        rdf.to_csv(os.path.join(a.out, "ringkasan_perbandingan.csv"), index=False)

    # jadwal terbaik dari konfigurasi utama
    terbaik = max(hasil_utama, key=lambda h: (h.fitness_terbaik, -h.generasi_valid_pertama))
    grid = tabel_jadwal(masalah, terbaik.terbaik)
    sesi = daftar_sesi(masalah, terbaik.terbaik)
    grid.to_csv(os.path.join(a.out, "jadwal_terbaik_grid.csv"), encoding="utf-8-sig")
    sesi.to_csv(os.path.join(a.out, "jadwal_terbaik_daftar.csv"), index=False, encoding="utf-8-sig")
    try:
        with pd.ExcelWriter(os.path.join(a.out, "jadwal_terbaik.xlsx")) as w:
            grid.to_excel(w, sheet_name="Grid Hari x Slot")
            sesi.to_excel(w, sheet_name="Daftar Sesi", index=False)
    except ImportError:
        print("(openpyxl tidak terpasang, ekspor Excel dilewati)")

    rinci = masalah.rincian_pelanggaran(terbaik.terbaik)
    lunak = masalah.penalti_lunak(terbaik.terbaik)
    print(f"\nJadwal terbaik (seed {terbaik.seed}): fitness = {terbaik.fitness_terbaik:.4f}")
    print("Pelanggaran kendala keras:", rinci)
    print("Penalti kendala lunak (belum dioptimalkan):", lunak)
    print("\nJadwal terbaik (urut hari dan jam):")
    pd.set_option("display.max_colwidth", 60, "display.width", 250)
    print(sesi.to_string(index=False))
    print(f"\nFile hasil tersimpan di folder '{a.out}/'")


if __name__ == "__main__":
    main()
