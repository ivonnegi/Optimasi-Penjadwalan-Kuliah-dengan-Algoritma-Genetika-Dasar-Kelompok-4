# Optimasi-Penjadwalan-Kuliah-dengan-Algoritma-Genetika-Dasar-Kelompok-4
# Optimasi Penjadwalan Kuliah dengan Algoritma Genetika (Tahap UTS)

Sistem penjadwalan kuliah otomatis berbasis Algoritma Genetika (AG) dasar. Semua operator
(inisialisasi, seleksi, crossover, mutasi, elitisme) ditulis sendiri tanpa DEAP/PyGAD.

## Data
`data.py` berisi jadwal Semester 5 kelas B (5 mata kuliah, 9 dosen, 7 ruang, 5 hari x 4 blok waktu
150 menit). Beberapa mata kuliah diajar tim 2 dosen, dan keduanya harus bebas di slot yang sama.
Jumlah mahasiswa dan kapasitas ruang belum ada di file jadwal, jadi nilainya placeholder (lihat TODO di `data.py`).
`data_contoh.py` berisi data dummy skala UTS (20 mata kuliah, 27 sesi, 8 dosen, 5 ruang, 5 x 5 slot)
untuk menguji AG pada skala yang lebih besar: `python main.py --data contoh --out output_contoh`.

## Struktur berkas
- `data.py`  : data kelas B (mata kuliah, dosen, ruang, hari, slot, preferensi dosen)
- `data_contoh.py` : data dummy 20 mata kuliah untuk uji skala
- `ga.py`    : model masalah, fungsi fitness, operator genetika, loop evolusi
- `main.py`  : menjalankan eksperimen, mencetak hasil, ekspor jadwal, grafik konvergensi
- `test_ga.py` : unit test fitness dan operator
- `output/`  : hasil data kelas B; `output_contoh/` : hasil data dummy 20 mata kuliah

## Cara menjalankan
    pip install -r requirements.txt
    python main.py                 # 10 run, tournament + uniform (default)
    python main.py --compare       # bandingkan 4 kombinasi seleksi x crossover
    python main.py --selection roulette --crossover onepoint --runs 10 --pop 100 --gen 200 --pc 0.8 --pm 0.03 --elite 2
    python -m unittest -v          # jalankan unit test

## Desain singkat
Kromosom: array integer sepanjang jumlah sesi (satu gen = satu sesi). Nilai gen = `slot * R + ruang` (R = jumlah ruang),
sehingga satu bilangan memuat pasangan (slot waktu, ruang).
Fitness: `1 / (1 + total pelanggaran kendala keras)`, dengan pelanggaran = bentrok dosen,
bentrok ruang, bentrok kelas mahasiswa, dan kapasitas ruang tidak cukup. Fitness 1.0 berarti jadwal valid.
Kendala lunak (preferensi hari dosen dan jeda kelas) sudah dihitung dan dilaporkan di `ga.py`
(`penalti_lunak`) tetapi belum dioptimalkan, sesuai ketentuan UTS.
Operator: seleksi roulette wheel dan turnamen (k=3), crossover satu titik dan uniform,
mutasi acak per gen, elitisme (2 individu terbaik).

## Keluaran di folder output/
`hasil_run_*.csv` (fitness terbaik, rata-rata, generasi valid pertama per seed), `jadwal_terbaik.xlsx`,
`jadwal_terbaik_grid.csv`, `jadwal_terbaik_daftar.csv`, `konvergensi_*.png`, `ringkasan_perbandingan.csv`.
