# capstone

Pemodelan Keuangan Derivatif adalah mata kuliah yang mempelajari:
1. Teori dasar penetapan harga aset keuangan.
2. Model pergerakan harga saham (misalnya Geometric Brownian Motion).
3. Simulasi numerik dan model stokastik untuk penentuan harga opsi (seperti opsi eksotik atau jenis opsi lainnya).

---

# Pricing Lookback Options: MC vs QMC vs RQMC

Repositori ini mereproduksi paper:

> **Evaluasi Kinerja Metode MC, QMC, dan RQMC pada Penentuan Harga Opsi Lookback**
> Donny Citra Lesmana, Gading Surya Permana, dkk. — IPB University

## 📋 Deskripsi

Perbandingan 5 metode simulasi untuk pricing 4 tipe opsi Lookback:

| Metode | Tipe |
|---|---|
| MC | Monte Carlo standar (pseudorandom) |
| QMC Sobol | Quasi-Monte Carlo barisan Sobol |
| QMC Halton | Quasi-Monte Carlo barisan Halton |
| RQMC Sobol | Randomized QMC Sobol |
| RQMC Halton | Randomized QMC Halton |

Tipe opsi: **Floating Call/Put** dan **Fixed Call/Put**.

## 🔧 Instalasi

```bash
git clone https://github.com/suryagamanaa/capstone.git
cd capstone
pip install -r requirements.txt
