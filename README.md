# capstone
# Pemodelan Keuangan Derivatif
Pemodelan Keuangan Derivatif adalah mata kuliah yang mempelajari:
1. Teori dasar penetapan harga aset keuangan.
2. Model pergerakan harga saham (misalnya Geometric Brownian Motion).
3. Simulasi numerik dan model stokastik untuk penentuan harga opsi (seperti opsi eksotik atau jenis opsi lainnya).

---

# Pricing Lookback Options: MC vs QMC vs RQMC

Repositori ini mereproduksi paper:

> **Evaluasi Kinerja Metode MC, QMC, dan RQMC pada Penentuan Harga Opsi Lookback**
> Donny Citra Lesmana, Gading Surya Permana, dkk. — IPB University

## 📄 Abstrak

Opsi merupakan salah satu produk derivatif yang bermanfaat untuk mengelola risiko, melakukan spekulasi, serta mengurangi biaya transaksi dari investasi. Salah satu jenis opsi adalah opsi *lookback*. Opsi *lookback* adalah opsi yang *payoff*-nya bergantung pada harga aset tertinggi atau terendah yang dicapai selama masa berlaku opsi. Meskipun solusi analitik untuk opsi *lookback* tersedia dalam kasus *continuous monitoring*, harga aset ketika diamati pada titik waktu tertentu tidak memiliki formula tertutup (*closed-form*) sehingga memerlukan pendekatan numerik. Oleh karena itu, penelitian ini bertujuan untuk membandingkan lima metode simulasi, yakni simulasi Monte Carlo standar, Quasi-Monte Carlo dengan barisan Sobol dan Halton, serta Randomized Quasi-Monte Carlo dengan barisan Sobol dan Halton dengan nilai analitik sebagai tolak ukur, sehingga didapatkan metode yang lebih efektif dalam menilai harga opsi *lookback*. Hasil penelitian menunjukkan bahwa metode **RQMC Sobol** memberikan hasil yang lebih akurat, galat relatif yang lebih kecil, serta laju konvergensi yang lebih cepat dibandingkan dengan keempat metode lainnya. Dengan demikian, metode ini terbukti lebih efektif dalam menilai harga opsi *lookback*.

**Kata Kunci:** Opsi Lookback, Monte Carlo, Quasi-Monte Carlo, Randomized Quasi-Monte Carlo

---

## 📋 Model Overview

Perbandingan 5 metode simulasi untuk pricing 4 tipe opsi Lookback:

| Metode | Tipe |
|---|---|
| MC | Monte Carlo standar (pseudorandom) |
| QMC Sobol | Quasi-Monte Carlo barisan Sobol |
| QMC Halton | Quasi-Monte Carlo barisan Halton |
| RQMC Sobol | Randomized QMC Sobol |
| RQMC Halton | Randomized QMC Halton |

Tipe opsi: **Floating Call/Put** dan **Fixed Call/Put**.

1. **Model Harga Aset**: Geometric Brownian Motion (GBM)
   $$S_t = S_{t-1} \exp\left[\left(r - \tfrac{\sigma^2}{2}\right)\Delta t + \sigma \varepsilon \sqrt{\Delta t}\right]$$

2. **Nilai Analitik**: Formula closed-form continuous monitoring
   - Goldman-Sosin-Gatto (1979) → floating strike
   - Conze-Viswanathan (1991) → fixed strike

3. **Simulasi Numerik**: 5 metode MC/QMC/RQMC
   - N ∈ {10k, 20k, 40k, 80k, 160k, 320k, 640k}
   - M = 30 replikasi untuk standard error
   - n_steps = 252 (daily monitoring)

4. **Metrik Evaluasi**:
   - Galat Relatif: $RE = \left|\frac{\hat V - V}{V}\right| \times 100\%$
   - Standard Error: $SE = \sqrt{\frac{\sum(X_i - \bar X)^2}{n(n-1)}}$

## 🔧 Instalasi

```bash
git clone https://github.com/suryagamanaa/capstone.git
cd capstone
pip install -r requirements.txt
