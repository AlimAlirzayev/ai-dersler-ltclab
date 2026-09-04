<h1 align="center">AI Mühəndisliyi kursu — ev tapşırıqları</h1>

<p align="center">
  <a href="https://colab.research.google.com/github/AlimAlirzayev/ai-dersler-ltclab/blob/main/ders-36-pytorch/ders36_ev_tapsirigi.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab-da aç"></a>
  <a href="https://raw.githack.com/AlimAlirzayev/ai-dersler-ltclab/main/ders-36-pytorch/simulyator.html"><img src="https://img.shields.io/badge/Canl%C4%B1%20simulyator-a%C3%A7-2a78d6?style=flat-square" alt="Canlı simulyator"></a>
  <img src="https://img.shields.io/badge/PyTorch-2.x-eb6834?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/qura%C5%9Fd%C4%B1rma-yoxdur-0ca30c?style=flat-square" alt="Quraşdırma yoxdur">
</p>

<p align="center"><b>Dərs #36</b> — modelə <code>y = 5x</code> düsturu deyilmir; o, cəmi <b>4 misaldan</b> özü tapır.</p>

<p align="center">
  <a href="https://raw.githack.com/AlimAlirzayev/ai-dersler-ltclab/main/ders-36-pytorch/simulyator.html">
    <img src="ders-36-pytorch/ekran.png" alt="Simulyator — epoch 200" width="880">
  </a>
</p>

## Dərs #36 · `ders-36-pytorch/`

| | |
|---|---|
| 📓 [`ders36_ev_tapsirigi.ipynb`](ders-36-pytorch/ders36_ev_tapsirigi.ipynb) | Tam həll, 10 hüceyrə · T4 GPU · loss qrafiki · `x = 10` proqnozu |
| 🎛 [`simulyator.html`](ders-36-pytorch/simulyator.html) | Öyrənməni addım-addım oynadır · tək fayl, internetsiz işləyir |
| 📋 [`hucreler.md`](ders-36-pytorch/hucreler.md) | Hüceyrələr, mövcud Colab-a yapışdırmaq üçün |
| 🎤 [`numayis.md`](ders-36-pytorch/numayis.md) | Nümayiş skripti + müəllim sualları və cavabları |

### Nəticə — dürüst

| | 200 epoch *(tapşırığın skeleti)* | 2000 epoch | `b` dondurulmuş |
|---|---|---|---|
| `w` | **4.620** | **4.998** | **5.000** |
| `b` | 1.117 | 0.005 | 0 *(sabit)* |
| `x = 10` | **47.3** | **49.99** | **50.00** |

Tapşırıq 50 vəd edir, skelet isə 47.3 verir. **Səbəb kod deyil:** `nn.Linear` `b`-ni
də öyrənir, 200 addımda `b ≈ 1.12` qalır və `w` onu kompensasiya edir. Simulyatorda
**«b-ni sıfırda dondur»** açarı bunu bir kliklə sübut edir — model ~56 addımda dəqiq
`50.0` verir. Skelet gizlicə dəyişdirilməyib; uzun təlim ayrıca hüceyrədədir.
