# AI Mühəndisliyi kursu — ev tapşırıqları

Kurs tapşırıqlarının hazır həlləri. Hər dərs öz qovluğundadır.

| Dərs | Mövzu | Qovluq | Vəziyyət |
|---|---|---|---|
| #36 | PyTorch əsasları — tensor, autograd, training loop (`y = 5x`) | [`ders-36-pytorch/`](ders-36-pytorch/) | hazır |

## Dərs #36 — nə var

| Fayl | Nədir |
|---|---|
| `ders36_ev_tapsirigi.ipynb` | Tam notebook. Colab → *File → Upload notebook* |
| `hucreler.md` | Eyni hüceyrələr, mövcud Colab-a yapışdırmaq üçün |
| `numayis.md` | Kursda nümayiş skripti + müəllim sualları və cavabları |
| `build.py` | `.ipynb` və `hucreler.md` faylını eyni mənbədən yazır (`python build.py`) |

Notebook heç bir asılılıq quraşdırmır — `torch` və `matplotlib` Colab-da hazırdır.
İşlətməzdən əvvəl `Runtime → Change runtime type → T4 GPU → Save` (dərsin 2-ci
bölməsi); model və məlumat dərsin 3.6-dakı kimi `.to(device)` ilə GPU-ya köçürülür.

## Dərs #36 — qısa nəticə

Model `y = 5x` düsturunu görmür, yalnız 4 misal alır (1→5, 2→10, 3→15, 4→20).

- Tapşırığın skeleti (200 epoch, `lr = 0.01`) olduğu kimi saxlanılıb →
  `w = 4.62`, `x = 10` üçün **47.3**.
- Tapşırıq 50 vəd edir. Fərqin səbəbi kod deyil: `nn.Linear` `b`-ni də öyrənir
  və 200 addımda `b ≈ 1.12` qalır, `w` isə onu kompensasiya edir.
- Ayrıca hüceyrədə eyni seed və eyni `lr` ilə **yeni** model 2000 epoch
  öyrədilir → `w = 5.00`, `x = 10` üçün **49.99**.

Skelet gizlicə dəyişdirilməyib; fərq izah edilib və sübutu notebook-un içindədir.
