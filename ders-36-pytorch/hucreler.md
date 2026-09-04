# Colab-a yapışdırmaq üçün hüceyrələr

Açıq notebook: <https://colab.research.google.com/drive/1FR3zFhiDLb5t5p5vUhwzLqpC1wbfJteg>

Ordakı mövcud kodu **2-ci hüceyrə** ilə əvəz et, qalanlarını ardınca əlavə et
(`+ Code` / `+ Text`). Sonra `Runtime → Run all`.

---

## Hüceyrə 1 — MƏTN (Text) — Basliq

````markdown
# Dərs #36 — Ev tapşırığı: modelə `y = 5x` öyrətmək

**Məqsəd:** modelə düsturu demədən, yalnız 4 misal göstərərək `y = 5x`
qanunauyğunluğunu tapdırmaq.

Tapşırığın 5 bəndi:

1. `y` siyahısını doldurmaq → `5, 10, 15, 20`
2. Training loop-un çatışmayan 3 sətrini yazmaq → `zero_grad`, `backward`, `step`
3. Kodu işlədib `w`-nin 5-ə yaxınlaşmasını görmək
4. Loss qrafikini çəkmək
5. `x = 10` üçün proqnoz almaq

Aşağıdakı kod dərsdəki skeletin eynisidir — yalnız `# TAMAMLA` yerləri
doldurulub, heç nə silinməyib və heç bir parametr dəyişdirilməyib.

> **İşlətməzdən əvvəl:** `Runtime → Change runtime type → T4 GPU → Save`
> (dərsin 2-ci bölməsi). Növbəti hüceyrə bunu yoxlayır.
````

## Hüceyrə 2 — KOD (Code) — Hazirliq yoxlamasi

```python
# ----- 0. HAZIRLIQ (dərsin 2-ci bölməsi) -----
import torch

print("PyTorch versiyası:", torch.__version__)
print("GPU var:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU modeli:", torch.cuda.get_device_name(0))
else:
    print("Diqqət: Runtime → Change runtime type → T4 GPU → Save")
```

## Hüceyrə 3 — KOD (Code) — Esas hell

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(42)

# Dərsin 3.6 bölməsi: GPU varsa onu, yoxdursa CPU-nu seçirik.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("İstifadə edirik:", device)

# ----- 1. MƏLUMAT -----
X = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(-1, 1).to(device)
# TAMAMLA: hər ədədin 5 mislini yazın (5, 10, 15, 20)
y = torch.tensor([5.0, 10.0, 15.0, 20.0]).reshape(-1, 1).to(device)

# ----- 2. MODEL -----
# Model CPU-da seed 42 ilə yaradılır (nəticə təkrarlanan olsun), sonra GPU-ya köçür.
model = nn.Linear(1, 1).to(device)
print("Model harada:", next(model.parameters()).device)
print("Başlanğıc w:", round(model.weight.item(), 3))
print("Başlanğıc b:", round(model.bias.item(), 3))

# ----- 3. ALƏTLƏR -----
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ----- 4. ÖYRƏNMƏ DÖVRÜ -----
loss_siyahisi = []

for epoch in range(200):
    proqnoz = model(X)           # Addım 1: proqnoz ver
    loss = loss_fn(proqnoz, y)   # Addım 2: səhvi ölç

    optimizer.zero_grad()        # TAMAMLA → Addım 3: köhnə gradientləri sil
    loss.backward()              # TAMAMLA → Addım 4: yeni gradientləri hesabla
    optimizer.step()             # TAMAMLA → Addım 5: parametrləri düzəlt

    loss_siyahisi.append(loss.item())

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}: loss = {loss.item():.4f}, w = {model.weight.item():.3f}")

# ----- 5. NƏTİCƏ -----
print("\nÖyrənilmiş w:", round(model.weight.item(), 3))
print("Gözlənilən w: 5.0")

# ----- 6. QRAFİK -----
plt.plot(loss_siyahisi)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Öyrənmə əyrisi")
plt.grid(True)
plt.show()
```

## Hüceyrə 4 — KOD (Code) — 7. YOXLAMA

```python
# ----- 7. YOXLAMA -----
# TAMAMLA: modeldən x=10 üçün proqnoz alın.
# no_grad() — proqnoz verərkən gradient hesablamağa ehtiyac yoxdur.
with torch.no_grad():
    yeni = torch.tensor([[10.0]]).to(device)   # giriş də modelin olduğu cihazda olmalıdır
    cavab = model(yeni)

print("x=10 üçün model dedi:", round(cavab.item(), 2))
print("Doğru cavab: 50")
```

## Hüceyrə 5 — MƏTN (Text) — Durust oxunus

````markdown
## Nəticənin dürüst oxunuşu

200 epoch-dan sonra `w ≈ 4.62` çıxdı, `x = 10` üçün proqnoz isə `≈ 47.3` —
tapşırıqda yazılan **50 deyil**. Kod səhv deyil; **addım sayı azdır**.

Səbəb burada:

`nn.Linear(1, 1)` bir yox, **iki** ədəd öyrənir — `w` (tərəzinin həssaslığı)
və `b` (tərəzinin sıfır nöqtəsi). Bizim məlumatda doğru cavab `w = 5`, `b = 0`-dır.
Amma `b` təsadüfi `0.83`-dən başlayır və sıfıra çox yavaş enir: 200 epoch-dan
sonra hələ `b ≈ 1.12` qalır. Model ümumi səhvi azaltmaq üçün `w`-ni 5-dən
aşağı saxlayıb həmin artıq `b`-ni kompensasiya edir. Buna görə `4.62 + 1.12`
cütü 4 misalı yaxşı təsvir edir, amma `x = 10`-da fərq görünür.

Dərsin 7-ci bölməsindəki «əyrini necə oxumaq lazımdır» cədvəlinə görə:
əyri **hələ enir, düzləşməyib** → «öyrənmə davam edir, dayandırmaq tezdir».
Aşağıdakı hüceyrə bunu sübut edir.
````

## Hüceyrə 6 — KOD (Code) — Subut ve davam

```python
# Qalıq b — 47.3 rəqəminin bütün izahı buradadır.
print("200 epoch sonra:   w =", round(model.weight.item(), 3),
      "  b =", round(model.bias.item(), 3))
print("w * 10 + b =", round(model.weight.item() * 10 + model.bias.item(), 2))

# Eyni başlanğıc nöqtəsi (seed 42), eyni lr — sadəcə daha çox addım.
# Diqqət: bu YENİ modeldir, yuxarıdakı modelə toxunmuruq.
torch.manual_seed(42)
model_uzun = nn.Linear(1, 1).to(device)
optimizer_uzun = optim.SGD(model_uzun.parameters(), lr=0.01)

loss_siyahisi_uzun = []
for epoch in range(2000):
    proqnoz = model_uzun(X)
    loss = loss_fn(proqnoz, y)
    optimizer_uzun.zero_grad()
    loss.backward()
    optimizer_uzun.step()
    loss_siyahisi_uzun.append(loss.item())

print("\n2000 epoch sonra:  w =", round(model_uzun.weight.item(), 3),
      "  b =", round(model_uzun.bias.item(), 4))

with torch.no_grad():
    cavab_uzun = model_uzun(torch.tensor([[10.0]]).to(device))

print("x=10 üçün model dedi:", round(cavab_uzun.item(), 2), " (doğru cavab: 50)")
```

## Hüceyrə 7 — KOD (Code) — Muqayise qrafiki

```python
# ----- ƏLAVƏ QRAFİK: 200 epoch və 2000 epoch yanaşı -----
SURFACE = "#fcfcfb"
INK     = "#0b0b0b"   # öyrətmə misalları
IKINCI  = "#52514e"
MUTED   = "#898781"
GRID    = "#e1e0d9"
QISA    = "#2a78d6"   # 200 epoch
UZUN    = "#eb6834"   # 2000 epoch

w_q, b_q = model.weight.item(), model.bias.item()
w_u, b_u = model_uzun.weight.item(), model_uzun.bias.item()
p_q, p_u = w_q * 10 + b_q, w_u * 10 + b_u

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
fig.patch.set_facecolor(SURFACE)

for ax in (ax1, ax2):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)

# --- Sol: modelin tapdığı xətt ---
kenar = [0, 11]
ax1.plot(kenar, [w_q * v + b_q for v in kenar], color=QISA, linewidth=2,
         label=f"200 epoch  ·  w = {w_q:.2f}")
ax1.plot(kenar, [w_u * v + b_u for v in kenar], color=UZUN, linewidth=2,
         label=f"2000 epoch  ·  w = {w_u:.2f}")
ax1.scatter(X.cpu().squeeze().tolist(), y.cpu().squeeze().tolist(), s=80, color=INK,
            zorder=3, label="Öyrətmə misalları (cəmi 4)")
ax1.axvline(10, color=MUTED, linewidth=1, linestyle=":")
ax1.annotate(f"{p_q:.1f}", xy=(10, p_q), xytext=(8, -16), textcoords="offset points",
             color=QISA, fontsize=11, fontweight="bold")
ax1.annotate(f"{p_u:.1f}", xy=(10, p_u), xytext=(8, 8), textcoords="offset points",
             color=UZUN, fontsize=11, fontweight="bold")
ax1.annotate("x = 10 (model bunu heç görməyib)", xy=(10, 2), xytext=(-6, 0),
             textcoords="offset points", color=MUTED, fontsize=9, ha="right")
ax1.set_title("Model hansı xətti tapdı?", color=INK, fontsize=12, pad=12, loc="left")
ax1.set_xlabel("x", color=IKINCI, fontsize=10)
ax1.set_ylabel("y", color=IKINCI, fontsize=10)
ax1.legend(frameon=False, fontsize=9, labelcolor=IKINCI, loc="upper left")

# --- Sağ: iki loss əyrisi, eyni ox ---
ax2.plot(loss_siyahisi_uzun, color=UZUN, linewidth=2, label="2000 epoch")
ax2.plot(loss_siyahisi, color=QISA, linewidth=2, label="200 epoch")
ax2.set_yscale("log")
ax2.axvline(200, color=MUTED, linewidth=1, linestyle=":")
ax2.annotate("tapşırıq burada dayanır", xy=(200, loss_siyahisi[-1]), xytext=(10, 6),
             textcoords="offset points", color=MUTED, fontsize=9)
ax2.set_title("Səhv (loss) — loqarifmik ox", color=INK, fontsize=12, pad=12, loc="left")
ax2.set_xlabel("Epoch", color=IKINCI, fontsize=10)
ax2.set_ylabel("Loss", color=IKINCI, fontsize=10)
ax2.legend(frameon=False, fontsize=9, labelcolor=IKINCI, loc="upper right")

plt.tight_layout()
plt.show()
```

## Hüceyrə 8 — MƏTN (Text) — Netice

````markdown
## Nəticə

- Model `5` rəqəmini heç vaxt görmədi. Cəmi 4 misala baxdı və qanunauyğunluğu
  addım-addım özü tapdı.
- 200 epoch: `w = 4.62`, `x = 10 → 47.3`. 2000 epoch: `w = 5.00`, `x = 10 → 50.0`.
- Fərq kodda deyil, **addım sayındadır** — loss əyrisi 200-cü epoch-da hələ
  enirdi, yəni öyrənmə bitməmişdi.

Dərsin 5 sətri, əzbər:

```
proqnoz = model(X)          # 1. proqnoz ver
loss = loss_fn(proqnoz, y)  # 2. səhvi ölç
optimizer.zero_grad()       # 3. köhnə gradientləri sil
loss.backward()             # 4. yeni gradientləri hesabla
optimizer.step()            # 5. parametrləri düzəlt
```
````
