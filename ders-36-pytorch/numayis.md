# Kursda nümayiş — 6 addım və müəllim sualları

Notebook: <https://colab.research.google.com/drive/1FR3zFhiDLb5t5p5vUhwzLqpC1wbfJteg>

Kod artıq işlədilmiş vəziyyətdə saxlanılıb — internet zəif olsa belə çıxışlar
və qrafiklər görünəcək. Yenidən işlətmək lazım gəlsə: `Runtime → Run all`
(~10 saniyə, GPU lazım deyil).

---

## Nümayiş skripti

**1. Şərti bir cümlə ilə qoy.**
«Modelə `y = 5x` düsturunu demədim. Cəmi 4 misal verdim: 1→5, 2→10, 3→15, 4→20.
Qalanını özü tapmalı idi.»

**2. Training loop-un 5 sətrini göstər** (2-ci hüceyrə, dövrün içi).
«Bütün süni intellektin öyrənməsi bu 5 sətirdir: proqnoz ver → səhvi ölç →
köhnə gradientləri sil → yeniləri hesabla → parametrləri düzəlt.»
Tapşırıqda çatışmayan 3 sətir məhz `zero_grad`, `backward`, `step` idi.

**3. Loss qrafikinə keç.**
«Əyri sürətlə enir, sonra yavaşıyır — bu, sağlam öyrənmənin formasıdır.
Dərsdəki cədvələ görə sıçrayan əyri `lr`-in böyük olduğunu, düz xətt isə
kiçik olduğunu göstərərdi. Burada hər ikisi qaydasındadır.»

**4. `x = 10` nəticəsini aç — və özün qabaqla.**
«Model 10 rəqəmini heç görməyib. Cavab **47.3** verdi. Tapşırıqda 50 yazılıb.
Fərqin səbəbini bilirəm.» *(Bu, bütün nümayişin ən güclü nöqtəsidir — səhvi
gizlətmək yox, izah etmək.)*

**5. Səbəbi sübutla göstər** (5-ci hüceyrə).
«`nn.Linear` iki ədəd öyrənir: `w` və `b`. Doğru cavab `w=5, b=0`-dır, amma `b`
təsadüfi `0.83`-dən başlayır və çox yavaş enir — 200 epoch-dan sonra hələ
`b ≈ 1.12`. Model `w`-ni aşağı saxlayıb bunu kompensasiya edir. Skeleti
dəyişmədim; **ayrıca yeni modeli** eyni seed və eyni `lr` ilə 2000 epoch
öyrətdim → `w = 5.00`, `x=10 → 49.99`.»

**6. Müqayisə qrafiki ilə bağla** (6-cı hüceyrə).
«Solda: mavi xətt 200 epoch, narıncı 2000. 4 misalın üstündən hər ikisi keçir,
fərq yalnız məlumatdan kənarda — `x = 10`-da — görünür. Sağda: loss loqarifmik
oxda, nöqtəli xətt tapşırığın dayandığı yeri göstərir. Yəni problem kodda yox,
addım sayında idi.»

---

## Müəllim sualları və cavablar

**«Niyə 50 çıxmadı, kodunda səhv var?»**
Xeyr. Skelet olduğu kimidir. `nn.Linear` `b`-ni də öyrənir; 200 addımda `b`
sıfıra enməyə macal tapmır (`b ≈ 1.12`) və `w` onu kompensasiya edir.
2000 addımda `b ≈ 0.005`, `w = 5.00`, proqnoz 49.99.

**«`optimizer.zero_grad()` nə edir, olmasa nə olar?»**
PyTorch gradientləri **toplayır**. Silməsən, əvvəlki addımın gradienti üstünə
gəlir və model səhv istiqamətə, getdikcə böyüyən addımlarla gedir. Ən pisi —
heç bir xəta mesajı çıxmır, model sadəcə öyrənmir. Dərsin 4.5 bölməsi.

**«`loss.backward()` nə hesablayır?»**
Hər parametr üçün gradienti — «bu ədədi bir az artırsan, səhv nə qədər dəyişir»
sualının cavabını. Dumanlı dağ analogiyasında bu, yamacın istiqamətidir.

**«`torch.no_grad()` niyə lazımdır?»**
Proqnoz verərkən öyrənmirik, ona görə gradient saxlamağa ehtiyac yoxdur —
yaddaşa qənaət edir və sürəti artırır. Modeli FastAPI ilə servis edəndə bunu
unutmaq serverin yaddaşının dolması deməkdir.

**«GPU işlətdinmi?»**
Yox, lazım deyil. 4 misal və 2 parametr üçün CPU kifayətdir. GPU minlərlə
paralel vurma lazım olanda — böyük modellərdə — qazanc verir.

**«`lr`-i dəyişsən nə olar?»**
`0.01` → `0.1` daha tez yığardı və 200 epoch-da 50-yə çatardı. Amma çox
böyükdə (məsələn `1`) loss yuxarı-aşağı sıçrayır və model korlanır — dərsin
6.2 cədvəlindəki üç haldan üçüncüsü.

**«`b` nədir?»**
Tərəzinin sıfır nöqtəsi. `nn.Linear` daxildə `çıxış = w * giriş + b` hesablayır;
`w` həssaslıq, `b` isə sürüşmədir.

---

## Rəqəmlərin mənbəyi

Aşağıdakı gözlənilən çıxış `torch.manual_seed(42)` ilə eyni riyaziyyatın
(MSE `mean`, SGD, `lr = 0.01`, başlanğıc `w = 0.7645`, `b = 0.8300`)
addım-addım hesablanmasıdır. **Colab-da işlətdikdən sonra real çıxışla
tutuşdur** — fərq varsa, mətndəki rəqəmlər real nəticəyə görə düzəldilməlidir.

| | 200 epoch | 2000 epoch |
|---|---|---|
| `Epoch 50` | loss 0.5156 · w 4.404 | — |
| `Epoch 100` | loss 0.3820 · w 4.487 | — |
| `Epoch 150` | loss 0.2831 · w 4.559 | — |
| `Epoch 200` | loss 0.2097 · w 4.620 | — |
| Öyrənilmiş `w` | **4.62** | **4.998** |
| Qalıq `b` | **1.117** | **0.0051** |
| `x = 10` | **47.32** | **49.99** |

Qeyd: dərsin öz `y = 2x` nümunəsindəki «gözlənilən çıxış» (`Epoch 100: w = 1.980`)
heç bir `lr` / başlanğıc kombinasiyası ilə təkrarlanmır — o rəqəmlər real
işlətmə deyil, illüstrasiyadır. Ona görə 200 epoch-da 50 gözləmək düzgün deyil.
