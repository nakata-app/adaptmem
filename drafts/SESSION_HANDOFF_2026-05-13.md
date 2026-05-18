# AdaptMem, Session Handoff (2026-05-13)

Yeni session bu dosyayı okuyarak devam etsin. Mnemonics ns: `proj:adaptmem`.

---

## 1. Bağlam (neyle uğraşıyoruz)

AdaptMem = MiniLM tabanlı bi-encoder retrieval modeli. Domain-specific fine-tune yaklaşımıyla **encoder axis** üzerine çalışıyor. Test arena iki:

- **CodeSearchNet python full 22k test set** (in-domain)
- **jphein/mempalace chunk_strategy_ablation.py probe** (cross-harness, n=20 mempalace .py corpus)

MemPalace fork tarafında jphein 4-axis modeli kurmuş (storage / encoder / retrieval / consumption), adaptmem encoder ekseninde konumlanıyor. Daha önce **negative result** flag'lenmiş (FT-300 LongMemEval üzerinde eğitilmişti, kod corpus'unda transfer ölmüştü). Bu session domain-mismatch confound'unu test etti.

---

## 2. Bu session'da kanıtlanan sayılar

### 2.1 CodeSearchNet python full 22k (in-domain)

| Model | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| Baseline CodeBERT (no FT) | 0.6477 | 0.8551 | 0.8972 | 0.7406 |
| FT-Code-300 | 0.800 | 0.941 | 0.959 | 0.864 |
| FT-Code-1000 | 0.902 | 0.976 | 0.981 | 0.936 |
| **FT-Code-5000** | **0.926** | **0.982** | **0.985** | **0.952** |

Δ baseline → FT-Code-5000: **+0.278 R@1, +0.211 MRR**. Encoder FT kod domain'inde çalışıyor, domain-mismatch confound elimine edildi.

### 2.2 Cross-harness jphein probe (mempalace .py corpus, n=20, 6 strategy × cs400/cs800)

MRR per strategy:

| Strategy | default | FT-300 | FT-Code-300 | FT-Code-1000 | FT-Code-5000 |
|---|---|---|---|---|---|
| A_paragraph cs400 | 0.4583 | 0.5125 | 0.4917 | 0.5500 | 0.5433 |
| A_paragraph cs800 | 0.4850 | 0.5142 | 0.5058 | 0.4780 | 0.5333 |
| B_heading cs400 | 0.4583 | 0.5125 | 0.4917 | 0.5500 | 0.5433 |
| B_heading cs800 | 0.4850 | 0.5142 | 0.5058 | 0.4875 | 0.5292 |
| C_ast_python cs400 | 0.4583 | 0.4833 | 0.5417 | 0.5750 | 0.5600 |
| **C_ast_python cs800** | **0.5600** | 0.5542 | 0.5333 | 0.5588 | **0.5167** |

5/6 strategy pozitif. C-AST cs800 (jphein'in orijinal negative-result strategy'si) FT-Code-5000'de **-0.043 MRR**.

### 2.3 Bootstrap %95 CI (paired, 10K resample, n=20)

C-AST cs800 default vs FT-Code-5000: Δ = −0.043, **CI = [−0.125, +0.030]**, P(Δ>0) = 0.126.

**Kritik bulgu:** Hiçbir strategy n=20'de tamamen significant değil (tüm CI'lar 0'ı içeriyor). Orijinal FT-300 −0.006 ve yeni FT-Code-5000 −0.043 ikisi de **noise floor**'unda. "Significant regression" iddiası tutmuyor.

### 2.4 RRF 3-way ensemble (default + FT-Code-1k + FT-Code-5k)

| Strategy | default | best solo | **3-way RRF** | Δ vs best solo |
|---|---|---|---|---|
| A_paragraph cs400 | 0.4583 | 0.5500 | **0.6123** | +0.062 |
| A_paragraph cs800 | 0.4850 | 0.5333 | **0.6023** | +0.069 |
| B_heading cs400 | 0.4583 | 0.5500 | **0.6123** | +0.062 |
| B_heading cs800 | 0.4850 | 0.5292 | **0.5981** | +0.069 |
| C_ast_python cs400 | 0.4583 | 0.5750 | **0.6373** | +0.062 |
| **C_ast_python cs800** | **0.5600** | 0.5600 | **0.6356** | **+0.076** |

6/6 strategy pozitif lift. R@10 uniform 70% (vs default 60-65%). C-AST cs800 orijinal "negative" strategy en büyük ensemble lift (+0.076).

### 2.5 Scaling non-monotonic

FT-Code-1000 4/6 strategy'de lokal optimum, FT-Code-5000 ya tied ya hafif altta. Bu **CodeSearchNet distribution'una overfit** sinyali, saf kod retrieval'da R@1 yukarı gitmeye devam ediyor (0.902→0.926), ama mempalace karma corpus'ta tepe 1000 step'te.

### 2.6 Generic cross-encoder reranker, negative transfer

İki ayrı cross-encoder denendi CodeSearchNet 22k test üzerinde:

| Reranker | Test sample | R@1 ortalama | vs FT-Code-5000 alone (0.926) |
|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 3500 (early stop) | ~0.90 | **−0.026** |
| `BAAI/bge-reranker-base` | 3000 (early stop) | ~0.90 | **−0.026** |

İki ayrı mimarı aynı sonuç: **generic English-QA / IR reranker'lar code retrieval'da negative transfer yapar**. FT bi-encoder'ın code-specific sinyalini bozar. Code-specific reranker zorunlu.

### 2.7 CodeCrossEnc-v1 training (KAYIP, yeniden gerek)

Base: `cross-encoder/ms-marco-MiniLM-L-6-v2` (90MB).
Data: CodeSearchNet python train 100K positive + 2 random negative each = 300K pair.
1 epoch, batch=32, lr=2e-5, warmup=500, max_length=384.

Loss progression:
- 500: 0.0094 → 1500: 0.0007 (minimum cluster) → 8000: 0.000117 (en düşük)
- Random negative discrimination near-perfect

**Training tamamlandı 74.9dk T4 ama Drive'a SENKRONIZE OLMADI.** Training cell `[6] Saved to /content/drive/MyDrive/adaptmem-bench/code-crossenc/v1` mesajı atmıştı; ancak 2026-05-14 00:30 civarı eval cell'i koşturmaya çalışırken `FileNotFoundError: Path /content/drive/MyDrive/adaptmem-bench/code-crossenc/v1 not found`. Drive `find` taraması cross-encoder model dosyalarını HİÇBİR YERDE bulamadı. **Sebep: Colab Drive FUSE write-back cache trap**, save FUSE cache'e yazıldı, [6] Saved mesajı çıktı, ama runtime kapanmadan önce cache → real Drive upload tamamlanmadı. 75dk T4 boşa gitti.

**Yeniden training kuralı:** save sonrası
```python
trainer.save_model("/content/drive/MyDrive/.../v1")
import subprocess, time
subprocess.run(["sync"])
time.sleep(120)  # FUSE upload buffer
!ls -la /content/drive/MyDrive/.../v1  # canlı doğrulama
```
Bu olmadan asla "kaydedildi" diye güvenme.

⚠️ Trivial discrimination çok güçlü, hard candidate'lar arasında performans eval'de görülecek (training yeniden yapılınca).

---

## 3. Drive linkleri (anyone-with-link reader)

| Model | Link |
|---|---|
| FT-Code-300 | https://drive.google.com/drive/folders/1fe5t5LWWHFGDV5CC5GcfCk5aOVaRKKRb |
| FT-Code-1000 | https://drive.google.com/drive/folders/1QhjDc63M4vKdOxVMP7ZbpyhYoLcXjnlC |
| FT-Code-5000 | https://drive.google.com/drive/folders/1GZXGQG4LJL8jm1ajbo8JgiIPPICf_QtT |
| **CodeCrossEnc-v1** | **share link henüz alınmadı**, Colab'da Drive klasöre permissions/create cell koş |

Mac local: `/Users/macmini/Projects/adaptmem/checkpoints/ft-code-{300,1000,5000}-drive/`. CodeCrossEnc-v1 sadece Drive'da.

---

## 4. Yardımcı script'ler (Mac local)

- `benchmarks/codesearchnet_eval.py`, bi-encoder eval (R@k + MRR, JSONL out)
- `benchmarks/jphein_chunk_x_encoder.py`, cross-harness probe wrapper (FT model swap'la chunk_strategy_ablation.py koşturur)
- `benchmarks/bootstrap_paired_mrr.py`, paired bootstrap %95 CI for paired runs
- `benchmarks/rrf_ensemble.py`, 2-way RRF surrogate
- `benchmarks/rrf_ensemble_nway.py`, N-way RRF surrogate (N=2,3,4 kombolar)

---

## 5. Çıktı dosyaları

- `benchmarks/v335/chunk_x_encoder/`, original FT-300 vs default (Atakan'ın eski koşumu)
- `benchmarks/v335/chunk_x_encoder_ftcode300/`, FT-Code-300 vs default
- `benchmarks/v335/chunk_x_encoder_ftcode1000/`, FT-Code-1000 vs default
- `benchmarks/v335/chunk_x_encoder_ftcode5000/`, FT-Code-5000 vs default (asıl probe sonuçları)

Her klasörde `ablation_default_encoder.{json,log}` + `ablation_ft300_encoder.{json,log}` (label legacy, içerik gerçek FT model'a karşılık geliyor).

---

## 6. Sırada yapılacaklar

### 6.1 Hemen (B sprint devamı, öncelik sırasıyla)

1. **CodeCrossEnc-v1 eval CodeSearchNet 22k** (Colab T4, ~15-20dk)
   - **Cell hazır:** `drafts/codecrossenc_v1_eval_colab.py` (6 hücreye bölünmüş, Colab'a yapıştır)
   - Pipeline: FT-Code-5000 bi top-20 retrieve → CodeCrossEnc-v1 cross rerank → R@1/R@5/R@10/MRR
   - Karşılaştırma referansları:
     - Baseline CodeBERT alone: R@1 = 0.6477
     - FT-Code-5000 bi alone: R@1 = 0.926, MRR = 0.952
     - FT-Code-5000 + ms-marco rerank: R@1 ~0.90 (negatif transfer)
     - FT-Code-5000 + BGE rerank: R@1 ~0.90 (negatif transfer)
   - **Karar matrisi (3-tier):**
     - R@1 ≥ 0.95 → en iyi: encoder + reranker compose ediyor, jphein post için 6. pozitif kart, üretim için bi+cross pipeline
     - R@1 0.93-0.95 → marjinal pozitif: kabul, üretime alınabilir, cross-harness probe sırada
     - R@1 < 0.93 → trivial overfit kanıtı (loss curve 8000 step minimum + relaps zaten ima ediyor), hard negative mining v2 (FT-Code-5000 ile her query top-20 retrieve → yanlış candidate'lar = hard negative → CodeCrossEnc-v1 üzerinden ek 1 epoch)
   - Çıktı: `/content/drive/MyDrive/adaptmem-bench/eval/codecrossenc-v1/summary.json`
   - Drive share link CodeCrossEnc-v1 için henüz alınmadı, Colab'da klasör → Share → anyone with link reader yapılacak (Mac'e gdown ile çekmek gerekirse)

2. **Jphein post final** (`drafts/jphein_chunk_x_encoder_response.md` taslak hazır)
   - ✅ CLI bug fix: `codesearchnet_eval.py` reproduce komutu (--checkpoint/--n/--out)
   - ✅ §7 cross-encoder: generic reranker negative transfer findings + CodeCrossEnc-v1 pending notu
   - ✅ Reproduce §: 3 FT-Code Drive linki dolduruldu
   - ⏳ CodeCrossEnc-v1 eval sonucu gelir gelmez §7 placeholder ("__PENDING__") ve `__DRIVE_LINK_PENDING__` doldurulacak
   - ⏳ Eğer R@1 ≥ 0.93 → §4 sonuna "reranker compose" minik bölümü eklenecek (RRF 3-way + cross-encoder fusion)
   - Atakan review → post Discussion #1384 (https://github.com/MemPalace/mempalace/discussions/1384)

3. **Mnemonics özet** (bu session'ın tüm önemli sayıları zaten kaydedildi)

### 6.2 Orta vadeli (CodeCrossEnc-v1 sonucuna göre)

4. **Hard negative mining → CodeCrossEnc-v2:**
   - FT-Code-5000 ile CodeSearchNet train her query için top-20 retrieve
   - Doğru olmayan top-K = hard negative (random negative yerine)
   - CodeCrossEnc-v1 üzerinden 1 epoch daha eğit
   - Beklenen +0.01-0.02 R@1 lift (hard negative discrimination çok daha güçlü)

5. **Multi-task hybrid FT-Code-v2 (A planı):**
   - CodeSearchNet python (~457K pair) + sentetik mempalace pair (Claude API ile mempalace corpus'tan üretilecek ~5-10K) + markdown-code pair'leri
   - Tek bi-encoder, çok-domain awareness
   - Trade-off: saf CodeSearchNet R@1 -2 ila -5 puan düşebilir, mempalace +0.05-0.10 MRR
   - Bir gün Colab T4

6. **Bigger probe set chunk_x_encoder için:**
   - n=20 hand-curated probe yetmiyor (tüm CI'lar 0'ı içeriyor)
   - Mempalace git log + issue references'tan derive edilen 100+ probe yazılmalı
   - Bootstrap power'ı artırır, real statistical significance test edilebilir

### 6.3 Uzun vadeli (gerekirse)

7. **In-domain FT (D planı):** mempalace corpus'tan sentetik pair → FT-Code-5000 ek 500 step. Overfit riski yüksek (corpus tiny ~3700 chunk). Sadece 4, 5, 6 yetersizse.

8. **Larger backbone:** 384-dim MiniLM yerine 768-dim BGE-base + FT. R@1 +2-5 ama 3x büyük model. CodeCrossEnc sonucuna göre değer kararı.

9. **Production pipeline:** FT-Code-5000 bi-encoder + (CodeCrossEnc-v1 pozitifse) cross-encoder reranker. Familiar/MemPalace/RAG için ortak retrieval katmanı.

---

## 7. Açık sorular + hipotezler

- **CodeCrossEnc-v1'in trivial discrimination'ı eval'de tutar mı?** Loss 0.0001 minimum gördük, ama bu **random negative ayırma** üzerinden. Hard candidate'lar arasında ranking quality test edilmedi.
- **Domain hybrid FT (A planı) gerçekten lift verir mi?** Trade-off var (saf CodeSearchNet R@1 düşer). Cost/benefit net değil, deneysel kanıt lazım.
- **Reranker axis sadece code-aware ile mi çalışıyor?** İki generic reranker negatif, **CodeCrossEnc** ile pozitif çıkarsa "yes". Sonuç negatifse "reranker axis fundamentally kod retrieval'a uymuyor" daha kuvvetli iddia.

---

## 8. Pratik notlar

- **GitHub Actions runner** durduruldu disk taşıma için, başlangıç: `cd /Users/macmini/actions-runner && ./svc.sh start`
- **WD-Backup'a Projects yedeklendi** (`/Volumes/WD-Backup/Projects/`). 25G → 5G (filter ile node_modules/.venv/checkpoints/run*.jsonl exclude). Mac iç hala primary, symlink kurulmadı.
- **Drive checkpoint paths**: `/content/drive/MyDrive/adaptmem-bench/ft-code/ft-code-{300,1000,5000}/model` (Colab path), Mac local Drive olmadığı için `/Users/macmini/Projects/adaptmem/checkpoints/ft-code-{300,1000,5000}-drive/ft-code-*/model/` (gdown ile çekilmiş).
- **CodeCrossEnc-v1**: sadece Colab Drive'da. Mac'e gerekirse share link al + gdown.

---

## 9. Önemli karar memory'leri (mevcut, korunur)

- Zeus testleri DeepSeek V4 Pro zorunlu (`feedback_zeus_tests_use_deepseek_v4pro.md`)
- ANTHROPIC_API_KEY kod'da yasak (`feedback_no_anthropic_api_in_code.md`)
- Gemini API yasak (`feedback_no_gemini_api.md`)
- Tek free API NIM/NVIDIA (`feedback_only_nvidia_free_api.md`)
- Em-dash yasak yazılı içerikte (`feedback_no_emdash.md`)
- TR/EN i18n her iki dil zorunlu (`feedback_i18n_both_languages.md`)
- Edge-case-first kuralı (CLAUDE.md)
- Karpathy guidelines (CLAUDE.md)
- /goal hedef odaklı icra çerçevesi (CLAUDE.md)
- Mnemonics V2 RELEASED ve sprint state ayrı memory'de
