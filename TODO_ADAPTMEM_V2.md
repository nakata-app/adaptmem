# AdaptMem v2 - R@1 0.99 Hedefi

Mevcut: BGE-large FT-300, R@1=0.950, R@5=0.995, R@10=1.000 (MemPalace 0.920 gecildi)
Hedef: R@1 >= 0.98 (200 soruda max 4 miss, ideal 2)

## Kaggle v2 sonucu bekleniyor
- 400 train / 100 test + 5 epoch + cross-encoder rerank (bge-reranker-v2-m3)
- Script: `benchmarks/kaggle_v2_cell.py`

## Kod degisiklikleri (priority order) - TAMAMLANDI

### 1. Multi-negative mining [DONE]
- `adaptmem/miner.py`: `n_negatives` parametresi, her query icin K hard negative
- `adaptmem/types.py`: `TrainConfig.n_negatives: int = 1`
- Test: `tests/test_miner.py` (6 test)

### 2. Gradient accumulation [DONE]
- `adaptmem/core.py`: `accumulation_steps` parametresi `base.fit()`'e wired
- `adaptmem/types.py`: `TrainConfig.gradient_accumulation_steps: int = 1`
- batch_size=2 x accum=16 = efektif 32 batch, VRAM degismez

### 3. CachedMultipleNegativesRankingLoss [DONE]
- `adaptmem/core.py`: `loss_type` switch (mnrl | cached_mnrl)
- `adaptmem/types.py`: `TrainConfig.loss_type: str = "mnrl"`
- CachedMNRL gradient cache ile buyuk batch, MNRL'den daha verimli

### 4. Cross-encoder reranker default [DONE]
- `adaptmem/core.py`: default reranker BAAI/bge-reranker-v2-m3 oldu
- Eski: cross-encoder/ms-marco-MiniLM-L-12-v2 (kucuk, 33M)
- Yeni: BAAI/bge-reranker-v2-m3 (buyuk, 560M, cok daha iyi)

### 5. Eval pipeline miss analizi [DONE]
- `benchmarks/longmemeval_eval.py`: `r1_misses` alani results dict'e eklendi
- Her miss: question_id, question_type, question, expected vs retrieved
- cmd_test() miss'leri stdout'a question_type breakdown ile yaziyor

## Diger degisiklikler (session 3)
- [x] `__init__.py`: TrainConfig top-level export eklendi
- [x] `__init__.py`: versiyon 0.6.0 -> 0.7.0 (breaking: reranker default degisti)
- [x] `tests/test_train_config.py`: 7 yeni integration test (config round-trip, wiring)
- [x] `benchmarks/kaggle_v3_cell.py`: tum yeni feature'lari kullanan optimal notebook

## Sonraki adimlar (Kaggle v3 kostu mu)
- [ ] Kaggle v3 sonucunu calistir, R@1 != 0.98 ise:
  - n_negatives=5 dene (daha fazla sinyal)
  - top_k_mine=20 dene (daha zor negatives bulsun)
  - epoch=8 dene (daha fazla iterasyon)
  - learning_rate sweep (1e-5, 3e-5, 5e-5)
- [ ] Miss analizine gore: zayif question_type'lara ozel augmentation
- [ ] Mnemonics baglantisi: MNEMONICS_ENCODER_MODEL + DIM 384->1024

## Tum session degisiklikleri (2026-05-23)

Session 1:
- [x] core.py: model_kwargs destegi (trust_remote_code)
- [x] core.py: save/load'da model_kwargs persist
- [x] benchmarks/kaggle_single_cell.py: BGE-large FT notebook
- [x] benchmarks/kaggle_v2_cell.py: cross-encoder rerank + 400 train + 5 epoch
- [x] benchmarks/data/split_ids_300_200.json: kopyalandi

Session 2:
- [x] miner.py: n_negatives parametresi
- [x] types.py: TrainConfig'e n_negatives + gradient_accumulation_steps + loss_type
- [x] core.py: gradient accum + cached_mnrl loss wiring
- [x] tests/test_miner.py: n_negatives testleri
- [x] benchmarks/longmemeval_eval.py: R@1 miss analizi

Session 3:
- [x] core.py: default reranker -> BAAI/bge-reranker-v2-m3
- [x] __init__.py: TrainConfig export + v0.7.0
- [x] tests/test_train_config.py: 7 integration test
- [x] benchmarks/kaggle_v3_cell.py: multi-neg + accum + cached_mnrl notebook

## Notlar
- Stella (dunzhang/stella_en_400M_v5) Kaggle'da xformers uyumsuzlugu yuzunden calismadi
- BGE-large sorunsuz calisti, 335M param yeterli
- T4 (15.6GB) batch_size=2 zorunda, gradient accumulation ile asabiliriz
- AdaptMem PyPI'da v0.2.1, GitHub'da artik v0.7.0
- 51 test yesil (44 onceki + 7 yeni)
