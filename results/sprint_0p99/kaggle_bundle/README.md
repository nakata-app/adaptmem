# Chat-CE v1, Kaggle/Colab bundle (Task 2 train job)

Bu klasörü Kaggle Dataset olarak yükleyip notebook'u çalıştır. Colab da çalışır (.ipynb tek başına).

## İçerik

| Dosya | Ne |
|---|---|
| `train_chat_ce.ipynb` | Tek tıkla train notebook (Kaggle + Colab uyumlu) |
| `task2_train_pairs.jsonl` | 848 (q, doc, label) çift, LongMemEval-S 100 train split'inden |
| `task2_val_pairs.jsonl` | 155 val çifti, train'den ayrı qid'ler |
| `README.md` | Bu dosya |

## Kaggle akışı (~20-30 dk wallclock)

1. https://www.kaggle.com/datasets → New Dataset → bu klasörü yükle ya da sadece iki .jsonl
   - Title: `chat-ce-v1-20260516`
   - Slug otomatik
2. https://www.kaggle.com/code → New Notebook → `train_chat_ce.ipynb`'yi import (File → Import Notebook)
3. Sağ panel → Add data → yüklediğin dataset'i ekle
4. Sağ panel → Settings → Accelerator: **GPU T4 x2** veya **GPU P100**
5. Run All
6. Bittiğinde sağ panel → Output → `chat-ce-v1.zip`'i indir (~90 MB)
7. Mac'e indir, aşağıdaki "Mac'e kurulum"a geç

## Colab akışı (alternatif)

1. https://colab.research.google.com → File → Upload notebook → `train_chat_ce.ipynb`
2. Runtime → Change runtime type → **T4 GPU**
3. Sol panel (klasör ikonu) → Upload → `task2_train_pairs.jsonl` ve `task2_val_pairs.jsonl`'i yükle
4. Run All
5. Sol panelden `chat-ce-v1.zip`'i indir

## Mac'e kurulum (download sonrası)

```bash
mkdir -p ~/Projects/adaptmem/checkpoints/chat-ce-v1-20260516
unzip -d ~/Projects/adaptmem/checkpoints/chat-ce-v1-20260516 ~/Downloads/chat-ce-v1.zip
ls ~/Projects/adaptmem/checkpoints/chat-ce-v1-20260516/  # config.json, model.safetensors, vs olmalı
```

## Sonra Task 4 re-run

`results/sprint_0p99/task4_rerank.py` içinde `CKPT_CODE` yerine yeni path kullanmak için
şu satırı değiştir veya `--ckpt code` benzeri yeni branch ekle. Ya da hızlıca:

```bash
cd ~/Projects/adaptmem && source .venv/bin/activate
python -c "
from sentence_transformers import CrossEncoder
m = CrossEncoder('/Users/macmini/Projects/adaptmem/checkpoints/chat-ce-v1-20260516', max_length=384)
print('loads OK')
"
```

Yüklendikten sonra ban Atakan'a haber, ben Task 4'ü `--ckpt chat` modu ile re-run ederim.
