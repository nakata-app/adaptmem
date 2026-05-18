# CodeCrossEnc-v1 RETRAIN, Colab T4
# Save protection: /tmp local + cp -rv + sync + sleep + Drive API verify
#
# Önceki training kayıp oldu çünkü model.fit(output_path=Drive) FUSE async upload
# tamamlanmadan kernel kapandı. Bu sefer üç katmanlı koruma:
#   1. Önce /tmp/ local SSD'ye save (anlık, garantili)
#   2. cp -rv ile Drive'a explicit kopya (progress görünür)
#   3. sync + sleep(180) + Drive API cloud-side check (cache değil cloud)
#
# Yapıştırma: cell tamamen boşalt, ilk satır `from google.colab import drive`.

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import gc, torch, random, time, subprocess
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
from datasets import load_dataset

torch.cuda.empty_cache(); gc.collect()
random.seed(42)

POS_LIMIT = 100_000
LOCAL_DIR = "/tmp/code-crossenc-v1"
DRIVE_DIR = "/content/drive/MyDrive/adaptmem-bench/code-crossenc/v1"

print("[1] Loading CodeSearchNet python train...", flush=True)
ds = load_dataset("code_search_net", "python", split="train")
print(f"  {len(ds)} raw rows", flush=True)

print(f"[2] Building pairs (limit={POS_LIMIT}, 1 pos + 2 neg each)...", flush=True)
clean = []
for r in ds:
    if len(clean) >= POS_LIMIT:
        break
    doc = (r.get("func_documentation_string") or "").strip()
    code = r.get("func_code_string") or ""
    if len(doc) < 10 or len(code) < 40:
        continue
    query = doc.splitlines()[0][:200]
    clean.append((query, code))
print(f"  {len(clean)} clean positive pairs", flush=True)

all_codes = [c for _, c in clean]
train_examples = []
for query, code in clean:
    train_examples.append(InputExample(texts=[query, code], label=1.0))
    for _ in range(2):
        neg = random.choice(all_codes)
        if neg != code:
            train_examples.append(InputExample(texts=[query, neg], label=0.0))
random.shuffle(train_examples)
print(f"  {len(train_examples)} total examples", flush=True)

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

print("[3] Loading base ms-marco-MiniLM-L-6-v2...", flush=True)
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', num_labels=1, max_length=384, device='cuda')

print(f"[4] Training 1 epoch -> {LOCAL_DIR} (LOCAL, hızlı, garantili)...", flush=True)
t0 = time.time()
model.fit(
    train_dataloader=train_dataloader,
    epochs=1,
    warmup_steps=500,
    optimizer_params={'lr': 2e-5},
    output_path=LOCAL_DIR,
    save_best_model=False,
    show_progress_bar=True,
)
elapsed = (time.time()-t0)/60
print(f"[5] Training done in {elapsed:.1f}min", flush=True)

# === SAVE PROTECTION: 3 KATMAN ===

print("[6.1] Local files (training output):", flush=True)
subprocess.run(["ls", "-la", LOCAL_DIR])

print("\n[6.2] cp -rv LOCAL -> DRIVE (explicit kopya, progress görünür)...", flush=True)
subprocess.run(["mkdir", "-p", DRIVE_DIR])
result = subprocess.run(["cp", "-rv", f"{LOCAL_DIR}/.", DRIVE_DIR], capture_output=True, text=True)
print(result.stdout[-2000:] if result.stdout else "")
print(result.stderr[-500:] if result.stderr else "")

print("\n[6.3] sync + sleep(180) FUSE flush...", flush=True)
subprocess.run(["sync"])
for i in range(6):
    time.sleep(30)
    print(f"  {(i+1)*30}/180s elapsed", flush=True)

print("\n[6.4] FUSE-side ls -la (cache gösterebilir):", flush=True)
subprocess.run(["ls", "-la", DRIVE_DIR])

print("\n[6.5] Drive API CLOUD-side check (cache bypass)...", flush=True)
try:
    from googleapiclient.discovery import build
    from google.colab import auth
    auth.authenticate_user()
    drive_api = build("drive", "v3")

    folder_q = drive_api.files().list(
        q="name='v1' and mimeType='application/vnd.google-apps.folder'",
        fields="files(id,name,parents,modifiedTime)"
    ).execute()
    print(f"  v1 folders found: {len(folder_q.get('files', []))}")
    for f in folder_q.get("files", []):
        print(f"    {f}")

    if folder_q.get("files"):
        # Find our specific v1 folder (parent should be code-crossenc)
        for folder in folder_q["files"]:
            files = drive_api.files().list(
                q=f"'{folder['id']}' in parents",
                fields="files(id,name,size,modifiedTime,mimeType)"
            ).execute()
            print(f"\n  Files in folder {folder['id']}:")
            for f in files.get("files", []):
                size_mb = int(f.get('size', 0))/1e6 if f.get('size') else 0
                print(f"    {f['name']:40s} {size_mb:7.2f}MB  {f.get('modifiedTime', '')}")
except Exception as e:
    print(f"  Drive API check failed: {e}")
    print("  (Auth gerekli olabilir, sonraki cell'de manuel kontrol)")

print("\n[7] DONE. Eğer [6.5] çıktısında model.safetensors görünüyorsa Drive'da kesin var.")
print("    Şimdi de share link al + gdown ile Mac'e yedekle:")
print(f"    1) Colab'da Drive folder permissions cell koş ({DRIVE_DIR})")
print("    2) Mac'te: gdown --folder <share_link>")
