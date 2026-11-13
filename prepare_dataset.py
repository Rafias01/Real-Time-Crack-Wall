import hashlib
import random
import shutil
from collections import defaultdict
from pathlib import Path
import yaml
from PIL import Image

def prepare_dataset(raw_dir, out_dir, seed=42):
    """Validasi YOLO detection, kelompokkan duplikat persis, lalu split 80/10/10."""
    raw, out = Path(raw_dir).resolve(), Path(out_dir).resolve()
    if raw == out or raw in out.parents or out in raw.parents:
        raise ValueError('Folder sumber dan hasil harus terpisah.')
    ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = sorted(f for f in raw.rglob('*') if f.is_file() and f.suffix.lower() in ext and f.parent.name == 'images')
    if not images:
        raise ValueError('Folder images tidak ditemukan. Periksa hasil ekspor YOLO.')
    groups = defaultdict(list)
    missing, empty, invalid = [], [], []
    for img in images:
        label = img.parent.parent / 'labels' / (img.stem + '.txt')
        if not label.exists():
            missing.append(str(img))
            continue
        try:
            with Image.open(img) as im:
                im.verify()
            rows = label.read_text(encoding='utf-8').splitlines()
            for row in rows:
                if not row.strip():
                    continue
                parts = row.split()
                if len(parts) != 5 or int(parts[0]) != 0:
                    raise ValueError('Harus 1 kelas dan 5 kolom YOLO detection')
                x,y,w,h = map(float, parts[1:])
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    raise ValueError('Koordinat bbox tidak valid')
            if not any(r.strip() for r in rows):
                empty.append(str(img))
            digest = hashlib.sha256(img.read_bytes()).hexdigest()
            groups[digest].append((img,label))
        except Exception as e:
            invalid.append((str(img),str(e)))
    print(f'Gambar sumber: {len(images)}; label kosong: {len(empty)}; label hilang: {len(missing)}; invalid: {len(invalid)}')
    if missing or invalid:
        print('Contoh label hilang:', missing[:5], 'Contoh invalid:', invalid[:5])
        raise ValueError('Audit label/gambar bermasalah sebelum melanjutkan. Untuk gambar negatif terverifikasi, buat file .txt kosong.')
    # Satu salinan untuk setiap hash; cegah kebocoran duplikat byte-identik.
    # Jika label duplikat berbeda, minta pemeriksaan manual.
    unique = []
    for digest, group in sorted(groups.items()):
        canonical = {tuple(sorted(row.strip() for row in lab.read_text(encoding='utf-8').splitlines() if row.strip())) for _,lab in group}
        if len(canonical) != 1:
            raise ValueError(f'Duplikat dengan anotasi berbeda: {group[0][0]}')
        unique.append((digest, *group[0]))
    random.Random(seed).shuffle(unique)
    n = len(unique)
    if n < 10:
        raise ValueError('Dataset unik terlalu kecil.')
    ntrain, nval = int(n * .8), int(n * .1)
    splits = {'train': unique[:ntrain], 'val': unique[ntrain:ntrain+nval], 'test': unique[ntrain+nval:]}
    # Folder hasil boleh dibuat ulang, tetapi hanya folder yang ditandai oleh fungsi ini.
    marker = out / '.prepared_wall_crack'
    if out.exists():
        if not marker.exists():
            raise FileExistsError(f'Folder hasil sudah ada: {out}. Pilih folder lain.')
        shutil.rmtree(out)
    out.mkdir(parents=True)
    marker.touch()
    for split, items in splits.items():
        for kind in ('images','labels'):
            (out / split / kind).mkdir(parents=True)
        for digest, img, label in items:
            shutil.copy2(img, out / split / 'images' / (digest + img.suffix.lower()))
            shutil.copy2(label, out / split / 'labels' / (digest + '.txt'))
    config = {'path': str(out), 'train': 'train/images', 'val': 'val/images', 'test': 'test/images', 'nc': 1, 'names': ['crack']}
    (out / 'data.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    print(f'Duplikat persis dihapus: {len(images)-n}. Gambar unik: {n}.')
    print({key: len(items) for key,items in splits.items()})
    if n < 5000:
        print('PERHATIAN: jumlah unik di bawah 5.000. Jika minimal 5.000 gambar unik wajib, tambahkan data beranotasi yang independen sebelum training.')
    return out / 'data.yaml'
