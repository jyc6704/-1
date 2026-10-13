"""Immutable source archives, exact-duplicate audit, paired ten-color design."""
from __future__ import annotations

import hashlib
import gzip
from pathlib import Path
import shutil
import struct
import zipfile
import numpy as np
import torch
from torchvision.datasets import MNIST, EMNIST
from torchvision.datasets.utils import download_url

from experiment_io import atomic, digest, read_json, save_npz, stable_seed, write_json


def ensure_emnist_digits(data_root, download):
    """Verify the official ZIP, then stream only the four Digits IDX members.

    torchvision's generic downloader also expands all unused letter/byclass splits.
    Avoid that multi-GB expansion and its whole-file decompression buffers.
    """
    raw=Path(data_root)/'EMNIST'/'raw'
    required={}
    for split,count in [('train',240000),('test',40000)]:
        required[f'emnist-digits-{split}-images-idx3-ubyte']=(2051,count,28,28)
        required[f'emnist-digits-{split}-labels-idx1-ubyte']=(2049,count)
    def valid(path,header):
        if not path.is_file():return False
        size=4*len(header)
        expected=size+int(np.prod(header[1:]))
        with path.open('rb') as stream:
            return path.stat().st_size==expected and stream.read(size)==struct.pack('>'+'I'*len(header),*header)
    missing=[name for name,header in required.items() if not valid(raw/name,header)]
    if not missing:return
    if not download:
        raise FileNotFoundError('EMNIST Digits IDX cache is missing/incomplete. Rerun without --offline.')
    raw.mkdir(parents=True,exist_ok=True)
    archive=raw/'gzip.zip'
    existed=archive.exists()
    download_url(EMNIST.url,str(raw),filename='gzip.zip',md5=EMNIST.md5)
    with zipfile.ZipFile(archive) as bundle:
        for name in missing:
            def extract(output):
                with bundle.open('gzip/'+name+'.gz') as member, gzip.GzipFile(fileobj=member) as pixels:
                    shutil.copyfileobj(pixels,output,length=1024*1024)
            atomic(raw/name,extract)
            if not valid(raw/name,required[name]):raise ValueError('Invalid extracted EMNIST IDX: '+name)
    # This function's own download is redundant once all Digits files are intact.
    if not existed:archive.unlink()


def load_source(name, data_root, download=True):
    if name == 'emnist_digits':
        ensure_emnist_digits(data_root,download)
        download=False
    cls = MNIST if name == 'mnist' else EMNIST
    extra = {} if name == 'mnist' else {'split': 'digits'}
    result = {}
    for split, train in [('train', True), ('test', False)]:
        ds = cls(root=str(data_root), train=train, download=download, **extra)
        result[split] = (ds.data.numpy().astype(np.uint8), ds.targets.numpy().astype(np.int64))
    return result


def hashes(images):
    return [hashlib.sha256(im.tobytes()).hexdigest() for im in images]


def archive_source(root, name, source):
    """Store loader-native pixels/labels in <16MB uncompressed shards, once."""
    folder = root / 'raw_sources' / name
    manifest = []
    for split, (images, labels) in source.items():
        if images.dtype != np.uint8 or images.ndim != 3 or images.shape[1:] != (28,28):
            raise ValueError(f'{name}/{split}: invalid source shape/dtype')
        if len(images) != len(labels) or set(labels.tolist()) != set(range(10)):
            raise ValueError(f'{name}/{split}: missing labels or size mismatch')
        for start in range(0, len(labels), 20000):
            path = folder / f'{split}_{start:06d}.npz'
            ids = np.arange(start, min(start+20000, len(labels)), dtype=np.int32)
            if path.exists():
                with np.load(path) as saved:
                    if not (np.array_equal(saved['images'], images[ids]) and
                            np.array_equal(saved['labels'], labels[ids]) and
                            np.array_equal(saved['source_ids'], ids)):
                        raise ValueError(f'Source archive changed: {path}')
            else:
                save_npz(path, images=images[ids], labels=labels[ids], source_ids=ids)
            manifest.append({'file': path.relative_to(root).as_posix(), 'sha256': digest(path),
                             'split': split, 'count': len(ids)})
    info = {'dataset': name, 'files': manifest,
            'orientation': 'transpose H/W after loading' if name == 'emnist_digits' else 'identity',
            'source': 'torchvision.datasets.' + ('MNIST' if name == 'mnist' else 'EMNIST(split=digits)'),
            'pixels': 'uint8, loader-native orientation; labels 0..9; ids are official split indices'}
    write_json(folder / 'manifest.json', info)
    return info


def balanced_subset(ids, labels, limit, seed):
    if limit is None or limit >= len(ids):
        return ids
    if limit < 100:
        raise ValueError('Smoke subsets need at least 100 sources.')
    rng = np.random.default_rng(seed)
    parts = [rng.permutation(ids[labels[ids] == y]) for y in range(10)]
    counts = [min(len(a), limit//10 + (y < limit%10)) for y,a in enumerate(parts)]
    result = np.concatenate([a[:n] for a,n in zip(parts,counts)])
    return np.sort(result)


def prepare_source(root, name, source, limit=None):
    manifest = archive_source(root, name, source)
    images, labels = source['train']
    test_images, test_labels = source['test']
    # The torchvision EMNIST IDX array uses transposed axes relative to upright digits.
    if name == 'emnist_digits':
        images = images.transpose(0,2,1).copy()
        test_images = test_images.transpose(0,2,1).copy()
    train_hash, test_hash = hashes(images), hashes(test_images)
    test_set = set(test_hash)
    seen = {}
    keep, exclusions, conflicts = [], [], []
    for idx,h in enumerate(train_hash):
        if h in test_set:
            exclusions.append(idx)
        elif h in seen:
            exclusions.append(idx)
            if labels[idx] != labels[seen[h]]:
                conflicts.append([seen[h], idx])
        else:
            keep.append(idx)
            seen[h] = idx
    ids = balanced_subset(np.array(keep, dtype=np.int64), labels, limit, stable_seed(name,'smoke_train'))
    test_ids = balanced_subset(np.arange(len(test_labels)), test_labels, limit, stable_seed(name,'smoke_test'))
    audit = {'dataset': name, 'official_train': len(labels), 'official_test': len(test_labels),
             'eligible_train': len(keep), 'selected_train': len(ids), 'selected_test': len(test_ids),
             'train_excluded_exact_duplicates': len(exclusions),
             'test_exact_duplicate_count': len(test_hash)-len(test_set),
             'conflicting_train_duplicate_labels': conflicts,
             'policy': 'Preserve official test; remove matching train pixels and keep first remaining train duplicate.',
             'near_duplicates': 'not certified', 'cross_dataset_overlap': 'not measured; datasets never pooled'}
    save_npz(root/'raw_sources'/name/'selection.npz', train_ids=ids, test_ids=test_ids,
             excluded_train_ids=np.array(exclusions,dtype=np.int32))
    write_json(root/'raw_sources'/name/'audit.json', audit)
    return {'images': torch.from_numpy(images), 'labels': labels, 'ids': ids,
            'test_images': torch.from_numpy(test_images), 'test_labels': test_labels, 'test_ids': test_ids,
            'manifest': manifest, 'audit': audit}


def apportion(counts, total):
    quota = np.asarray(counts,dtype=float) * total / sum(counts)
    n = np.floor(quota).astype(int)
    for i in np.argsort(-(quota-n), kind='stable')[:total-int(n.sum())]:
        n[i] += 1
    return n


def split_sources(ids, labels, seed):
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(ids[labels[ids] == y]) for y in range(10)]
    counts = np.array([len(a) for a in groups])
    total = round(.4*len(ids))
    ne = apportion(counts, total)
    nr = apportion(counts-ne, total)
    result = {'exposure': np.concatenate([a[:e] for a,e in zip(groups,ne)]),
              'recovery': np.concatenate([a[e:e+r] for a,e,r in zip(groups,ne,nr)]),
              'validation': np.concatenate([a[e+r:] for a,e,r in zip(groups,ne,nr)])}
    joined = np.concatenate(list(result.values()))
    if len(np.unique(joined)) != len(ids) or set(joined) != set(ids):
        raise ValueError('Source split overlaps or loses sources.')
    if any(set(labels[a]) != set(range(10)) for a in result.values()):
        raise ValueError('Every split must contain all 10 digits.')
    return result


def assign_colors(labels, p, permutation, seed):
    rng = np.random.default_rng(seed)
    out = np.empty(len(labels),dtype=np.uint8)
    for y in range(10):
        ids = rng.permutation(np.flatnonzero(labels == y))
        if not len(ids):
            raise ValueError('Missing digit in color allocation.')
        if np.isclose(p,.1):
            colors = np.tile(rng.permutation(10), (len(ids)+9)//10)[:len(ids)]
        else:
            aligned = round(len(ids)*p)
            rest = rng.permutation(np.array([c for c in range(10) if c != permutation[y]]))
            colors = np.concatenate([np.full(aligned,permutation[y]),
                                     np.tile(rest,(len(ids)-aligned+8)//9)[:len(ids)-aligned]])
        out[ids] = colors
    return out


def validate_colors(labels, colors, p, permutation):
    table = np.zeros((10,10),dtype=int)
    for y in range(10):
        row = np.bincount(colors[labels == y],minlength=10)
        n = row.sum()
        if np.isclose(p,.1):
            if row.max()-row.min()>1:
                raise ValueError('Neutral colors not balanced per digit.')
        else:
            other = np.delete(row,int(permutation[y]))
            if abs(row[permutation[y]]-n*p)>.500001 or other.max()-other.min()>1:
                raise ValueError('Biased color probabilities incorrect.')
        table[y] = row
    return table.tolist()


def palette(hue):
    h = ((np.arange(10)[None,:]*36 + hue[:,None]) % 360)/60
    x = 1-np.abs(h%2-1)
    r = np.where((h<1)|(h>=5),1,np.where(((h>=1)&(h<2))|((h>=4)&(h<5)),x,0))
    g = np.where((h>=1)&(h<3),1,np.where((h<1)|((h>=3)&(h<4)),x,0))
    b = np.where((h>=3)&(h<5),1,np.where(((h>=2)&(h<3))|(h>=5),x,0))
    return torch.from_numpy(np.stack([r,g,b],axis=-1).astype(np.float32))


def shared_design(root, name, source, seed, epochs, p_values):
    folder = root/'shared'/name/f'seed{seed}'
    pi = np.random.default_rng(stable_seed(seed,'permutation')).permutation(10)
    splits = split_sources(source['ids'],source['labels'],stable_seed(name,seed,'split'))
    hue = np.random.default_rng(stable_seed(name,seed,'train_hue')).uniform(-5,5,len(source['labels'])).astype(np.float32)
    thue = np.random.default_rng(stable_seed(name,seed,'test_hue')).uniform(-5,5,len(source['test_labels'])).astype(np.float32)
    arrays = {**splits,'permutation':pi,'hue':hue,'test_hue':thue,'test_ids':source['test_ids']}
    frequency = {}
    for p in p_values:
        key=f'exposure_p{round(p*100):03d}'
        a=assign_colors(source['labels'][splits['exposure']],p,pi,stable_seed(name,seed,'exposure_colors'))
        arrays[key]=a
        frequency[key]=validate_colors(source['labels'][splits['exposure']],a,p,pi)
    for e in range(1,epochs+1):
        key=f'recovery_colors_{e}'
        a=assign_colors(source['labels'][splits['recovery']],.1,pi,stable_seed(name,seed,'recovery_colors',e))
        arrays[key]=a
        frequency[key]=validate_colors(source['labels'][splits['recovery']],a,.1,pi)
    for phase,count in [('exposure',2),('recovery',epochs)]:
        for e in range(1,count+1):
            arrays[f'{phase}_order_{e}']=np.random.default_rng(stable_seed(name,seed,phase,'order',e)).permutation(len(splits[phase]))
    path=folder/'design.npz'
    if path.exists():
        with np.load(path) as old:
            if set(old.files)!=set(arrays) or any(not np.array_equal(old[k],a) for k,a in arrays.items()):
                raise ValueError('Shared design changed; choose a new output folder.')
    else:
        save_npz(path,**arrays)
    write_json(folder/'color_counts.json',frequency)
    write_json(folder/'manifest.json',{'dataset':name,'seed':seed,'design_sha256':digest(path),
               'permutation':pi.tolist(),'counts':{k:len(v) for k,v in splits.items()},
               'seed_scheme':'SHA256 of JSON tuple(dataset, seed, stage, purpose, epoch)',
               'exposure_epochs':2,'recovery_epochs':epochs})
    return {**arrays,'palette':palette(hue),'test_palette':palette(thue),'hash':digest(path)}
