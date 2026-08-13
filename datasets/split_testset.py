import shutil
from pathlib import Path

import click

MODALITIES = [
    ("pano_hdr_1024", ".exr"),
    ("pano_ldr_1024", ".png"),
    ("pers_hdr_512", ".exr"),
    ("pers_ldr_512", ".png"),
    ("sg_npy", ".npy"),
    ("sg_jpg", ".jpg"),
    ("asg_npy", ".npy"),
    ("asg_png", ".png"),
]
SUBDIR_EXTS = dict(MODALITIES)


def read_split_list(txt_path):
    items = set()
    for line in Path(txt_path).read_text().splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            items.add(item)
    return items


def parse_stem(stem):
    parts = stem.rsplit("_", 2)
    if len(parts) != 3:
        return None, f"invalid sample name: {stem}"

    base, folder, rotate = parts
    if not base or not folder or len(rotate) != 2 or not rotate.isdigit():
        return None, f"invalid sample name: {stem}"
    return (base, folder, rotate), None


def collect_subdir(data_root, subdir):
    root = data_root / subdir
    expected_ext = SUBDIR_EXTS[subdir]
    if not root.is_dir():
        raise FileNotFoundError(f"missing source folder: {root}")

    samples = {}
    errors = []
    for path in sorted(root.iterdir()):
        if not path.is_file():
            errors.append(f"unexpected non-file entry: {path}")
            continue
        if path.suffix.lower() != expected_ext:
            errors.append(
                f"unexpected extension in {subdir}: {path.name}, expected {expected_ext}"
            )
            continue
        parsed, error = parse_stem(path.stem)
        if error:
            errors.append(error)
            continue
        if path.stem in samples:
            errors.append(f"duplicate sample stem in {subdir}: {path.stem}")
            continue
        samples[path.stem] = {
            "path": path,
            "base": parsed[0],
        }
    return samples, errors


def collect_samples(data_root):
    by_subdir = {}
    errors = []
    for subdir in SUBDIR_EXTS:
        root = data_root / subdir
        if not root.is_dir():
            errors.append(f"missing source folder: {root}")
            continue
        by_subdir[subdir], subdir_errors = collect_subdir(data_root, subdir)
        errors.extend(subdir_errors)

    stems = (
        sorted(set().union(*[set(items) for items in by_subdir.values()]))
        if by_subdir
        else []
    )
    samples = {}
    for stem in stems:
        missing = [
            subdir for subdir in SUBDIR_EXTS if stem not in by_subdir.get(subdir, {})
        ]
        if missing:
            errors.append(f"missing modalities for {stem}: {', '.join(missing)}")
            continue
        samples[stem] = {
            subdir: by_subdir[subdir][stem]["path"] for subdir in SUBDIR_EXTS
        }
    return samples, errors


def target_split(base, indoor_set, outdoor_set):
    if base in indoor_set:
        return "testset/indoor"
    if base in outdoor_set:
        return "testset/outdoor"
    return "trainset"


def build_moves(data_root, samples, indoor_set, outdoor_set):
    moves = {}
    errors = []
    for stem, files in samples.items():
        parsed, error = parse_stem(stem)
        if error:
            errors.append(error)
            continue
        base = parsed[0]
        split = target_split(base, indoor_set, outdoor_set)
        stem_moves = []
        for subdir, src in files.items():
            dst = data_root / split / subdir / src.name
            if dst.exists():
                errors.append(f"target exists: {dst}")
            stem_moves.append((src, dst, split))
        moves[stem] = stem_moves
    return moves, errors


def print_summary(samples, moves, dry_run):
    sample_counts = {}
    file_counts_by_split = {}
    file_counts = {}
    flat_moves = [item for stem_moves in moves.values() for item in stem_moves]
    for stem in samples:
        split = moves[stem][0][2]
        sample_counts[split] = sample_counts.get(split, 0) + 1
    for _, _, split in flat_moves:
        file_counts_by_split[split] = file_counts_by_split.get(split, 0) + 1
    for src, _, _ in flat_moves:
        file_counts[src.parent.name] = file_counts.get(src.parent.name, 0) + 1

    action = "dry-run" if dry_run else "move"
    print(f"{action} samples: {len(samples)}")
    print(f"{action} files: {len(flat_moves)}")
    for split in sorted(sample_counts):
        print(
            f"{split}: {sample_counts[split]} samples, {file_counts_by_split[split]} files"
        )
    for subdir in SUBDIR_EXTS:
        print(f"{subdir}: {file_counts.get(subdir, 0)}")


def validate_splits(indoor_set, outdoor_set):
    overlap = sorted(indoor_set & outdoor_set)
    if overlap:
        preview = ", ".join(overlap[:20])
        suffix = " ..." if len(overlap) > 20 else ""
        raise ValueError(f"split list overlap: {preview}{suffix}")


def fail_if_errors(errors):
    if not errors:
        return
    preview = "\n".join(errors[:50])
    suffix = f"\n... and {len(errors) - 50} more errors" if len(errors) > 50 else ""
    raise ValueError(preview + suffix)


def create_destination_dirs(data_root, moves):
    splits = {split for stem_moves in moves.values() for _, _, split in stem_moves}
    for split in splits:
        for subdir in SUBDIR_EXTS:
            Path(data_root, split, subdir).mkdir(parents=True, exist_ok=True)


def move_files(data_root, moves):
    create_destination_dirs(data_root, moves)
    for stem_moves in moves.values():
        for src, dst, _ in stem_moves:
            shutil.move(src.as_posix(), dst.as_posix())


@click.command()
@click.option("--dataset-root", required=True)
@click.option("--indoor-list", default=None)
@click.option("--outdoor-list", default=None)
@click.option("--dry_run", "--dry-run", is_flag=True)
def main(dataset_root, indoor_list, outdoor_list, dry_run):
    data_root = Path(dataset_root).expanduser()
    indoor_set = read_split_list(indoor_list) if indoor_list else set()
    outdoor_set = read_split_list(outdoor_list) if outdoor_list else set()

    validate_splits(indoor_set, outdoor_set)
    samples, collect_errors = collect_samples(data_root)
    moves, move_errors = build_moves(data_root, samples, indoor_set, outdoor_set)
    fail_if_errors(collect_errors + move_errors)

    print_summary(samples, moves, dry_run)
    if dry_run:
        return
    move_files(data_root, moves)


if __name__ == "__main__":
    main()
