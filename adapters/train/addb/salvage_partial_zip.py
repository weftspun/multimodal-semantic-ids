# SPDX-License-Identifier: MIT
# Salvage complete entries from a front-truncated (still-downloading) zip: a zip's
# central directory is at the end, but each entry has a local header + data, so any
# file whose data fully precedes the truncation point can be extracted by scanning
# local headers.  Pulls complete B3D files from the partial AddBiomechanics archive while
# the ~389 GB download is still in progress.
#   python salvage_partial_zip.py <partial.zip> <out_dir> [--only-b3d] [--max N]
#   python salvage_partial_zip.py <partial.zip> --list   # inventory complete B3D entries, no extract
import sys, os, struct, zlib

LFH = b"PK\x03\x04"


def list_b3d(zip_path):  # walk local headers, report every complete .b3d entry (name, MB), no extract
    size = os.path.getsize(zip_path)
    off, found = 0, []
    with open(zip_path, "rb") as f:
        while off < size - 30:
            f.seek(off)
            if f.read(4) != LFH:
                break
            hdr = f.read(26)
            if len(hdr) < 26:
                break
            (ver, flags, method, mtime, mdate, crc, csize, usize, nlen, elen) = struct.unpack("<HHHHHIIIHH", hdr)
            name = f.read(nlen).decode("utf-8", "replace")
            f.read(elen)
            data_end = f.tell() + csize
            if flags & 0x08 or data_end > size:
                break
            if name.lower().endswith(".b3d") and not name.endswith("/"):
                found.append((name, usize))
            off = data_end
    for name, usize in found:
        print(f"{usize / 1e6:8.1f} MB  {name}")
    print(f"# {len(found)} complete B3D subjects in {size / 1e9:.1f} GB scanned")


def extract_headers(zip_path, names, out_dir, header_mb):
    # Decompress only the first header_mb MB of each named entry — enough for nimble to read the
    # B3D demographic header (sex/height/mass) for phenotype balancing, without the full
    # multi-hundred-MB file.  nimble reads demographics once the proto header is present.
    want, need = set(names), header_mb * 1024 * 1024
    size, off, got = os.path.getsize(zip_path), 0, 0
    os.makedirs(out_dir, exist_ok=True)
    with open(zip_path, "rb") as f:
        while off < size - 30:
            f.seek(off)
            if f.read(4) != LFH:
                break
            hdr = f.read(26)
            if len(hdr) < 26:
                break
            (ver, flags, method, mtime, mdate, crc, csize, usize, nlen, elen) = struct.unpack("<HHHHHIIIHH", hdr)
            name = f.read(nlen).decode("utf-8", "replace")
            f.read(elen)
            data_off = f.tell()
            data_end = data_off + csize
            if flags & 0x08 or data_end > size:
                break
            if name in want:
                f.seek(data_off)
                if method == 0:
                    raw = f.read(min(need, csize))
                else:
                    d, raw, remaining = zlib.decompressobj(-15), b"", csize
                    while remaining > 0 and len(raw) < need:
                        chunk = f.read(min(1 << 20, remaining))
                        remaining -= len(chunk)
                        if not chunk:
                            break
                        raw += d.decompress(chunk)
                    raw = raw[:need]
                outp = os.path.join(out_dir, name)
                os.makedirs(os.path.dirname(outp), exist_ok=True)
                with open(outp, "wb") as o:
                    o.write(raw)
                got += 1
                print(f"header {got}/{len(want)}: {name} ({len(raw)} B)")
            off = data_end
    print(f"extracted {got} headers")


def salvage(zip_path, out_dir, only_b3d=True, max_files=1):
    size = os.path.getsize(zip_path)
    os.makedirs(out_dir, exist_ok=True)
    got = 0
    with open(zip_path, "rb") as f:
        off = 0
        while off < size - 30:
            f.seek(off)
            sig = f.read(4)
            if sig != LFH:
                # not at a header (shouldn't happen if sizes are in headers) — bail
                print(f"no local header at {off}; stopping (likely data descriptors).")
                break
            hdr = f.read(26)
            if len(hdr) < 26:
                break
            (ver, flags, method, mtime, mdate, crc, csize, usize, nlen, elen) = struct.unpack(
                "<HHHHHIIIHH", hdr)
            name = f.read(nlen).decode("utf-8", "replace")
            f.read(elen)
            data_off = f.tell()
            if flags & 0x08:  # sizes in a trailing data descriptor → can't locate end
                print(f"entry '{name}' uses a data descriptor (streamed) — cannot salvage by scan.")
                break
            data_end = data_off + csize
            is_b3d = name.lower().endswith(".b3d")
            complete = data_end <= size
            if complete and (is_b3d or not only_b3d) and not name.endswith("/"):
                data = f.read(csize)
                if method == 0:
                    raw = data
                elif method == 8:
                    raw = zlib.decompress(data, -15)
                else:
                    print(f"skip '{name}': unsupported method {method}")
                    off = data_end
                    continue
                outp = os.path.join(out_dir, name)  # keep the path to avoid name collisions
                os.makedirs(os.path.dirname(outp), exist_ok=True)
                with open(outp, "wb") as o:
                    o.write(raw)
                print(f"extracted {name}  ({len(raw)} bytes) -> {outp}")
                got += 1
                if got >= max_files:
                    print(f"done: {got} file(s).")
                    return
            elif not complete:
                print(f"entry '{name}' incomplete (needs {data_end} bytes, have {size}); stopping.")
                break
            off = data_end
    print(f"finished scan; salvaged {got} file(s).")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--list" in args:
        list_b3d(next(a for a in args if not a.startswith("--")))
        sys.exit(0)
    if "--names" in args:  # --names <namelist> --header-mb N <zip> <out_dir>
        names = [ln.strip() for ln in open(args[args.index("--names") + 1]) if ln.strip()]
        hmb = int(args[args.index("--header-mb") + 1]) if "--header-mb" in args else 32
        skip = {args[args.index("--names") + 1]} | ({args[args.index("--header-mb") + 1]} if "--header-mb" in args else set())
        pos = [a for a in args if not a.startswith("--") and a not in skip]
        extract_headers(pos[0], names, pos[1], hmb)
        sys.exit(0)
    only_b3d = "--only-b3d" in args
    mx = 1
    if "--max" in args:
        mx = int(args[args.index("--max") + 1])
    pos = [a for a in args if not a.startswith("--") and a != str(mx)]
    salvage(pos[0], pos[1], only_b3d=only_b3d, max_files=mx)
