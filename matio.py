import os
"""Minimal MATLAB v5 .mat reader (no scipy dependency)."""
import struct
import zlib
import numpy as np

DTYPES = {
    1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4',
    7: 'f4', 9: 'f8', 12: 'i8', 13: 'u8', 16: 'u1', 17: 'u2', 18: 'u4',
}
MI_INT8, MI_UINT8, MI_INT16, MI_UINT16 = 1, 2, 3, 4
MI_INT32, MI_UINT32, MI_SINGLE, MI_DOUBLE = 5, 6, 7, 9
MI_INT64, MI_UINT64, MI_MATRIX, MI_COMPRESSED = 12, 13, 14, 15
MI_UTF8, MI_UTF16, MI_UTF32 = 16, 17, 18
MX_CLASSES = {
    1: 'cell', 2: 'struct', 3: 'object', 4: 'char', 5: 'sparse',
    6: 'double', 7: 'single', 8: 'int8', 9: 'uint8', 10: 'int16',
    11: 'uint16', 12: 'int32', 13: 'uint32', 14: 'int64', 15: 'uint64',
    16: 'function',
}


class _Reader:
    def __init__(self, buf):
        self.b = buf
        self.p = 0

    def take(self, n):
        v = self.b[self.p:self.p + n]
        self.p += n
        return v

    def u32(self):
        return struct.unpack('<I', self.take(4))[0]

    def at_end(self):
        return self.p >= len(self.b)


def _read_tag(r):
    """Returns (dtype, nbytes, small_data_or_None)."""
    word = r.take(8)
    a, b = struct.unpack('<II', word)
    if (a >> 16) & 0xFFFF:
        dt = a & 0xFFFF
        n = a >> 16
        return dt, n, word[4:8]
    return a, b, None


def _read_payload(r):
    dt, n, small = _read_tag(r)
    if small is not None:
        return dt, small[:n]
    data = r.take(n)
    pad = (8 - (n % 8)) % 8
    r.p += pad
    return dt, data


def _to_array(dt, data):
    if dt in DTYPES:
        n = DTYPES[dt]
        return np.frombuffer(data, dtype=np.dtype('<' + n))
    if dt in (MI_UTF8, MI_UTF16, MI_UTF32):
        return np.frombuffer(data, dtype=np.uint8)
    return np.frombuffer(data, dtype=np.uint8)


def _read_matrix(r):
    dt, data = _read_payload(r)
    assert dt in (MI_UINT32, MI_INT32), f'expected array flags, got {dt}'
    flags = np.frombuffer(data, dtype='<u4')
    cls = flags[0] & 0xFF
    dt, data = _read_payload(r)
    dims = tuple(int(x) for x in np.frombuffer(data, dtype='<i4'))
    dt, data = _read_payload(r)
    name = bytes(data).decode('latin1') if dt != MI_INT8 else bytes(data).decode('latin1')

    if cls == 1:  # cell array
        n = int(np.prod(dims)) if dims else 0
        flat = []
        for _ in range(n):
            v = _read_data_element(r)
            if isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], str):
                v = v[1]
            flat.append(v)
        out = np.empty(n, dtype=object)
        for i, v in enumerate(flat):
            out[i] = v
        return name, out.reshape(dims, order='F') if dims else out

    if cls == 2:  # struct
        n = int(np.prod(dims)) if dims else 1
        nfields = int(_read_payload(r)[1].view('<i4')[0]) if False else None
        dtf, df = _read_payload(r)
        # field name length is stored as int32 payload
        flen = int(np.frombuffer(df[:4], dtype='<i4')[0])
        dtf, df = _read_payload(r)
        fnames = []
        raw = bytes(df)
        for i in range(len(raw) // flen):
            fnames.append(raw[i * flen:(i + 1) * flen].split(b'\x00')[0].decode('latin1'))
        objs = []
        for _ in range(n):
            for fn in fnames:
                objs.append((fn, _read_data_element(r)))
        out = {}
        for i in range(n):
            d = {}
            for j, fn in enumerate(fnames):
                d[fn] = objs[i * len(fnames) + j][1]
            out[i if n > 1 else 0] = d
        return name, (out if n > 1 else out[0])

    if cls == 4:  # char
        dt, data = _read_payload(r)
        s = bytes(data).decode('latin1')
        return name, s

    if cls == 5:  # sparse
        ir = _read_data_element(r)
        jc = _read_data_element(r)
        pr = _read_data_element(r)
        return name, {'ir': ir, 'jc': jc, 'pr': pr, 'dims': dims}

    # numeric / logical
    dt, data = _read_payload(r)
    arr = _to_array(dt, data)
    if cls == 6:
        arr = arr.astype(np.float64)
    arr = arr.reshape(dims, order='F') if dims else arr
    return name, arr


def _read_data_element(r):
    dt, n, small = _read_tag(r)
    if small is not None:
        return _to_array(dt, small[:n])
    if dt == MI_COMPRESSED:
        # compressed elements are NOT padded in the MATLAB v5 format
        data = r.take(n)
        inner = _Reader(zlib.decompress(data))
        return _read_data_element(inner)
    if dt == MI_MATRIX:
        data = r.take(n)
        pad = (8 - (n % 8)) % 8
        r.p += pad
        return _read_matrix(_Reader(data))
    data = r.take(n)
    pad = (8 - (n % 8)) % 8
    r.p += pad
    return _to_array(dt, data)


def loadmat(path):
    with open(path, 'rb') as f:
        buf = f.read()
    r = _Reader(buf)
    r.p = 128
    out = {}
    while not r.at_end():
        if len(buf) - r.p < 8:
            break
        res = _read_data_element(r)
        if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], str):
            name, val = res
            if name:
                out[name] = val
    return out


if __name__ == '__main__':
    import sys
    d = loadmat(sys.argv[1])
    print('keys:', list(d.keys()))
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.dtype != object:
            print(f'  {k}: ndarray shape={v.shape} dtype={v.dtype}')
            if v.size and v.dtype.kind in 'fiu':
                print(f'      min={np.nanmin(v):.6g} max={np.nanmax(v):.6g} nan={int(np.isnan(v).sum()) if v.dtype.kind=="f" else 0}')
                print(f'      head={np.ravel(v)[:5]}')
        elif isinstance(v, np.ndarray):
            print(f'  {k}: object array shape={v.shape}')
        elif isinstance(v, dict):
            print(f'  {k}: dict keys={list(v.keys())[:10]}')
        elif isinstance(v, str):
            print(f'  {k}: str len={len(v)} preview={v[:120]!r}')
        else:
            print(f'  {k}: {type(v)} {v}')
