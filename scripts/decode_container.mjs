// Reference decoder for the .splatc container. Decode and compare only; the
// viewer lands in milestone 6. Kept dependency-free so it runs identically
// under node and in a browser module.

import { readFile, writeFile } from 'node:fs/promises';

const MAGIC = 'SPLATC';
const VERSION_MAJOR = 1;
const PREFIX_LEN = 24;

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i += 1) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

async function inflate(bytes) {
  // "deflate" and not "deflate-raw": the encoder emits a zlib-wrapped stream.
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

function paeth(a, b, c) {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  if (pa <= pb && pa <= pc) return a;
  return pb <= pc ? b : c;
}

// The encoder writes 8-bit greyscale, so one byte per pixel and one filter byte
// per row. Decoding here rather than through createImageBitmap keeps this file
// runnable under node, where there is no canvas.
async function decodePng(bytes) {
  for (let i = 0; i < PNG_SIGNATURE.length; i += 1) {
    if (bytes[i] !== PNG_SIGNATURE[i]) throw new Error('block is not a png');
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  let width = 0;
  let height = 0;
  const idat = [];
  for (let offset = 8; offset + 8 <= bytes.length; ) {
    const length = view.getUint32(offset, false);
    const type = String.fromCharCode(...bytes.subarray(offset + 4, offset + 8));
    const body = bytes.subarray(offset + 8, offset + 8 + length);
    if (type === 'IHDR') {
      width = view.getUint32(offset + 8, false);
      height = view.getUint32(offset + 12, false);
      const depth = body[8];
      const colour = body[9];
      const interlace = body[12];
      if (depth !== 8 || colour !== 0 || interlace !== 0) {
        throw new Error(`png block is depth ${depth} colour ${colour} interlace ${interlace}, expected 8/0/0`);
      }
    } else if (type === 'IDAT') {
      idat.push(body);
    } else if (type === 'IEND') {
      break;
    }
    offset += 12 + length;
  }
  if (!width || !height) throw new Error('png block has no IHDR');

  let total = 0;
  for (const part of idat) total += part.length;
  const joined = new Uint8Array(total);
  let cursor = 0;
  for (const part of idat) {
    joined.set(part, cursor);
    cursor += part.length;
  }

  const filtered = await inflate(joined);
  if (filtered.length < height * (width + 1)) throw new Error('png block decoded short');

  const pixels = new Uint8Array(width * height);
  for (let y = 0; y < height; y += 1) {
    const filter = filtered[y * (width + 1)];
    const source = y * (width + 1) + 1;
    const target = y * width;
    const above = target - width;
    for (let x = 0; x < width; x += 1) {
      const raw = filtered[source + x];
      const a = x > 0 ? pixels[target + x - 1] : 0;
      const b = y > 0 ? pixels[above + x] : 0;
      const c = x > 0 && y > 0 ? pixels[above + x - 1] : 0;
      let value;
      if (filter === 0) value = raw;
      else if (filter === 1) value = raw + a;
      else if (filter === 2) value = raw + b;
      else if (filter === 3) value = raw + ((a + b) >> 1);
      else if (filter === 4) value = raw + paeth(a, b, c);
      else throw new Error(`unknown png row filter ${filter}`);
      pixels[target + x] = value & 0xff;
    }
  }
  return pixels;
}

export async function decodeContainer(buffer) {
  const bytes = new Uint8Array(buffer);
  if (bytes.length < PREFIX_LEN) throw new Error('file is shorter than the header');

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const magic = new TextDecoder().decode(bytes.subarray(0, 6));
  if (magic !== MAGIC) throw new Error(`not a splatc container: magic is ${magic}`);

  const major = view.getUint8(6);
  if (major !== VERSION_MAJOR) throw new Error(`unsupported major version ${major}`);

  const jsonLen = view.getUint32(8, true);
  const jsonCrc = view.getUint32(12, true);
  const payloadLen = Number(view.getBigUint64(16, true));

  const descriptorBytes = bytes.subarray(PREFIX_LEN, PREFIX_LEN + jsonLen);
  if (crc32(descriptorBytes) !== jsonCrc) throw new Error('descriptor checksum mismatch');
  const descriptor = JSON.parse(new TextDecoder().decode(descriptorBytes));

  const payload = bytes.subarray(PREFIX_LEN + jsonLen, PREFIX_LEN + jsonLen + payloadLen);
  if (payload.length !== payloadLen) throw new Error('payload is truncated');

  const blocks = {};
  for (const block of descriptor.blocks) {
    const stored = payload.subarray(block.offset, block.offset + block.length);
    if (crc32(stored) !== block.crc32) {
      throw new Error(`block '${block.name}' failed its checksum`);
    }

    let raw;
    if (block.codec === 'raw') raw = stored;
    else if (block.codec === 'deflate') raw = await inflate(stored);
    else if (block.codec === 'png') raw = await decodePng(stored);
    else throw new Error(`unknown block codec '${block.codec}'`);

    if (raw.length < block.raw_length) {
      throw new Error(`block '${block.name}' decoded short`);
    }
    raw = raw.subarray(0, block.raw_length);

    // Copy into a fresh buffer: a subarray's byteOffset is not guaranteed to be
    // even, and Uint16Array throws on a misaligned offset.
    const aligned = raw.slice();
    const values =
      block.dtype === 'uint8'
        ? aligned
        : new Uint16Array(aligned.buffer, aligned.byteOffset, aligned.length / 2);

    blocks[block.name] = {
      dtype: block.dtype,
      stored_shape: block.stored_shape,
      values: Array.from(values),
    };
  }

  return {
    count: descriptor.count,
    sh_degree: descriptor.sh_degree,
    order: descriptor.order,
    blocks,
  };
}

const [input, output] = process.argv.slice(2);
if (!input || !output) {
  console.error('usage: node decode_container.mjs <in.splatc> <out.json>');
  process.exit(2);
}

try {
  const file = await readFile(input);
  const decoded = await decodeContainer(
    file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength),
  );
  await writeFile(output, JSON.stringify(decoded), 'utf-8');
} catch (error) {
  console.error(String(error && error.message ? error.message : error));
  process.exit(1);
}
