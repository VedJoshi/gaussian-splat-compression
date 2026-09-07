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
