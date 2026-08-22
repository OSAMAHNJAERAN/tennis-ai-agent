import { readdir, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const distDir = path.resolve(scriptDir, "..", "dist");

async function listFiles(directory, root = directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const nested = await Promise.all(
    entries.map(async (entry) => {
      const fullPath = path.join(directory, entry.name);
      if (entry.isDirectory()) return listFiles(fullPath, root);
      const metadata = await stat(fullPath);
      return [{ file: path.relative(root, fullPath), bytes: metadata.size }];
    }),
  );
  return nested.flat();
}

const files = (await listFiles(distDir)).sort((a, b) => b.bytes - a.bytes);
const applicationBytes = files
  .filter(({ file }) => !file.startsWith(`demo${path.sep}`))
  .reduce((sum, file) => sum + file.bytes, 0);
const mediaBytes = files
  .filter(({ file }) => file.startsWith(`demo${path.sep}`))
  .reduce((sum, file) => sum + file.bytes, 0);

console.table(
  files.map(({ file, bytes }) => ({ file, kilobytes: Number((bytes / 1024).toFixed(1)) })),
);
console.log(`Application assets: ${(applicationBytes / 1024 / 1024).toFixed(2)} MiB`);
console.log(`Demo media: ${(mediaBytes / 1024 / 1024).toFixed(2)} MiB`);
