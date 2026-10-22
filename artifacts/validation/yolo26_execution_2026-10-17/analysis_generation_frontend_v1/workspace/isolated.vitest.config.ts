import { fileURLToPath } from 'node:url';
export default { cacheDir: fileURLToPath(new URL('./test-cache', import.meta.url)),
  test: { environment: 'jsdom', globals: true, pool: 'forks', maxWorkers: 1,
    setupFiles: './src/test/setup.ts', css: false } };
