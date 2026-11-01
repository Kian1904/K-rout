import { randomUUID } from 'node:crypto';
import { chmod, lstat, mkdir, open, readFile, rename, rm } from 'node:fs/promises';
import { basename, dirname, join } from 'node:path';

export type StoredProvider = {
  name: string;
  type: 'openai' | 'gemini';
  baseUrl: string;
  apiKey: string;
  models: string[];
};

interface ProviderConfigFile {
  version: 1;
  providers: StoredProvider[];
}

function validProvider(value: unknown): value is StoredProvider {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const provider = value as Record<string, unknown>;
  if (Object.keys(provider).some((key) => !['name', 'type', 'baseUrl', 'apiKey', 'models'].includes(key))) return false;
  if (typeof provider.name !== 'string' || !/^[A-Za-z0-9_-]{1,80}$/.test(provider.name)) return false;
  if (provider.type !== 'openai' && provider.type !== 'gemini') return false;
  if (typeof provider.baseUrl !== 'string' || typeof provider.apiKey !== 'string' || !provider.apiKey.trim()) return false;
  if (!Array.isArray(provider.models) || provider.models.length === 0) return false;
  if (!provider.models.every((model) => typeof model === 'string' && model.length > 0 && model.length <= 200 && model === model.trim())) return false;
  if (new Set(provider.models).size !== provider.models.length) return false;
  try {
    const url = new URL(provider.baseUrl);
    if (!['https:', 'http:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) return false;
  } catch {
    return false;
  }
  return true;
}

function validRegistry(value: unknown): value is StoredProvider[] {
  if (!Array.isArray(value) || !value.every(validProvider)) return false;
  const ids = new Set<string>();
  const models = new Set<string>();
  for (const provider of value) {
    if (ids.has(provider.name)) return false;
    ids.add(provider.name);
    for (const model of provider.models) {
      if (models.has(model)) return false;
      models.add(model);
    }
  }
  return true;
}

export class ProviderConfigStore {
  constructor(readonly filePath: string) {}

  async load(): Promise<StoredProvider[] | null> {
    const directory = dirname(this.filePath);
    await mkdir(directory, { recursive: true, mode: 0o700 });
    await chmod(directory, 0o700);
    try {
      const metadata = await lstat(this.filePath);
      if (!metadata.isFile() || metadata.isSymbolicLink()) {
        throw new Error('Provider configuration path is not a regular file; refusing to replace it.');
      }
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return null;
      if (error instanceof Error && error.message.includes('regular file')) throw error;
      throw new Error('Provider configuration file is unreadable; refusing to replace it.');
    }
    let content: string;
    try {
      content = await readFile(this.filePath, 'utf8');
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return null;
      throw new Error('Provider configuration file is unreadable; refusing to replace it.');
    }

    let parsed: unknown;
    try {
      parsed = JSON.parse(content);
    } catch {
      throw new Error('Provider configuration file is invalid JSON; refusing to replace it.');
    }
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      throw new Error('Provider configuration file has an unsupported format; refusing to replace it.');
    }
    const record = parsed as Record<string, unknown>;
    if (record.version !== 1 || Object.keys(record).some((key) => !['version', 'providers'].includes(key)) || !validRegistry(record.providers)) {
      throw new Error('Provider configuration file is invalid; refusing to replace it.');
    }
    await chmod(this.filePath, 0o600);
    return record.providers.map((provider) => ({ ...provider, models: [...provider.models] }));
  }

  async save(providers: StoredProvider[]): Promise<void> {
    if (!validRegistry(providers)) throw new Error('Invalid provider registry.');
    const directory = dirname(this.filePath);
    await mkdir(directory, { recursive: true, mode: 0o700 });
    await chmod(directory, 0o700);
    const temporaryPath = join(directory, `.${basename(this.filePath)}.${randomUUID()}.tmp`);
    const content = JSON.stringify({ version: 1, providers }, null, 2) + '\n';
    let handle: Awaited<ReturnType<typeof open>> | undefined;
    try {
      handle = await open(temporaryPath, 'wx', 0o600);
      await handle.writeFile(content, 'utf8');
      await handle.sync();
      await handle.close();
      handle = undefined;
      await chmod(temporaryPath, 0o600);
      await rename(temporaryPath, this.filePath);
    } finally {
      await handle?.close().catch(() => undefined);
      await rm(temporaryPath, { force: true }).catch(() => undefined);
    }
  }
}
