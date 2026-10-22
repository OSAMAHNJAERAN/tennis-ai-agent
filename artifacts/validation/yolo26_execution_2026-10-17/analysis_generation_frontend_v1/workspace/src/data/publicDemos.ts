import { z } from 'zod';

export const MediaAttributionSchema = z.object({
  author: z.string().min(1),
  sourcePage: z.string(),
  license: z.string().min(1),
  licenseUrl: z.string(),
  derivativeLicense: z.string().nullable().optional(),
  modifications: z.union([z.string(), z.array(z.string())]),
});

const ExcerptSchema = z.object({
  startSeconds: z.number().nonnegative(), endSeconds: z.number().nonnegative(),
  startFrame: z.number().int().nonnegative(), endFrameExclusive: z.number().int().nonnegative(),
  sourceTimeBase: z.string(), firstSourcePts: z.string(), lastSourcePts: z.string(),
}).passthrough().refine(excerpt => excerpt.endSeconds > excerpt.startSeconds && excerpt.endFrameExclusive > excerpt.startFrame, { message: 'The source excerpt interval is invalid.' });

export const MediaSourceSchema = MediaAttributionSchema.extend({
  id: z.string().optional(),
  title: z.string().optional(),
  sourceSha256: z.string().regex(/^[a-f0-9]{64}$/).optional(),
  inputSha256: z.string().regex(/^[a-f0-9]{64}$/).optional(),
  excerpt: ExcerptSchema.nullable().optional(),
});

const PublicDemoSchema = z.object({
  id: z.string().min(1), title: z.string().min(1),
  analysisId: z.string().regex(/^match_[a-f0-9]{32}$/),
  previewUrl: z.string(),
  attribution: MediaAttributionSchema,
  excerpt: ExcerptSchema.optional(),
  qualificationEvidence: z.literal(false),
}).refine(demo => {
  const path = `/api/analyses/${demo.analysisId}/media/annotated`;
  return demo.previewUrl === path || (demo.previewUrl.startsWith(`${path}?`) &&
    (/^\?v=[a-f0-9]{1,16}-[a-f0-9]{1,16}$/.test(demo.previewUrl.slice(path.length)) ||
      /^\?generation=[a-f0-9]{64}$/.test(demo.previewUrl.slice(path.length))));
}, { message: 'The demo preview does not match its analysis.' });

export const PublicDemosSchema = z.object({ schemaVersion: z.literal(1), demos: z.array(PublicDemoSchema) }).refine(
  catalog => new Set(catalog.demos.map(demo => demo.id)).size === catalog.demos.length && new Set(catalog.demos.map(demo => demo.analysisId)).size === catalog.demos.length,
  { message: 'The demo catalog contains repeated analysis identities.' },
);

export type MediaAttribution = z.infer<typeof MediaAttributionSchema>;
export type MediaSource = z.infer<typeof MediaSourceSchema>;
export type PublicDemo = z.infer<typeof PublicDemoSchema>;

export function safeCreditUrl(value: string): string | null {
  try {
    const url = new URL(value);
    return ['https:', 'http:'].includes(url.protocol) && !url.username && !url.password ? url.href : null;
  } catch { return null; }
}
