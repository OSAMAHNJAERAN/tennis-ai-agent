import { z } from 'zod';
import archive from './featured/matches.json';

const playerSchema = z.object({ id: z.string(), name: z.string(), country: z.string(), portrait: z.url(), profileUrl: z.url(), photoCredit: z.string() });
const matchSchema = z.object({
  id: z.string(), tournament: z.string(), edition: z.number(), round: z.string(), surface: z.enum(['Grass', 'Hard', 'Clay']), date: z.iso.date(), venue: z.string(), durationMinutes: z.number().positive(),
  players: z.tuple([z.string(), z.string()]), winner: z.string(), sets: z.array(z.tuple([z.number().int().nonnegative(), z.number().int().nonnegative()])).min(3).max(5),
  tiebreaks: z.record(z.string(), z.tuple([z.number(), z.number()])),
  video: z.object({ provider: z.string(), youtubeId: z.string().regex(/^[\w-]{11}$/), url: z.url(), kind: z.literal('official_full_replay') }),
  resultSource: z.object({ name: z.string(), url: z.url() }), review: z.object({ title: z.string(), text: z.string(), basis: z.string() }),
});
export type FeaturedPlayer = z.infer<typeof playerSchema>;
export type FeaturedMatch = z.infer<typeof matchSchema>;
export const featuredCatalog = z.object({ schemaVersion: z.literal(1), reviewedAt: z.iso.date(), players: z.record(z.string(), playerSchema), matches: z.array(matchSchema) }).parse(archive);
export const featuredMatches = featuredCatalog.matches;
export const getFeaturedMatch = (id: string | null | undefined) => featuredMatches.find(match => match.id === id);
export function playersFor(match: FeaturedMatch): [FeaturedPlayer, FeaturedPlayer] { return [featuredCatalog.players[match.players[0]]!, featuredCatalog.players[match.players[1]]!]; }
export const matchDuration = (minutes: number) => `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, '0')}m`;
export const setsWon = (match: FeaturedMatch, player: number) => match.sets.filter(set => set[player]! > set[1-player]!).length;
export const gamesWon = (match: FeaturedMatch, player: number) => match.sets.reduce((total, set) => total + set[player]!, 0);
export const featuredTitle = (match: FeaturedMatch) => playersFor(match).map(player => player.name).join(' vs ');
