import { useMemo } from 'react';
import { PublicDemosSchema } from '../data/publicDemos';
import { useProductResource } from './useProductResource';

export function usePublicDemos() {
  const resource = useProductResource<unknown>('/api/demos', 10000);
  const parsed = useMemo(() => resource.data ? PublicDemosSchema.safeParse(resource.data) : null, [resource.data]);
  return {
    demos: parsed?.success ? parsed.data.demos : [],
    loading: !resource.data && !resource.error,
    error: resource.error || (parsed && !parsed.success ? 'The example catalog could not be verified. Please retry.' : ''),
    refresh: resource.refresh,
  };
}
