import { useCallback, useEffect, useRef, useState } from 'react';

export function useDashboard() {
  const [request, setRequest] = useState(0);
  const [state, setState] = useState({ data: null, loading: true, error: null });
  const pendingRef = useRef(true);
  const refresh = useCallback(() => setRequest((value) => value + 1), []);

  useEffect(() => {
    const interval = window.setInterval(() => {
      if (!pendingRef.current) setRequest((value) => value + 1);
    }, 30000);
    return () => window.clearInterval(interval);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    let ignore = false;
    const timeout = window.setTimeout(() => controller.abort('timeout'), 15000);
    pendingRef.current = true;
    setState((previous) => ({ ...previous, loading: true }));

    async function load() {
      try {
        const response = await fetch('/api/dashboard', {
          signal: controller.signal,
          cache: 'no-store',
          headers: { Accept: 'application/json' },
        });
        if (!response.ok) throw new Error(`The local dashboard returned ${response.status}.`);
        const data = await response.json();
        if (
          !data ||
          typeof data !== 'object' ||
          Array.isArray(data) ||
          !data.training ||
          typeof data.training !== 'object' ||
          (data.dataset !== null &&
            data.dataset !== undefined &&
            (typeof data.dataset !== 'object' || Array.isArray(data.dataset)))
        ) {
          throw new Error('The local dashboard response could not be read.');
        }
        if (!ignore)
          setState({ data: { ...data, dataset: data.dataset || {} }, loading: false, error: null });
      } catch (error) {
        if (!ignore) {
          const message = controller.signal.aborted
            ? 'The local dashboard did not respond in time.'
            : error instanceof TypeError || error instanceof SyntaxError
              ? 'The local dashboard is unavailable. Check that its server is running.'
              : error.message;
          setState((previous) => ({ ...previous, loading: false, error: message }));
        }
      } finally {
        window.clearTimeout(timeout);
        if (!ignore) pendingRef.current = false;
      }
    }
    load();
    return () => {
      ignore = true;
      window.clearTimeout(timeout);
      controller.abort();
    };
  }, [request]);

  return { ...state, refresh };
}
