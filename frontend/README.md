# Local research dashboard

This is a read-only React dashboard for the local HFT research API. It displays
real dataset history and recorded reports. It has no credentials, broker calls,
order controls, paid services, or sample-data fallback.

From this directory:

```sh
npm ci
npm run build
npm test
```

From the repository root, run `python -m hft dashboard` to serve the production
bundle and API on `http://127.0.0.1:8765`.

For development, keep the Python dashboard running and use `npm run dev` in this
directory. Vite proxies `/api` to the loopback Python server on port 8765. The
production bundle goes to `frontend/dist` and is served by Python; the Vite
preview command alone does not provide the API.

The dashboard reads `/api/dashboard` immediately, every 30 seconds, and on
Refresh. Reads time out after 15 seconds. Failed reads keep the last snapshot
visible and label it stale. Empty workspaces show unavailable counts and empty
panels. Historical prices include development sessions only and represent the
last reported trade per session, not official closes or strategy returns.

Design tokens and component rules are documented in the root `DESIGN.md`.
Frontend tests cover exchange-date formatting, unavailable numerical evidence,
market ranges, and CSS token drift. Browser verification covers the rendered
views and API recovery behavior.
