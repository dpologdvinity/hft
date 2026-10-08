---
version: alpha
name: "Intraday Trading Engine Dashboard"
description: "A calm local research desk for inspecting real market history, training evidence, and paper eligibility."
colors:
  primary: "#365df3"
  primary-hover: "#2447d9"
  background: "#f4f7fc"
  surface: "#ffffff"
  text: "#101d31"
  muted: "#5a6d8e"
  border: "#dce5f1"
  rail: "#19283b"
  rail-selected: "#293f60"
  rail-text: "#d7e5f8"
  success: "#008b68"
  warning: "#a65b00"
  danger: "#b42e43"
  track: "#e8edf5"
  scrollbar: "#9baac0"
  scrollbar-hover: "#6e819c"
  scrollbar-active: "#4f647f"
typography:
  sans:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    fontSize: "14px"
    lineHeight: "1.5"
rounded:
  panel: "10px"
  control: "6px"
spacing:
  page: "24px"
  panel: "20px"
  gap: "16px"
components:
  panel:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text}"
    rounded: "{rounded.panel}"
    padding: "{spacing.panel}"
  button:
    height: "38px"
    backgroundColor: "{colors.primary}"
    textColor: "{colors.surface}"
    rounded: "{rounded.control}"
  navigation:
    height: "49px"
    backgroundColor: "{colors.rail}"
    textColor: "{colors.rail-text}"
---

# Trading Dashboard Design System

## Overview

The reference is a quiet analyst's research desk: a midnight blue navigation rail,
cool off-white workspace, open summary band, and readable chart and report panels.

This is a product dashboard for the local stock-research operator. Its job is to
inspect actual development prices and recorded evidence. It uses en-US and
America/New_York for exchange dates, timestamps, numbers, and USD. The current
brief establishes a local US-stock workflow; no Japan-market scope is inferred.
Desktop analysis is primary; narrow mobile widths retain every view and report.

The memorable move is the contrast between the dark rail and the light research
workspace. Data readability takes priority over decoration. Avoid trading-terminal
noise, fake live tickers, marketing badges, glows, invented profits, and dashboards
that replace comparison tables with card grids.

Runtime token ownership is Model B: [frontend/src/styles.css](frontend/src/styles.css)
is canonical and this document mirrors its accepted values. Each `colors.*` maps
directly to `--color-*`, `rounded.*` to `--radius-*`, `spacing.*` to `--space-*`, and
`typography.sans.fontFamily` to `--font-sans`. Shared components consume these
variables. Change runtime and this document together; `frontend/tests/design.test.js`
checks documented token values against CSS.

## Colors

The workspace is cool off-white (`background`), with true white panels (`surface`)
and subtle blue-gray borders. Midnight blue belongs to navigation. The blue
`primary` indicates the refresh action, selected chart range, chart line, and
keyboard focus. Success green means recorded completion; amber signals
interruption or incomplete evidence; red signals a failed result or error.
Status words always accompany color. This is a light workspace with a dark rail,
not a theme switcher. Forced colors retain system scrollbars and readable states.

## Typography

Use local system sans fonts without network font requests. Heading sizes are
36px desktop and 29px mobile, panel titles 21px/18px, body 14px, utility labels
11–13px. Headings use compact tracking; paragraphs keep normal tracking and a
1.5 line height. Labels use sentence case. Values use tabular numbers where
comparison matters. No artificial all-caps eyebrows.

## Layout

At 1440px the navigation rail is 216px; at 1536px it is 236px. Main gutters are
24px, panels 20px, and section gaps 16px. The four-column summary is an open band
with vertical dividers. The main row pairs the market chart and training process
in a 1.57:1 ratio, followed by a full-width semantic research table. Research and
Paper trading reuse the same shell, summary, panels, and status treatment.

Below 1000px, content panels stack. Below 640px, the rail becomes a compact top
navigation bar, gutters shrink to 16px, and summary metrics form two columns.
The document owns vertical scrolling. Tables own horizontal overflow, and the
optional price table alone has a 300px vertical bound. Reports paginate at eight
runs per page. Do not clip rows or hide columns silently.

## Elevation & Depth

Hierarchy comes from surface contrast, spacing, and one-pixel borders. Panels
have no decorative drop shadow or nested card frame. Error text remains inline
with a quiet persistent banner. Refresh keeps the previous snapshot visible.

## Shapes

Panels have 10px radii; buttons and navigation selections have 6px radii. Stage
markers and supplemental status dots are circular. Icons use restrained vector
strokes and are decorative when adjacent text conveys the meaning.

## Components

### Foundational visual states

Shared `Panel`, `Status`, and `EmptyState` live in `frontend/src/primitives.jsx`.
`useDashboard` owns initial loading, manual refresh, 30-second polling, timeout,
errors, cancellation, and stale snapshot retention. Polling avoids overlapping
requests; failed reads retain the last snapshot and warning. The initial load uses a real spinner in a stable
reserved region, with text remaining visible under reduced motion. Empty and
unavailable are distinct from zero. Snapshot timestamps indicate when the API
was read; they do not imply a live feed or a freshly modified research result.

### Buttons and actions

Refresh is blue and solid, with fixed width and an explicit busy state. Chart
range buttons use `aria-pressed`; report disclosure uses `aria-expanded` and
explicit View/Hide labels. Secondary buttons use borders, and text actions have
a quiet hover background. All controls have visible focus and native keyboard
behavior. No order, account, funding, or live activation actions exist here.

### Navigation and data display

Overview, Research, and Paper trading are hash-backed navigation links. Current
view uses `aria-current="page"`; route changes update the document title to
`{Page} — HFT` and focus its heading. There are no overlays or forms in this
read-only scope. Charts use accessible title/description and an optional semantic
data table. Tables preserve headers and column relationships at narrow widths.

### Canonical UI map

| Capability | Canonical owner | Source of truth | Allowed variants | Verification |
| --- | --- | --- | --- | --- |
| Scrollbar | `frontend/src/styles.css` global baseline | DESIGN.md | geometry-only table gutter | computed styles / static audit |
| Read-only async state | `frontend/src/useDashboard.js` | API contract | initial / refresh / stale / error | browser failure and recovery |
| Status | `frontend/src/primitives.jsx` | API status | success / warning / danger / neutral | sibling views / labels |
| Report disclosure | `frontend/src/ResearchRuns.jsx` | Local report | expanded / collapsed | keyboard / `aria-expanded` |
| Navigation | `frontend/src/App.jsx` | Hash routes | three hash routes | browser Back / title / focus |

Selection, forms, authored select popups, date pickers, toasts, mutations, and
CRUD are outside the application scope. No second UX contract is needed for
this single report flow and three read-only views.

### Iconography and motion

`frontend/src/Icon.jsx` owns small custom SVG navigation and utility icons using
1.8px rounded strokes. The HFT brand uses the concept's three ascending blue
bars. Motion is limited to loading rotation and 130ms control feedback, disabled
with `prefers-reduced-motion`. No persistent decorative animation.

### Content and data visualization

Write direct explanations tied to actual evidence. Market prices are daily raw
last reported trades, not strategy returns. Reserved final-test history stays
outside the chart. Never seed chart points or simulate a successful account.
Use actual API dates, counts, status, and report details, with no mock fallback.
Candidate trials and training steps belong in the research context.

Report disclosure distinguishes frozen settings and readable reasons from actual
final-test results. The API exposes those results only after final-test consumption.
Cash results and expectancy use USD, return and drawdown are fractions displayed
as percentages, and the paired daily-return advantage interval uses basis points
(1 bp = 0.01 percentage points). Missing evidence stays unavailable. Decision
coverage and diagnostics are explicitly scoped to development data. The last
training stage is Paper eligibility; its check never implies an active account.

## Do's and Don'ts

- Do preserve the summary, chart, training timeline, and table hierarchy.
- Do use the same status and recovery treatment in all three views.
- Do expose real reports through accessible disclosure and real price data through a table.
- Don't present unavailable metrics as zero, interrupted research as running, or eligibility as a connected paper account.
- Don't add credentials, paid services, order controls, or artificial freshness.

Intentional deviations from the concept: the image's invented chart dates and
values are replaced by API data; inert ellipses are replaced by working report
disclosure; snapshot time is explicit; the footer and workspace label state
read-only context; development date metadata and dataset notes are truthful.
