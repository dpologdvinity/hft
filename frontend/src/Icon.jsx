const paths = {
  overview: (
    <>
      <path d="M4 13h3v7H4zM10 9h3v11h-3zM16 4h3v16h-3z" />
    </>
  ),
  research: (
    <>
      <rect x="5" y="3" width="14" height="18" rx="1" />
      <path d="M8 7h8M8 11h8M8 15h5" />
    </>
  ),
  paper: (
    <>
      <path d="M4 3v17h17M7 15l5-6 4 3 5-7M17 5h4v4" />
    </>
  ),
  refresh: (
    <>
      <path d="M20 7v5h-5M19 12a7 7 0 1 0-2 5M20 7l-3-3" />
    </>
  ),
  folder: (
    <>
      <path d="M3 7V5h6l2 3h10v12H3z" />
    </>
  ),
  chevron: <path d="m9 5 7 7-7 7" />,
  check: <path d="m5 12 4 4L19 6" />,
  pause: <path d="M9 6v12M15 6v12" />,
};

export default function Icon({ name, className = '' }) {
  return (
    <svg
      className={`icon ${className}`}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name] || paths.research}
    </svg>
  );
}
