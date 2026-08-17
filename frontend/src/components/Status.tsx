type StatusProps = {
  loading?: boolean;
  error?: string | null;
  empty?: string | null;
};

export function Status({ loading = false, error = null, empty = null }: StatusProps) {
  if (loading) {
    return <p className="status">Loading from the API…</p>;
  }
  if (error) {
    return (
      <p className="status status--error" role="alert">
        {error}
      </p>
    );
  }
  if (empty) {
    return <p className="status">{empty}</p>;
  }
  return null;
}
