import { WarningOctagon } from "@phosphor-icons/react";
import { isRouteErrorResponse, Link, useRouteError } from "react-router-dom";

export default function RouteError() {
  const error = useRouteError();
  const detail = isRouteErrorResponse(error)
    ? `${error.status} ${error.statusText}`
    : error instanceof Error
      ? error.message
      : "An unexpected route error occurred.";
  return (
    <main className="route-error">
      <WarningOctagon size={40} weight="duotone" aria-hidden="true" />
      <p className="eyebrow">Workspace interruption</p>
      <h1>This view could not be rendered</h1>
      <p>{detail}</p>
      <Link className="button button-primary" to="/">Return to overview</Link>
    </main>
  );
}

