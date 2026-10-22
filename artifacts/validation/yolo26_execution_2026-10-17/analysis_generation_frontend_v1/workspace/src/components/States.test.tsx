import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { DataError, EmptyState, SkeletonPage } from "./ui";

describe("shared states", () => {
  it("renders loading and empty messages", () => {
    const { rerender } = render(<SkeletonPage />);
    expect(screen.getByRole("status", { name: "Loading analysis" })).toBeInTheDocument();
    rerender(<EmptyState title="Nothing here" detail="Choose another run." />);
    expect(screen.getByRole("status")).toHaveTextContent("Choose another run");
  });

  it("offers a retry action for errors", async () => {
    const retry = vi.fn();
    const user = userEvent.setup();
    render(<DataError message="Snapshot invalid" onRetry={retry} />);
    expect(screen.getByRole("alert")).toHaveTextContent("Snapshot invalid");
    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(retry).toHaveBeenCalledOnce();
  });
});

