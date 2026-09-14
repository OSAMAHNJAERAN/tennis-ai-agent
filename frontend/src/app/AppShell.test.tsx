import axe from "axe-core";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { AnalysisProvider } from "./AnalysisContext";
import { AppShell } from "./AppShell";

function renderShell() {
  return render(
    <MemoryRouter initialEntries={["/?run=phase5_scoring_1"]}>
      <Routes>
        <Route element={<AnalysisProvider><AppShell /></AnalysisProvider>}>
          <Route index element={<div>Overview route content</div>} />
          <Route path="match-center" element={<div>Analysis route content</div>} />
          <Route path="analysis" element={<div>Analysis route content</div>} />
        </Route>
      </Routes>
    </MemoryRouter>,
  );
}

describe("application shell", () => {
  it("navigates with semantic primary links", async () => {
    const user = userEvent.setup();
    renderShell();
    expect(await screen.findByText("Overview route content")).toBeInTheDocument();
    await user.click(screen.getAllByRole("link", { name: /Match Center|Match analysis/i })[0]!);
    expect(screen.getByText("Analysis route content")).toBeInTheDocument();
  });

  it("has no automated accessibility violations", async () => {
    const { container } = renderShell();
    await screen.findByText("Overview route content");
    const results = await axe.run(container, { rules: { "color-contrast": { enabled: false } } });
    expect(results.violations).toEqual([]);
  });
});
