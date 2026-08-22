import * as Select from "@radix-ui/react-select";
import { CaretDown, Check } from "@phosphor-icons/react";
import { useAnalysis } from "../app/AnalysisContext";

export function RunPicker() {
  const { runs, selectedRunId, selectRun } = useAnalysis();
  return (
    <Select.Root value={selectedRunId} onValueChange={selectRun}>
      <Select.Trigger className="run-picker" aria-label="Select analysis run">
        <span className="run-picker-copy">
          <span>Analysis run</span>
          <Select.Value />
        </span>
        <Select.Icon><CaretDown size={16} aria-hidden="true" /></Select.Icon>
      </Select.Trigger>
      <Select.Portal>
        <Select.Content className="select-content" position="popper" sideOffset={8}>
          <Select.Viewport>
            {runs.map((run) => (
              <Select.Item className="select-item" value={run.id} key={run.id}>
                <Select.ItemText>{run.label}</Select.ItemText>
                <Select.ItemIndicator><Check size={15} aria-hidden="true" /></Select.ItemIndicator>
              </Select.Item>
            ))}
          </Select.Viewport>
        </Select.Content>
      </Select.Portal>
    </Select.Root>
  );
}

