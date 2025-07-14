import React, { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

type Segment = {
  segmentId?: number;
  label?: string;
  start: number;
  end: number;
  topics?: string[];
  segments: Segment[];
  text?: string;
};

type Props = {
  segments: Segment[];
};

export default function Preview({ segments }: Props) {
  const [openMap, setOpenMap] = useState<Record<number, boolean>>({});
  const toggle = (idx: number) =>
    setOpenMap((prev) => ({ ...prev, [idx]: !prev[idx] }));

  return (
    <div className="p-4 text-zinc-800 bg-zinc-50 rounded overflow-y-auto max-h-[600px] w-full">
      {segments.map((seg, idx) => {
        const isOpen = !!openMap[idx];
        return (
          <div key={idx} className="border-b border-zinc-200 pb-2 mb-2">
            <div
              onClick={() => toggle(idx)}
              className="flex justify-between items-center cursor-pointer px-2 py-1"
            >
              <span className="text-sm font-medium">
                {seg.label ?? `Segment ${seg.segmentId ?? idx + 1}`}: <span className="">[{seg.start.toFixed(2)} - {seg.end.toFixed(2)}]</span>
              </span>
              {isOpen ? (
                <ChevronDown size={20} className="text-zinc-500" />
              ) : (
                <ChevronRight size={20} className="text-zinc-500" />
              )}
            </div>
            {isOpen && (
              <div className="mt-2 ml-4 text-xs text-zinc-700 space-y-1">
                {seg.segments.map((sub, j) => (
                  <p key={j}>
                    <span className="font-semibold text-zinc-500">
                      [{sub.start.toFixed(2)} - {sub.end.toFixed(2)}]
                    </span>{" "}
                    {sub.text}
                  </p>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}