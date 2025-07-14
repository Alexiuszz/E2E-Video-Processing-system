import React from "react";

type Props = {
  segments: any[];
};

export default function Button({ segments }: Props) {
  const downloadJSON = () => {
    const blob = new Blob([JSON.stringify(segments, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "segments.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <button
      onClick={downloadJSON}
      className="mt-4 bg-green-600 text-white px-4 py-2 rounded"
    >
      Download JSON
    </button>
  );
}