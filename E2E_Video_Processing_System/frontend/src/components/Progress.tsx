import { CheckIcon } from "lucide-react";

import {
  Timeline,
  TimelineContent,
  TimelineHeader,
  TimelineIndicator,
  TimelineItem,
  TimelineSeparator,
  TimelineTitle,
} from "@/components/ui/timeline";

const items = [
  {
    title: "File Uploading",
    description: "The file is uploading.",
  },
  {
    title: "File Uploaded",
    description: "The file has been uploaded successfully.",
  },
  {
    title: "Transcribing",
    description: "The transcription process has begun.",
  },
    {
        title: "Transcription Completed",
        description: "The transcription has been completed successfully.",
    },
    {
        title: "Segmenting",
        description: "The segmentation process has begun.",
    },
  {
    title: "Segmentation Completed",
    description: "The segmentation has been completed successfully.",
  },
  {
    title: "Processing Finished",
    description: "All processes have been completed successfully.",
  },
];

export default function Progress({step}: { step: number }) {
  return (
    <Timeline defaultValue={0} value={step}>
      {items.map((item, i) => (
        <TimelineItem
          key={i}
          step={i}
          className="group-data-[orientation=vertical]/timeline:ms-10"
        >
          <TimelineHeader>
            <TimelineSeparator className="group-data-[orientation=vertical]/timeline:-left-7 group-data-[orientation=vertical]/timeline:h-[calc(100%-1.5rem-0.25rem)] group-data-[orientation=vertical]/timeline:translate-y-6.5" />
            <TimelineTitle>{item.title}</TimelineTitle>
            <TimelineIndicator className="group-data-completed/timeline-item:bg-primary group-data-completed/timeline-item:text-primary-foreground flex size-6 items-center justify-center group-data-completed/timeline-item:border-none group-data-[orientation=vertical]/timeline:-left-7">
              <CheckIcon
                className="group-not-data-completed/timeline-item:hidden"
                size={16}
              />
            </TimelineIndicator>
          </TimelineHeader>
          <TimelineContent>{item.description}</TimelineContent>
        </TimelineItem>
      ))}
    </Timeline>
  );
}
