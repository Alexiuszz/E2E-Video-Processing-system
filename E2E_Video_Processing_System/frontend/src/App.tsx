import { useEffect, useState } from "react";
import api from "./utils/api";
import FileUpload from "./components/FileUpload";
import Preview from "./components/Preview";
import Button from "./components/Button";
import Progress from "./components/Progress";
import { LoaderCircleIcon } from "lucide-react";

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [model, setModel] = useState<string>("openai");
  const [segments, setSegments] = useState<any[]>([]);
  const [transcript, setTranscript] = useState<string>("");
  const [processing, setProcessing] = useState<boolean>(false);
  const [step, setStep] = useState<number>(0);

  const handleUploadAndProcess = async () => {
    if (!file) return;

    setSegments([]);
    setProcessing(true);
    setStep(()=> 0);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const uploadRes = await api.post("/upload", formData);
      setStep(()=> 1);
      const filePath = uploadRes.data.file_path;
      setStep(() => 2);
      const transcribeRes = await api.post(
        `/transcribe?file_path=${filePath}&model=${model}`
      );
      setStep(() => 3);
      const fullTranscript =
        transcribeRes.data
      setTranscript(fullTranscript);
      setStep(()=> 4);
      const segmentRes = await api.post("/segment", 
        transcribeRes.data,
        { params: { with_timestamps: true } });
      setStep(()=> 5);
      setSegments(segmentRes.data);
      setStep(()=> 6);
      setProcessing(false);
    } catch (err: any) {
        alert("Error: " + (err?.response?.data?.detail || err.message));
        setProcessing(false);
        setStep(0);
        setSegments([]);
        setTranscript("");
    }
  };

  useEffect(() => {
    if(segments.length > 0) {
      // setProcessing(false);
      // setStep(0);
    }
  }, [segments]);

  return (
    <div className="flex items-center flex-col max-w-2xl mx-auto p-6 space-y-4 min-h-svh min-w-[390px] w-screen">
      <h1 className=" font-bold">Video Processing System</h1>
      <FileUpload setFile={setFile} />
      <div className="flex items-center space-x-4">
        <label className="text-sm font-semibold">Model:</label>
        <select
          value={model}
          onChange={(e) => setModel(e.target.value)}
          className="border rounded px-2 py-1"
        >
          <option value="openai">OpenAI</option>
          <option value="whisper">Whisper</option>
          <option value="nemo">Nemo Parakeet</option>
        </select>
      </div>
      <button
        onClick={handleUploadAndProcess}
        disabled={!file}
        className="bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
      >
        {processing? 
        (<LoaderCircleIcon
          className="-ms-1 animate-spin"
          size={16}
          aria-hidden="true"
        />):"Upload & Process"}
      </button>
      {(segments.length > 0 || processing) && (
        <Progress step={step}/>
      )}
      {segments.length > 0 && (
        <>
          <h2 className="text-lg font-semibold mt-6">Segmented Transcript</h2>
          <Preview segments={segments} />
          <Button segments={segments} />
        </>
      )}
    </div>
  );
}

export default App;
