import { useState, useEffect, useCallback } from "react";
import { Clock, Gauge, ScanFace } from "lucide-react";
import { toast } from "sonner";

import { Header } from "@/components/dashboard/Header";
import { VideoFeed } from "@/components/dashboard/VideoFeed";
import { ConfidenceGauge } from "@/components/dashboard/ConfidenceGauge";
import { DetectionLog, LogEntry } from "@/components/dashboard/DetectionLog";
import { SecurityScore } from "@/components/dashboard/SecurityScore";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { ControlPanel } from "@/components/dashboard/ControlPanel";
import { AnalysisSummary } from "@/components/dashboard/AnalysisSummary";
import { uploadImage } from "../../api";

const MOCK_LOG_MESSAGES = [
  { message: "Biometric Mismatch Detected", type: "error" as const },
  { message: "Skin Texture Anomaly", type: "warning" as const },
  { message: "Facial Landmarks Verified", type: "success" as const },
  { message: "Micro-Expression Analysis Complete", type: "info" as const },
  { message: "Lip Sync Deviation Detected", type: "warning" as const },
  { message: "Eye Reflection Inconsistency", type: "error" as const },
  { message: "Frame Integrity Check Passed", type: "success" as const },
  { message: "Neural Network Confidence: High", type: "info" as const },
  { message: "Temporal Consistency Warning", type: "warning" as const },
  { message: "GAN Artifact Signature Found", type: "error" as const },
];

const Index = () => {
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [sensitivity, setSensitivity] = useState(65);
  const [confidenceScore, setConfidenceScore] = useState(0);
  const [securityScore, setSecurityScore] = useState(87);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [latency, setLatency] = useState(42);
  const [fps, setFps] = useState(30);
  const [facesScanned, setFacesScanned] = useState(1247);
  const [threatLevel, setThreatLevel] = useState<"safe" | "warning" | "danger">("safe");
  const [lastPrediction, setLastPrediction] = useState<"Real" | "Fake" | "Unknown">("Unknown");
  const [lastConfidence, setLastConfidence] = useState<number | null>(null);
  const [lastFilename, setLastFilename] = useState<string | undefined>(undefined);
  const [lastIsVideo, setLastIsVideo] = useState<boolean | undefined>(undefined);
  const [mediaSrc, setMediaSrc] = useState<string | null>(null);
  const [mediaType, setMediaType] = useState<"image" | "video" | null>(null);

  const addLogEntry = useCallback((entry: Omit<LogEntry, "id" | "timestamp">) => {
    const now = new Date();
    const timestamp = now.toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    }) + "." + now.getMilliseconds().toString().padStart(3, "0");

    const newEntry: LogEntry = {
      id: crypto.randomUUID(),
      timestamp,
      ...entry,
    };

    setLogs((prev) => [newEntry, ...prev].slice(0, 50));
  }, []);

  // Simulate monitoring activity (metrics only – NOT fake predictions)
  useEffect(() => {
    if (!isMonitoring) {
      setThreatLevel("safe");
      return;
    }

    // Only update metrics (latency, fps, faces) – do NOT fake prediction scores
    // Confidence / prediction are only set by real backend responses (handleUploadMedia)
    const metricInterval = setInterval(() => {
      setLatency(35 + Math.floor(Math.random() * 30));
      setFps(28 + Math.floor(Math.random() * 5));
      setFacesScanned((prev) => prev + Math.floor(Math.random() * 3));
    }, 1500);

    // Random log entries to show the system is active
    const logInterval = setInterval(() => {
      const randomLog = MOCK_LOG_MESSAGES[Math.floor(Math.random() * MOCK_LOG_MESSAGES.length)];
      addLogEntry(randomLog);
    }, 3000);

    return () => {
      clearInterval(metricInterval);
      clearInterval(logInterval);
    };
  }, [isMonitoring, addLogEntry]);

  // Trigger alert toast when deepfake detected
  useEffect(() => {
    if (threatLevel === "danger" && isMonitoring) {
      toast.error("⚠️ DEEPFAKE DETECTED", {
        description: "High probability of synthetic media manipulation detected in video feed.",
        duration: 5000,
        className: "cyber-glow-red",
      });
    }
  }, [threatLevel, isMonitoring]);

  const handleStartStop = () => {
    setIsMonitoring(!isMonitoring);
    if (!isMonitoring) {
      // Reset previous upload results when starting a new monitoring session
      setLastPrediction("Unknown");
      setLastConfidence(null);
      setLastFilename(undefined);
      setLastIsVideo(undefined);
      setConfidenceScore(0);
      setThreatLevel("safe");
      setMediaSrc(null);
      setMediaType(null);
      addLogEntry({ message: "Monitoring session started – upload a file to analyse", type: "info" });
      toast.success("Monitoring Started", {
        description: "Upload an image or video file to run deepfake analysis.",
      });
    } else {
      addLogEntry({ message: "Monitoring session ended", type: "info" });
      setConfidenceScore(0);
      setThreatLevel("safe");
      toast.info("Monitoring Stopped", {
        description: "Video analysis has been paused.",
      });
    }
  };

  const handleUploadMedia = async (file: File) => {
    addLogEntry({
      message: `Upload received: "${file.name}". Running deepfake analysis...`,
      type: "info",
    });

    try {
      const result = await uploadImage(file);

      const prediction =
        result?.prediction ??
        (result?.isFake === true ? "Fake" : result?.isFake === false ? "Real" : "Unknown");

      const rawConfidence =
        typeof result?.confidence === "string"
          ? parseFloat(result.confidence)
          : result?.confidence;

      if (typeof rawConfidence === "number" && !Number.isNaN(rawConfidence)) {
        const percentScore = rawConfidence <= 1 ? rawConfidence * 100 : rawConfidence;
        setConfidenceScore(Math.round(Math.min(100, Math.max(0, percentScore))));
        setLastConfidence(percentScore);
      }

      setLastPrediction(prediction as "Real" | "Fake" | "Unknown");
      setLastFilename(result?.filename ?? file.name);
      setLastIsVideo(result?.isVideo === true);

      // Update threat level based on REAL backend prediction
      if (prediction === "Fake") {
        setThreatLevel("danger");
      } else if (prediction === "Real") {
        setThreatLevel("safe");
      } else {
        setThreatLevel("warning");
      }

      // If backend provided a public file URL, show it in the player
      if (result?.file_url) {
        setMediaSrc(result.file_url);
        setMediaType(result.isVideo ? "video" : "image");
      } else {
        // Create a local blob URL for preview
        const localUrl = URL.createObjectURL(file);
        setMediaSrc(localUrl);
        setMediaType(result?.isVideo ? "video" : "image");
      }

      addLogEntry({
        message: `Analysis complete for "${file.name}": ${prediction}`,
        type: prediction === "Fake" ? "error" : prediction === "Real" ? "success" : "info",
      });

      toast.success(`Media analysis complete (${prediction})`, {
        description:
          typeof rawConfidence === "number"
            ? `Confidence: ${Math.round(
                rawConfidence <= 1 ? rawConfidence * 100 : rawConfidence
              ).toString()}%`
            : "Confidence score not available.",
      });
    } catch (error) {
      console.error("Upload failed", error);
      addLogEntry({
        message: `Analysis failed for "${file.name}". Please try again.`,
        type: "error",
      });
      toast.error("Media analysis failed", {
        description: "Could not analyze the uploaded file. Please try again.",
      });
    }
  };

  return (
    <div className="min-h-screen bg-background cyber-grid p-4 md:p-6">
      <div className="max-w-[1600px] mx-auto space-y-4 md:space-y-6">
        {/* Header */}
        <Header />

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 md:gap-6">
          {/* Left Column - Video Feed & Gauge */}
          <div className="lg:col-span-2 space-y-4 md:space-y-6">
            <VideoFeed isMonitoring={isMonitoring} threatLevel={threatLevel} mediaSrc={mediaSrc} mediaType={mediaType} />
            <ConfidenceGauge value={confidenceScore} isMonitoring={isMonitoring} />
          </div>

          {/* Right Column - Sidebar */}
          <div className="space-y-4 md:space-y-6">
            <SecurityScore score={securityScore} trend="up" />
            <ControlPanel
              isMonitoring={isMonitoring}
              sensitivity={sensitivity}
              onStartStop={handleStartStop}
              onSensitivityChange={setSensitivity}
              onUploadMedia={handleUploadMedia}
            />
            <AnalysisSummary
              prediction={lastPrediction}
              confidence={lastConfidence}
              filename={lastFilename}
              isVideo={lastIsVideo}
            />
            <div className="h-[300px]">
              <DetectionLog logs={logs} />
            </div>
          </div>
        </div>

        {/* Bottom Metrics Row */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <MetricCard
            title="Average Latency"
            value={latency}
            unit="ms"
            icon={Clock}
            trend={{ value: 12, isPositive: true }}
          />
          <MetricCard
            title="Frame Analysis Rate"
            value={fps}
            unit="FPS"
            icon={Gauge}
            trend={{ value: 5, isPositive: true }}
          />
          <MetricCard
            title="Total Faces Scanned"
            value={facesScanned.toLocaleString()}
            icon={ScanFace}
          />
        </div>
      </div>
    </div>
  );
};

export default Index;
