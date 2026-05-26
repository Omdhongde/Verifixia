import { ShieldAlert, ShieldCheck, ImageIcon, Film, Info, Zap } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

type PredictionType = "Real" | "Fake" | "Unknown" | "Deepfake" | "AIGenerated";

interface AnalysisSummaryProps {
  prediction?: PredictionType;
  confidence?: number | null;
  filename?: string;
  isVideo?: boolean;
  threatLevel?: string;
  modelUsed?: string;
  analysis?: {
    level?: string;
    description?: string;
    recommendation?: string;
  };
  classProbs?: {
    Real?: number;
    Deepfake?: number;
    AIGenerated?: number;
  };
}

export const AnalysisSummary = ({
  prediction = "Unknown",
  confidence,
  filename,
  isVideo,
  threatLevel,
  modelUsed,
  analysis,
  classProbs,
}: AnalysisSummaryProps) => {
  const normalizedConfidence =
    typeof confidence === "number" && !Number.isNaN(confidence)
      ? Math.round(confidence <= 1 ? confidence * 100 : confidence)
      : null;

  const isFake = prediction === "Fake" || prediction === "Deepfake";
  const isReal = prediction === "Real";
  const isAIGen = prediction === "AIGenerated";

  const getThreatBadgeColor = (level?: string) => {
    switch (level) {
      case "high":
        return "bg-destructive/10 text-destructive border-destructive/40";
      case "medium":
        return "bg-warning/10 text-warning border-warning/40";
      case "low":
        return "bg-emerald-500/10 text-emerald-400 border-emerald-500/40";
      default:
        return "bg-muted text-muted-foreground border-border/60";
    }
  };

  const getPredictionIcon = () => {
    if (isFake) return <ShieldAlert className="w-3 h-3 mr-1" />;
    if (isReal) return <ShieldCheck className="w-3 h-3 mr-1" />;
    if (isAIGen) return <Zap className="w-3 h-3 mr-1" />;
    return null;
  };

  const getPredictionLabel = () => {
    if (isFake) return "Deepfake";
    if (isReal) return "Authentic";
    if (isAIGen) return "AI-Generated";
    return "No analysis yet";
  };

  return (
    <Card className="border border-border/60 bg-card/80 backdrop-blur-sm">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
        <CardTitle className="text-xs font-medium text-muted-foreground tracking-[0.18em] uppercase">
          Last Analysis
        </CardTitle>
        <span
          className={cn(
            "inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-semibold border",
            isFake &&
              "bg-destructive/10 text-destructive border-destructive/40 shadow-[0_0_18px_rgba(248,113,113,0.35)]",
            isReal && "bg-emerald-500/10 text-emerald-400 border-emerald-500/40",
            isAIGen && "bg-purple-500/10 text-purple-400 border-purple-500/40",
            !isFake && !isReal && !isAIGen && "bg-muted text-muted-foreground border-border/60",
          )}
        >
          {getPredictionIcon()}
          {getPredictionLabel()}
        </span>
      </CardHeader>
      <CardContent className="space-y-3 pt-0 text-xs">
        {/* File Info */}
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2 text-muted-foreground">
            {isVideo ? (
              <Film className="w-4 h-4 text-primary" />
            ) : (
              <ImageIcon className="w-4 h-4 text-primary" />
            )}
            <span className="truncate max-w-[160px] font-mono text-[11px]">
              {filename ?? "Awaiting upload"}
            </span>
          </div>

          {normalizedConfidence !== null && (
            <span className="font-mono text-[11px] text-primary font-semibold">
              {normalizedConfidence.toString().padStart(2, "0")}%
            </span>
          )}
        </div>

        {/* Class Probabilities (Multi-class model) */}
        {classProbs && (classProbs.Real || classProbs.Deepfake || classProbs.AIGenerated) && (
          <div className="pt-2 border-t border-border/40 space-y-2">
            <p className="text-[10px] font-semibold text-muted-foreground">CLASS PROBABILITIES</p>
            <div className="space-y-1.5">
              {/* Real */}
              <div className="space-y-0.5">
                <div className="flex justify-between text-[10px]">
                  <span className="text-muted-foreground">Real</span>
                  <span className="text-emerald-400 font-mono">
                    {(classProbs.Real || 0).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full bg-emerald-500"
                    style={{ width: `${classProbs.Real || 0}%` }}
                  />
                </div>
              </div>

              {/* Deepfake */}
              <div className="space-y-0.5">
                <div className="flex justify-between text-[10px]">
                  <span className="text-muted-foreground">Deepfake</span>
                  <span className="text-destructive font-mono">
                    {(classProbs.Deepfake || 0).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full bg-destructive"
                    style={{ width: `${classProbs.Deepfake || 0}%` }}
                  />
                </div>
              </div>

              {/* AI-Generated */}
              <div className="space-y-0.5">
                <div className="flex justify-between text-[10px]">
                  <span className="text-muted-foreground">AI-Generated</span>
                  <span className="text-purple-400 font-mono">
                    {(classProbs.AIGenerated || 0).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full bg-purple-500"
                    style={{ width: `${classProbs.AIGenerated || 0}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Threat Level */}
        {threatLevel && (
          <div className="flex items-center justify-between gap-2 pt-1">
            <span className="text-[11px] text-muted-foreground">Threat Level</span>
            <Badge className={cn("text-[10px] font-semibold border", getThreatBadgeColor(threatLevel))}>
              {threatLevel.toUpperCase()}
            </Badge>
          </div>
        )}

        {/* Model Used */}
        {modelUsed && (
          <div className="flex items-center justify-between gap-2">
            <span className="text-[11px] text-muted-foreground">Model</span>
            <span className="text-[10px] font-mono text-foreground">{modelUsed}</span>
          </div>
        )}

        {/* Analysis Details */}
        {analysis && analysis.description && (
          <div className="pt-2 border-t border-border/40 space-y-2">
            <div className="flex items-start gap-2">
              <Info className="w-3 h-3 text-primary mt-0.5 flex-shrink-0" />
              <div className="space-y-1">
                <p className="text-[11px] leading-relaxed text-muted-foreground">
                  {analysis.description}
                </p>
                {analysis.recommendation && (
                  <p className="text-[10px] leading-relaxed text-primary/80 italic">
                    {analysis.recommendation}
                  </p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Default Description */}
        {!analysis?.description && (
          <p className="text-[11px] leading-relaxed text-muted-foreground">
            Upload any face image or short clip. Verifixia AI estimates the likelihood of{" "}
            <span className="font-semibold text-primary/90">synthetic manipulation</span> (deepfake or AI-generated) and feeds
            results into your forensic logs.
          </p>
        )}
      </CardContent>
    </Card>
  );
};
