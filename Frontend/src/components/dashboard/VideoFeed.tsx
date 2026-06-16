import { Video, AlertTriangle } from "lucide-react";
import React, { useCallback, useEffect, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";

interface VideoFeedProps {
  isMonitoring: boolean;
  threatLevel: "safe" | "warning" | "danger";
  mediaSrc?: string | null;
  mediaType?: "image" | "video" | null;
}

export const VideoFeed = ({ isMonitoring, threatLevel, mediaSrc, mediaType }: VideoFeedProps) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);

  const isLocalhost = () => {
    if (typeof window === "undefined") return false;
    const host = window.location.hostname;
    return host === "localhost" || host === "127.0.0.1" || host === "::1";
  };

  const isCameraContextSecure = useCallback(() => {
    if (typeof window === "undefined") return false;
    return window.isSecureContext || isLocalhost();
  }, []);

  const getLocalhostRedirectUrl = () => {
    if (typeof window === "undefined") return null;
    const { hostname, port, pathname, search, hash } = window.location;

    const isLikelyLocalDevHost =
      hostname === "0.0.0.0" ||
      hostname === "[::]" ||
      hostname === "::" ||
      hostname.endsWith(".local") ||
      /^127\./.test(hostname) ||
      /^10\./.test(hostname) ||
      /^192\.168\./.test(hostname) ||
      /^172\.(1[6-9]|2\d|3[0-1])\./.test(hostname);

    if (!isLikelyLocalDevHost || hostname === "localhost") {
      return null;
    }

    return `http://localhost${port ? `:${port}` : ""}${pathname}${search}${hash}`;
  };

  const getUserMediaCompat = (constraints: MediaStreamConstraints): Promise<MediaStream> => {
    if (typeof navigator === "undefined") {
      return Promise.reject(new Error("Navigator is not available in this environment."));
    }

    if (navigator.mediaDevices?.getUserMedia) {
      return navigator.mediaDevices.getUserMedia(constraints);
    }

    const legacyGetUserMedia =
      (navigator as Navigator & {
        webkitGetUserMedia?: (
          constraints: MediaStreamConstraints,
          successCallback: (stream: MediaStream) => void,
          errorCallback: (err: unknown) => void
        ) => void;
        mozGetUserMedia?: (
          constraints: MediaStreamConstraints,
          successCallback: (stream: MediaStream) => void,
          errorCallback: (err: unknown) => void
        ) => void;
      }).webkitGetUserMedia ||
      (navigator as Navigator & {
        mozGetUserMedia?: (
          constraints: MediaStreamConstraints,
          successCallback: (stream: MediaStream) => void,
          errorCallback: (err: unknown) => void
        ) => void;
      }).mozGetUserMedia;

    if (!legacyGetUserMedia) {
      return Promise.reject(new Error("This browser does not support camera access (getUserMedia)."));
    }

    return new Promise((resolve, reject) => legacyGetUserMedia.call(navigator, constraints, resolve, reject));
  };

  const buildCameraErrorMessage = (err: unknown) => {
    if (err && typeof err === "object" && "name" in err) {
      const name = String((err as { name?: string }).name || "");
      if (name === "NotAllowedError" || name === "PermissionDeniedError") {
        return "Camera permission was denied. Allow camera access in your browser site settings and reload.";
      }
      if (name === "NotFoundError" || name === "DevicesNotFoundError") {
        return "No camera device was found. Connect a camera and try again.";
      }
      if (name === "NotReadableError" || name === "TrackStartError") {
        return "Camera is already in use by another app. Close other apps using the camera and retry.";
      }
    }

    return (err && (err as Error).message) || "Unable to access camera";
  };

  const getThreatColor = () => {
    switch (threatLevel) {
      case "danger":
        return "border-destructive cyber-glow-red";
      case "warning":
        return "border-warning";
      default:
        return "border-primary/30";
    }
  };

  useEffect(() => {
    // Only start camera when monitoring and no external media is provided
    let stream: MediaStream | null = null;
    const videoEl = videoRef.current;
    let retries = 0;
    const maxRetries = 2;

    const startCamera = async () => {
      if (!isMonitoring) return;
      if (!isCameraContextSecure()) {
        const redirectUrl = getLocalhostRedirectUrl();
        if (redirectUrl) {
          window.location.replace(redirectUrl);
          return;
        }
        setCameraError("Camera requires a secure context. Use HTTPS, or open this app on http://localhost.");
        return;
      }

      try {
        stream = await getUserMediaCompat({ video: { facingMode: "user" }, audio: false });
        if (videoEl) {
          videoEl.srcObject = stream;
          await videoEl.play().catch(() => {});
        }
        setCameraError(null);
      } catch (err) {
        const errorMsg = buildCameraErrorMessage(err);
        setCameraError(errorMsg);
        
        // If camera not found, show demo mode instead of blocking
        if ((err && typeof err === "object" && ("name" in err)) && 
            (String((err as { name?: string }).name || "") === "NotFoundError" || 
             String((err as { name?: string }).name || "") === "DevicesNotFoundError")) {
          // Set a helpful message for demo mode
          setCameraError("Camera not detected. Using demo mode - upload files to analyze or try again with a camera connected.");
        }
      }
    };

    // start only if monitoring and no uploaded media is shown
    if (isMonitoring && !mediaSrc) {
      startCamera();
    }

    return () => {
      if (stream) stream.getTracks().forEach((t) => t.stop());
      if (videoEl) {
        const src = videoEl.srcObject as MediaStream | null;
        if (src) {
          try {
            src.getTracks().forEach((t) => t.stop());
          } catch (err) {
            // ignore
          }
        }
        videoEl.srcObject = null;
      }
    };
  }, [isMonitoring, mediaSrc, isCameraContextSecure]);

  return (
    <div className={`glass-card relative overflow-hidden aspect-video ${getThreatColor()} border-2 transition-all duration-500`}>
      {/* Video Background - live camera when monitoring */}
      <div className="absolute inset-0 bg-cyber-slate cyber-grid">
        {/* Parent should pass mediaSrc/mediaType when user uploads a file. If provided, render that instead of camera. */}
        {mediaSrc ? (
          mediaType === "image" ? (
            <img src={mediaSrc} alt="Uploaded media" className="absolute inset-0 w-full h-full object-cover" />
          ) : (
            <video key={mediaSrc} src={mediaSrc} className="absolute inset-0 w-full h-full object-cover" playsInline muted autoPlay loop />
          )
        ) : isMonitoring ? (
          <video
            ref={videoRef}
            className="absolute inset-0 w-full h-full object-cover"
            playsInline
            muted
          />
        ) : (
          // fallback static simulated feed when not monitoring
          <>
            <div className="absolute inset-0 bg-gradient-to-br from-secondary/50 via-background to-secondary/30" />
          </>
        )}

        {/* Face detection placeholder */}
        {isMonitoring && (
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-32 h-40 border-2 border-primary/60 rounded-lg animate-reticle">
            <div className="absolute -top-2 left-1/2 -translate-x-1/2 px-2 py-0.5 bg-primary/20 rounded text-[10px] font-mono text-primary">
              SUBJECT 01
            </div>
          </div>
        )}

        {cameraError && !mediaSrc && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/40 backdrop-blur-sm text-white p-6">
            <div className="text-center flex flex-col items-center gap-3">
              <AlertTriangle className="w-8 h-8 text-yellow-500" />
              <div>
                <div className="font-semibold mb-1">Camera Not Available</div>
                <div className="text-xs text-gray-300 leading-relaxed max-w-xs">{cameraError}</div>
                <div className="text-xs text-gray-400 mt-3">💡 Tip: You can upload images or videos to test the detector</div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Scanning Animation */}
      {isMonitoring && (
        <div className="scan-line" />
      )}

      {/* Reticle Corners */}
      <div className="reticle-corner top-left animate-reticle" />
      <div className="reticle-corner top-right animate-reticle" />
      <div className="reticle-corner bottom-left animate-reticle" />
      <div className="reticle-corner bottom-right animate-reticle" />

      {/* Live Badge */}
      <div className="absolute top-4 left-12">
        <Badge 
          variant="destructive" 
          className={`font-mono text-xs px-3 py-1 ${isMonitoring ? 'animate-pulse bg-destructive' : 'bg-muted text-muted-foreground'}`}
        >
          <span className={`w-2 h-2 rounded-full mr-2 ${isMonitoring ? 'bg-white' : 'bg-muted-foreground'}`} />
          {isMonitoring ? "LIVE" : "OFFLINE"}
        </Badge>
      </div>

      {/* Threat Warning Overlay */}
      {threatLevel === "danger" && (
        <div className="absolute inset-0 bg-destructive/10 flex items-center justify-center animate-pulse">
          <div className="flex items-center gap-3 px-6 py-3 bg-destructive/90 rounded-lg">
            <AlertTriangle className="w-6 h-6 text-white" />
            <span className="font-semibold text-white text-glow-red">DEEPFAKE DETECTED</span>
          </div>
        </div>
      )}

      {/* Bottom Stats Bar */}
      <div className="absolute bottom-0 left-0 right-0 px-4 py-3 bg-gradient-to-t from-background/90 to-transparent">
        <div className="flex items-center justify-between text-xs font-mono text-muted-foreground">
          <div className="flex items-center gap-4">
            <span>RES: 1920x1080</span>
            <span>FPS: 30</span>
            <span>CODEC: H.264</span>
          </div>
          <div className="flex items-center gap-2">
            <Video className="w-4 h-4 text-primary" />
            <span className="text-primary">CAM-01</span>
          </div>
        </div>
      </div>
    </div>
  );
};
