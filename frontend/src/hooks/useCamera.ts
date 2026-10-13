import { useRef, useState, useCallback, useEffect } from "react";

export type CameraStatus = "idle" | "requesting" | "active" | "denied" | "error";

export interface VideoDimensions {
  width: number;
  height: number;
}

export interface UseCameraReturn {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  status: CameraStatus;
  error: string | null;
  /** Native resolution of the active video track, or null when inactive. */
  dimensions: VideoDimensions | null;
  startCamera: () => Promise<void>;
  stopCamera: () => void;
  captureFrame: () => Blob | null;
}

/**
 * Manages browser webcam access.
 * - Requests permission only when startCamera() is called
 * - Cleans up stream tracks on stopCamera() or unmount
 * - captureFrame() draws the current video frame to a canvas and returns a JPEG blob
 */
export function useCamera(): UseCameraReturn {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(document.createElement("canvas"));
  const [status, setStatus] = useState<CameraStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [dimensions, setDimensions] = useState<VideoDimensions | null>(null);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setStatus("idle");
    setError(null);
    setDimensions(null);
  }, []);

  const startCamera = useCallback(async () => {
    if (status === "active") return;
    setStatus("requesting");
    setError(null);

    // Camera access requires a secure context (localhost or HTTPS)
    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus("error");
      setError(
        "Camera access is not available. This app requires a secure context (localhost or HTTPS)."
      );
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;

      // Surface a track that ends unexpectedly (camera unplugged / taken by
      // another app) as an error state instead of a silently frozen frame.
      const [track] = stream.getVideoTracks();
      if (track) {
        track.addEventListener("ended", () => {
          setStatus("error");
          setError("The camera stream ended unexpectedly (device disconnected or taken by another app).");
          setDimensions(null);
          streamRef.current = null;
        });
      }

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        // videoWidth/Height are only valid once metadata has loaded.
        const applyDims = () => {
          const v = videoRef.current;
          if (v && v.videoWidth > 0) {
            setDimensions({ width: v.videoWidth, height: v.videoHeight });
          }
        };
        if (videoRef.current.readyState >= 1) {
          applyDims();
        } else {
          videoRef.current.addEventListener("loadedmetadata", applyDims, { once: true });
        }
      }
      setStatus("active");
    } catch (err) {
      const e = err as DOMException;
      if (e.name === "NotAllowedError" || e.name === "PermissionDeniedError") {
        setStatus("denied");
        setError(
          "Camera permission was denied. Please allow camera access in your browser settings and try again."
        );
      } else if (e.name === "NotFoundError") {
        setStatus("error");
        setError("No camera device was found. Please connect a webcam and try again.");
      } else {
        setStatus("error");
        setError(`Camera error: ${e.message}`);
      }
    }
  }, [status]);

  /**
   * Captures the current video frame and returns it as a JPEG Blob.
   * Returns null if the camera is not active.
   */
  const captureFrame = useCallback((): Blob | null => {
    const video = videoRef.current;
    if (!video || status !== "active" || video.readyState < 2) return null;

    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0);

    // Convert to Blob synchronously via toDataURL
    const dataUrl = canvas.toDataURL("image/jpeg", 0.85);
    const byteString = atob(dataUrl.split(",")[1]);
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);
    for (let i = 0; i < byteString.length; i++) {
      ia[i] = byteString.charCodeAt(i);
    }
    return new Blob([ab], { type: "image/jpeg" });
  }, [status]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  return { videoRef, status, error, dimensions, startCamera, stopCamera, captureFrame };
}
