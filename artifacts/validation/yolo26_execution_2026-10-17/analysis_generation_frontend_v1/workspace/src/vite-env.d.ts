/// <reference types="vite/client" />

interface VideoFrameCallbackMetadata {
  mediaTime: number;
  presentedFrames: number;
}

interface HTMLVideoElement {
  requestVideoFrameCallback?: (
    callback: (now: DOMHighResTimeStamp, metadata: VideoFrameCallbackMetadata) => void,
  ) => number;
  cancelVideoFrameCallback?: (handle: number) => void;
}
