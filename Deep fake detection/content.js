chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "analyze_video") {
    const videos = document.getElementsByTagName('video');
    if (videos.length > 0) {
      const video = videos[0];
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const context = canvas.getContext('2d');

      video.addEventListener('seeked', () => {
        context.drawImage(video, 0, 0);
        const frameDataURL = canvas.toDataURL('image/jpeg');
        chrome.runtime.sendMessage({ action: "frame_captured", frame: frameDataURL });
        video.play(); // Resume video playback
      }, { once: true });

      video.pause(); // Pause video temporarily
      video.currentTime += 0.1; // Seek forward slightly to trigger the 'seeked' event
    } else {
      sendResponse({ error: "No video found" });
    }
  }
  return true;
});
