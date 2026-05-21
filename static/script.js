(() => {
    if (window.__raagaSharedScriptReady) {
        return;
    }
    window.__raagaSharedScriptReady = true;

    const root = document.documentElement;
    const themeToggle = document.querySelector("[data-theme-toggle]");
    const themeIcon = document.querySelector("[data-theme-icon]");

    function syncThemeLabel() {
        const theme = root.dataset.theme || "dark";
        if (themeIcon) {
            themeIcon.textContent = theme === "dark" ? "Light" : "Dark";
        }
    }

    if (themeToggle && themeToggle.dataset.bound !== "true") {
        themeToggle.dataset.bound = "true";
        themeToggle.addEventListener("click", () => {
            const nextTheme = root.dataset.theme === "dark" ? "light" : "dark";
            root.dataset.theme = nextTheme;
            localStorage.setItem("raaga-theme", nextTheme);
            syncThemeLabel();
        });
    }

    syncThemeLabel();

    document.querySelectorAll(".file-picker input[type='file']").forEach((input) => {
        if (input.dataset.bound === "true") {
            return;
        }
        input.dataset.bound = "true";
        input.addEventListener("change", () => {
            const label = input.closest(".file-picker")?.querySelector("[data-file-name]");
            if (label) {
                label.textContent = input.files?.[0]?.name || "No file selected";
            }
        });
    });

    document.querySelectorAll("form").forEach((form) => {
        if (form.dataset.bound === "true") {
            return;
        }
        form.dataset.bound = "true";
        form.addEventListener("submit", () => {
            const submitter = form.querySelector("button[type='submit']");
            if (!submitter) {
                return;
            }

            submitter.dataset.originalLabel = submitter.textContent.trim();
            submitter.textContent = submitter.dataset.loadingLabel || "Working...";
            submitter.classList.add("is-loading");
            submitter.disabled = true;
        });
    });

    // =====================================
    // LIVE MICROPHONE RECORDER
    // =====================================

    const recorderPanel = document.querySelector(".recorder-panel");
    const startRecordingBtn = document.getElementById("startRecordingBtn");
    const stopRecordingBtn = document.getElementById("stopRecordingBtn");
    const recordingStatus = document.getElementById("recordingStatus");
    const recordingTimer = document.getElementById("recordingTimer");
    const liveWaveform = document.getElementById("liveWaveform");
    const recordedPlayback = document.getElementById("recordedPlayback");
    const recordingDownload = document.getElementById("recordingDownload");

    let mediaRecorder = null;
    let recordingChunks = [];
    let recordingStream = null;
    let audioContext = null;
    let analyser = null;
    let waveformAnimation = null;
    let timerInterval = null;
    let recordingStartedAt = null;

    function getSupportedMimeType() {
        if (!window.MediaRecorder) {
            return "";
        }

        const candidates = [
            "audio/webm;codecs=opus",
            "audio/webm",
            "audio/ogg;codecs=opus",
            "audio/ogg",
        ];

        return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || "";
    }

    function setRecorderStatus(message, state = "idle") {
        if (!recordingStatus) {
            return;
        }

        recordingStatus.textContent = message;
        recordingStatus.dataset.state = state;
    }

    function formatTimer(seconds) {
        const minutes = Math.floor(seconds / 60).toString().padStart(2, "0");
        const remainingSeconds = Math.floor(seconds % 60).toString().padStart(2, "0");
        return `${minutes}:${remainingSeconds}`;
    }

    function startTimer() {
        if (!recordingTimer) {
            return;
        }

        recordingStartedAt = Date.now();
        recordingTimer.textContent = "00:00";
        timerInterval = window.setInterval(() => {
            const elapsed = (Date.now() - recordingStartedAt) / 1000;
            recordingTimer.textContent = formatTimer(elapsed);
        }, 250);
    }

    function stopTimer() {
        if (timerInterval) {
            window.clearInterval(timerInterval);
            timerInterval = null;
        }
    }

    function drawIdleWaveform() {
        if (!liveWaveform) {
            return;
        }

        const canvasContext = liveWaveform.getContext("2d");
        const { width, height } = liveWaveform;
        canvasContext.clearRect(0, 0, width, height);
        canvasContext.fillStyle = "#15111a";
        canvasContext.fillRect(0, 0, width, height);
        canvasContext.strokeStyle = "rgba(255, 155, 115, 0.45)";
        canvasContext.lineWidth = 2;
        canvasContext.beginPath();
        canvasContext.moveTo(0, height / 2);
        canvasContext.lineTo(width, height / 2);
        canvasContext.stroke();
    }

    function drawLiveWaveform() {
        if (!analyser || !liveWaveform) {
            return;
        }

        const canvasContext = liveWaveform.getContext("2d");
        const bufferLength = analyser.fftSize;
        const dataArray = new Uint8Array(bufferLength);
        const { width, height } = liveWaveform;

        function draw() {
            waveformAnimation = window.requestAnimationFrame(draw);
            analyser.getByteTimeDomainData(dataArray);

            const gradient = canvasContext.createLinearGradient(0, 0, width, 0);
            gradient.addColorStop(0, "#ff7f93");
            gradient.addColorStop(0.5, "#ff9b73");
            gradient.addColorStop(1, "#76e7f2");

            canvasContext.fillStyle = "#15111a";
            canvasContext.fillRect(0, 0, width, height);
            canvasContext.globalAlpha = 0.16;
            for (let x = 0; x < width; x += 32) {
                canvasContext.fillStyle = "rgba(255, 255, 255, 0.12)";
                canvasContext.fillRect(x, 0, 1, height);
            }
            canvasContext.globalAlpha = 1;

            canvasContext.lineWidth = 2.4;
            canvasContext.strokeStyle = gradient;
            canvasContext.beginPath();

            const sliceWidth = width / bufferLength;
            let x = 0;

            for (let i = 0; i < bufferLength; i += 1) {
                const value = dataArray[i] / 128.0;
                const y = (value * height) / 2;

                if (i === 0) {
                    canvasContext.moveTo(x, y);
                } else {
                    canvasContext.lineTo(x, y);
                }

                x += sliceWidth;
            }

            canvasContext.lineTo(width, height / 2);
            canvasContext.stroke();
        }

        draw();
    }

    function stopWaveform() {
        if (waveformAnimation) {
            window.cancelAnimationFrame(waveformAnimation);
            waveformAnimation = null;
        }

        if (audioContext) {
            audioContext.close();
            audioContext = null;
        }

        analyser = null;
    }

    async function isValidRecordingBlob(blob) {
        if (!blob || blob.size < 4096) {
            return false;
        }

        if (!blob.type.includes("webm")) {
            return true;
        }

        const header = new Uint8Array(await blob.slice(0, 4).arrayBuffer());
        return header[0] === 0x1a && header[1] === 0x45 && header[2] === 0xdf && header[3] === 0xa3;
    }

    async function uploadRecording(blob, filename) {
        const uploadUrl = recorderPanel?.dataset.uploadUrl || window.location.pathname;
        const formData = new FormData();
        formData.append("audio", blob, filename);

        setRecorderStatus("Uploading for analysis...", "uploading");

        const response = await fetch(uploadUrl, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error("Upload failed. Please try again.");
        }

        const html = await response.text();
        document.open();
        document.write(html);
        document.close();
    }

    async function startRecording() {
        if (!recorderPanel) {
            return;
        }

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.MediaRecorder) {
            setRecorderStatus("Recording is not supported in this browser.", "error");
            return;
        }

        try {
            recordingStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                },
            });

            const mimeType = getSupportedMimeType();
            mediaRecorder = new MediaRecorder(
                recordingStream,
                mimeType ? { mimeType } : undefined,
            );
            recordingChunks = [];

            audioContext = new AudioContext();
            const source = audioContext.createMediaStreamSource(recordingStream);
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 2048;
            source.connect(analyser);

            mediaRecorder.addEventListener("dataavailable", (event) => {
                if (event.data && event.data.size > 0) {
                    recordingChunks.push(event.data);
                }
            });

            mediaRecorder.addEventListener("stop", async () => {
                stopTimer();
                stopWaveform();
                recordingStream?.getTracks().forEach((track) => track.stop());

                const elapsed = recordingStartedAt ? (Date.now() - recordingStartedAt) / 1000 : 0;
                const type = mediaRecorder.mimeType || "audio/webm";
                const extension = type.includes("ogg") ? "ogg" : "webm";
                const blob = new Blob(recordingChunks, { type });
                const filename = `raaga-live-${Date.now()}.${extension}`;

                if (elapsed < 1.5 || !(await isValidRecordingBlob(blob))) {
                    setRecorderStatus("Recording was too short or incomplete. Please try again.", "error");
                    if (startRecordingBtn) startRecordingBtn.disabled = false;
                    if (stopRecordingBtn) stopRecordingBtn.disabled = true;
                    return;
                }

                const recordedUrl = URL.createObjectURL(blob);
                if (recordedPlayback) {
                    recordedPlayback.src = recordedUrl;
                    recordedPlayback.hidden = false;
                }
                if (recordingDownload) {
                    recordingDownload.href = recordedUrl;
                    recordingDownload.download = filename;
                    recordingDownload.hidden = false;
                }

                setRecorderStatus("Recording saved. Sending to analysis...", "saved");

                try {
                    await uploadRecording(blob, filename);
                } catch (error) {
                    setRecorderStatus(error.message, "error");
                    if (startRecordingBtn) startRecordingBtn.disabled = false;
                    if (stopRecordingBtn) stopRecordingBtn.disabled = true;
                }
            }, { once: true });

            mediaRecorder.start();
            startRecordingBtn.disabled = true;
            stopRecordingBtn.disabled = false;
            if (recordedPlayback) recordedPlayback.hidden = true;
            if (recordingDownload) recordingDownload.hidden = true;
            setRecorderStatus("Recording...", "recording");
            startTimer();
            drawLiveWaveform();
        } catch (error) {
            setRecorderStatus("Microphone permission denied or unavailable.", "error");
            console.error(error);
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== "inactive") {
            setRecorderStatus("Finalizing recording...", "saving");
            mediaRecorder.stop();
        }

        if (startRecordingBtn) startRecordingBtn.disabled = true;
        if (stopRecordingBtn) stopRecordingBtn.disabled = true;
    }

    if (recorderPanel && recorderPanel.dataset.bound !== "true") {
        recorderPanel.dataset.bound = "true";
        drawIdleWaveform();
        startRecordingBtn?.addEventListener("click", startRecording);
        stopRecordingBtn?.addEventListener("click", stopRecording);
    }
})();
