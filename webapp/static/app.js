function $(id) {
  return document.getElementById(id);
}

function setStatus(text) {
  $("status").textContent = text || "";
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderTypes(types) {
  const el = $("types");
  el.innerHTML = "";
  if (!types || types.length === 0) {
    el.innerHTML = '<span class="muted">탐지된 객체가 없습니다. (임계값을 낮춰보세요)</span>';
    return;
  }
  for (const t of types) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = t;
    el.appendChild(chip);
  }
}

function renderTable(objects, isPipeline, emotionBackend) {
  const wrap = $("tableWrap");
  if (!objects || objects.length === 0) {
    wrap.innerHTML = '<div class="muted">탐지 결과가 없습니다.</div>';
    return;
  }

  const rows = objects
    .map((o, idx) => {
      const box = Array.isArray(o.box_xyxy) ? o.box_xyxy.map((v) => Number(v).toFixed(1)).join(", ") : "";
      
      // 🆕 VisionAI Pipeline 결과
      const pi = o.pipeline_insights || null;
      if (isPipeline && pi) {
        const emotion = pi.emotion ? `${escapeHtml(pi.emotion)} (${Number(pi.emotion_confidence || 0).toFixed(2)})` : "-";
        const pose = pi.pose ? `${escapeHtml(pi.pose)} (${Number(pi.pose_confidence || 0).toFixed(2)})` : "-";
        const state = pi.combined_state ? escapeHtml(pi.combined_state) : "-";
        const predicted = pi.predicted_action ? `${escapeHtml(pi.predicted_action)} (${Number(pi.prediction_confidence || 0).toFixed(2)})` : "-";
        return `
          <tr>
            <td>${idx + 1}</td>
            <td>${escapeHtml(o.label)}</td>
            <td>${Number(o.score).toFixed(3)}</td>
            <td class="mono">[${box}]</td>
            <td>${emotion}</td>
            <td>${pose}</td>
            <td>${state}</td>
            <td>${predicted}</td>
          </tr>
        `;
      }
      
      // 기존 animal_insights
      const ai = o.animal_insights || null;
      const behavior = ai && ai.behavior ? `${escapeHtml(ai.behavior)} (${Number(ai.behavior_confidence || 0).toFixed(2)})` : "-";
      const expr = ai && ai.expression ? `${escapeHtml(ai.expression)} (${Number(ai.expression_confidence || 0).toFixed(2)})` : "-";
      const state = ai && ai.estimated_state ? escapeHtml(ai.estimated_state) : "-";
      const next = ai && Array.isArray(ai.predicted_next_actions) ? ai.predicted_next_actions.map(escapeHtml).join(", ") : "-";
      return `
        <tr>
          <td>${idx + 1}</td>
          <td>${escapeHtml(o.label)}</td>
          <td>${Number(o.score).toFixed(3)}</td>
          <td class="mono">[${box}]</td>
          <td>${behavior}</td>
          <td>${expr}</td>
          <td>${state}</td>
          <td>${next}</td>
        </tr>
      `;
    })
    .join("");

  // 🆕 Pipeline 모드면 컬럼명 변경
  const headers = isPipeline
    ? `<th>emotion*</th><th>pose*</th><th>state*</th><th>predicted next*</th>`
    : `<th>behavior*</th><th>expression*</th><th>state*</th><th>next actions*</th>`;

  const caption =
    isPipeline && emotionBackend && String(emotionBackend).trim()
      ? `<caption class="table-caption">감정/자세 백엔드: <strong>${escapeHtml(emotionBackend)}</strong></caption>`
      : "";

  wrap.innerHTML = `
    <table>
      ${caption}
      <thead>
        <tr>
          <th>#</th>
          <th>label</th>
          <th>score</th>
          <th>box_xyxy</th>
          ${headers}
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

async function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

document.addEventListener("DOMContentLoaded", () => {
  const form = $("uploadForm");
  const imageInput = $("image");
  const submitBtn = $("submitBtn");
  const modelSelect = $("model");
  const tabImage = $("tabImage");
  const tabVideo = $("tabVideo");
  const panelImage = $("panelImage");
  const panelVideo = $("panelVideo");
  const imageResultsSection = $("imageResultsSection");

  function showTab(tab) {
    const isImage = tab === "image";
    if (tabImage) tabImage.classList.toggle("active", isImage);
    if (tabVideo) tabVideo.classList.toggle("active", !isImage);
    if (panelImage) panelImage.style.display = isImage ? "" : "none";
    if (panelVideo) panelVideo.style.display = isImage ? "none" : "";
    if (imageResultsSection) imageResultsSection.style.display = isImage ? "" : "none";
  }
  if (tabImage) tabImage.addEventListener("click", () => showTab("image"));
  if (tabVideo) tabVideo.addEventListener("click", () => showTab("video"));

  imageInput.addEventListener("change", async () => {
    const file = imageInput.files && imageInput.files[0];
    if (!file) return;
    $("originalImg").src = await fileToDataUrl(file);
    $("annotatedImg").removeAttribute("src");
    renderTypes([]);
    $("tableWrap").innerHTML = "";
    $("pipelineInfo").style.display = "none";
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const file = imageInput.files && imageInput.files[0];
    if (!file) return;

    submitBtn.disabled = true;
    setStatus("모델 로딩/추론 중... (첫 실행은 시간이 걸릴 수 있어요)");

    try {
      const fd = new FormData(form);
      fd.set("image", file);

      const res = await fetch("/api/detect", { method: "POST", body: fd });
      let data = null;
      let text = null;
      try {
        data = await res.json();
      } catch (_) {
        // FastAPI가 500일 때 plain-text "Internal Server Error"가 올 수 있음
        text = await res.text().catch(() => null);
      }
      if (!res.ok) {
        const msg =
          (data && data.detail) ||
          (text && text.trim()) ||
          `Request failed (HTTP ${res.status})`;
        throw new Error(msg);
      }
      if (!data) throw new Error("Empty/invalid JSON response from server");

      $("annotatedImg").src = data.annotated_image_url + "?t=" + Date.now();
      renderTypes(data.object_types);
      
      // 🆕 Pipeline 모드 체크
      const isPipeline = data.pipeline_enabled || data.model === "visionai_pipeline";
      renderTable(data.objects, isPipeline, data.emotion_backend);

      const notice = $("animalNotice");
      const pipelineInfo = $("pipelineInfo");
      
      if (isPipeline) {
        // 🆕 VisionAI Pipeline 사용
        const emotionBackend = data.emotion_backend && String(data.emotion_backend).trim();
        const backendLabel = emotionBackend ? emotionBackend : "감정/자세 분석";
        notice.style.display = "block";
        notice.innerHTML =
          "<b>🆕 VisionAI Pipeline</b>을 사용한 분석 결과입니다. " +
          "YOLOv8 사람 탐지 + <b>OpenFace 2.0 (AU)</b> 표정·자세 + 행동 예측";
        
        pipelineInfo.style.display = "block";
        if (data.emotion_backend) {
          var backendEl = document.getElementById("emotionBackendLabel");
          if (backendEl) backendEl.textContent = data.emotion_backend;
          
          // 백엔드별 감정/자세 라벨 업데이트
          var emotionListEl = $("pipelineEmotionList");
          var poseListEl = $("pipelinePoseList");
          var backend = String(data.emotion_backend).toLowerCase();
          
          if (backend.includes("openclip")) {
            emotionListEl.textContent = "relaxed, happy, content, curious, alert, excited, playful, sleepy, bored, fearful, anxious, stressed, nervous, aggressive, dominant, submissive, affectionate (16종)";
            poseListEl.textContent = "sitting, standing, lying, running, jumping, walking, crouching, stretching, sleeping, eating, drinking, sniffing, grooming, playing, begging, hiding, rolling, stalking (18종)";
          } else if (backend.includes("deepface")) {
            emotionListEl.textContent = "happy, sad, angry, surprise, fear, disgust, neutral → real_smile, sad, displeased, surprised, attention, neutral (7종)";
            poseListEl.textContent = "front (기본값, DeepFace는 head pose 미지원)";
          } else if (backend.includes("openface") || backend.includes("pyfaceau")) {
            emotionListEl.textContent = "neutral, real_smile(진짜 웃음), fake_smile(가짜 웃음), focused(집중), surprised(놀람), sad, displeased(찡그림), attention (8종, AU 기반)";
            poseListEl.textContent = "front, looking_down, looking_up, looking_side (4종, head pose)";
          } else {
            emotionListEl.textContent = "-";
            poseListEl.textContent = "-";
          }
        }
        if (data.processing_time) {
          $("processingTime").textContent = Number(data.processing_time).toFixed(3);
        }
        } else if (data.animal_insights_enabled) {
        notice.style.display = "block";
        notice.innerHTML =
          "<b>*동물 행동/표정/상태</b>는 이미지 기반 <b>추정(Zero-shot)</b> 결과입니다. " +
          "정확하지 않을 수 있으며 수의학적 진단이 아닙니다.";
        pipelineInfo.style.display = "none";
      } else {
        notice.style.display = "block";
        notice.innerHTML =
          "<b>표정/자세 분석</b> 기능이 현재 서버에서 비활성화되어 있습니다. " +
          "(open_clip_torch 미설치 또는 로드 실패).";
        pipelineInfo.style.display = "none";
      }

      setStatus(`완료: ${data.objects.length}개 탐지`);
    } catch (err) {
      console.error(err);
      setStatus("실패: " + (err && err.message ? err.message : String(err)));
    } finally {
      submitBtn.disabled = false;
    }
  });

  // 영상 분석
  const videoForm = $("videoForm");
  const videoSubmitBtn = $("videoSubmitBtn");
  const videoFileInput = $("videoFile");
  function setVideoStatus(msg) {
    const el = $("videoStatus");
    if (el) el.textContent = msg;
  }
  if (videoForm && videoSubmitBtn) {
    videoForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const file = videoFileInput && videoFileInput.files && videoFileInput.files[0];
      if (!file) return;
      videoSubmitBtn.disabled = true;
      setVideoStatus("영상 분석 중... (프레임 수에 따라 시간이 걸립니다)");
      const videoResultEl = $("videoResult");
      try {
        const fd = new FormData(videoForm);
        fd.set("video", file);
        const res = await fetch("/api/analyze-video", { method: "POST", body: fd });
        let data = null;
        let text = null;
        try {
          data = await res.json();
        } catch (_) {
          text = await res.text().catch(() => null);
        }
        if (!res.ok) {
          const msg = (data && data.detail) || (text && text.trim()) || "Request failed";
          throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
        }
        if (!data || !data.video_analysis) throw new Error("Invalid response");
        const summary = data.summary || {};
        const videoPreview = $("videoPreview");
        if (videoPreview) {
          videoPreview.src = data.video_url ? data.video_url + "?t=" + Date.now() : "";
          videoPreview.load();
        }
        const moodEl = $("videoMoodSummary");
        if (moodEl) moodEl.textContent = summary.mood_summary || "—";
        const emoEl = $("videoDominantEmotion");
        if (emoEl) emoEl.textContent = summary.dominant_emotion || "—";
        const poseEl = $("videoDominantPose");
        if (poseEl) poseEl.textContent = summary.dominant_pose || "—";
        const framesEl = $("videoFramesCount");
        if (framesEl) framesEl.textContent = String(data.frames_analyzed || 0);
        const timeEl = $("videoProcessingTime");
        if (timeEl) timeEl.textContent = String(data.processing_time_sec ?? "—");
        const backendEl = $("videoBackend");
        if (backendEl) backendEl.textContent = data.emotion_backend || "—";
        const emotionCountsEl = $("videoEmotionCounts");
        if (emotionCountsEl && summary.emotion_counts) {
          const items = Object.entries(summary.emotion_counts)
            .sort((a, b) => b[1] - a[1])
            .map(([k, v]) => k + ": " + v + "회");
          emotionCountsEl.innerHTML = "<p>" + (items.length ? items.join(", ") : "—") + "</p>";
        }
        const poseCountsEl = $("videoPoseCounts");
        if (poseCountsEl && summary.pose_counts) {
          const items = Object.entries(summary.pose_counts)
            .sort((a, b) => b[1] - a[1])
            .map(([k, v]) => k + ": " + v + "회");
          poseCountsEl.innerHTML = "<p>" + (items.length ? items.join(", ") : "—") + "</p>";
        }
        const frameTableWrap = $("videoFrameTableWrap");
        if (frameTableWrap && data.frames && data.frames.length) {
          let html = '<table><caption class="table-caption">시간(초) · 표정 · 자세</caption><thead><tr><th>시간(초)</th><th>표정</th><th>자세</th></tr></thead><tbody>';
          data.frames.slice(0, 50).forEach(function (f) {
            html += "<tr><td>" + escapeHtml(String(f.timestamp)) + "</td><td>" + escapeHtml(String(f.emotion)) + "</td><td>" + escapeHtml(String(f.pose)) + "</td></tr>";
          });
          if (data.frames.length > 50) html += "<tr><td colspan=\"3\">… 외 " + (data.frames.length - 50) + "프레임</td></tr>";
          html += "</tbody></table>";
          frameTableWrap.innerHTML = html;
        }
        if (videoResultEl) videoResultEl.style.display = "block";
        setVideoStatus("완료: " + (data.frames_analyzed || 0) + "프레임 분석");
      } catch (err) {
        console.error(err);
        setVideoStatus("실패: " + (err && err.message ? err.message : String(err)));
      } finally {
        videoSubmitBtn.disabled = false;
      }
    });
  }
});

