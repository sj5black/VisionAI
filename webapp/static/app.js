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

function renderTable(objects, isPipeline) {
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

  wrap.innerHTML = `
    <table>
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
      renderTable(data.objects, isPipeline);

      const notice = $("animalNotice");
      const pipelineInfo = $("pipelineInfo");
      
      if (isPipeline) {
        // 🆕 VisionAI Pipeline 사용
        notice.style.display = "block";
        notice.innerHTML =
          "<b>🆕 VisionAI Pipeline</b>을 사용한 분석 결과입니다. " +
          "YOLOv8 기반 객체 탐지 + MobileNetV3 감정/자세 분석 + 행동 예측 (~10MB 경량 모델)";
        
        pipelineInfo.style.display = "block";
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
          "<b>동물 행동/표정 분석</b> 기능이 현재 서버에서 비활성화되어 있습니다. " +
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
});

