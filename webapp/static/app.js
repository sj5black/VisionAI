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

function renderTable(objects) {
  const wrap = $("tableWrap");
  if (!objects || objects.length === 0) {
    wrap.innerHTML = '<div class="muted">탐지 결과가 없습니다.</div>';
    return;
  }

  const rows = objects
    .map((o, idx) => {
      const box = Array.isArray(o.box_xyxy) ? o.box_xyxy.map((v) => Number(v).toFixed(1)).join(", ") : "";
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

  wrap.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>label</th>
          <th>score</th>
          <th>box_xyxy</th>
          <th>behavior*</th>
          <th>expression*</th>
          <th>state*</th>
          <th>next actions*</th>
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
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Request failed");

      $("annotatedImg").src = data.annotated_image_url + "?t=" + Date.now();
      renderTypes(data.object_types);
      renderTable(data.objects);

      const notice = $("animalNotice");
      if (data.animal_insights_enabled) {
        notice.style.display = "block";
        notice.innerHTML =
          "<b>*동물 행동/표정/상태</b>는 이미지 기반 <b>추정(Zero-shot)</b> 결과입니다. " +
          "정확하지 않을 수 있으며 수의학적 진단이 아닙니다.";
      } else {
        notice.style.display = "block";
        notice.innerHTML =
          "<b>동물 행동/표정 분석</b> 기능이 현재 서버에서 비활성화되어 있습니다. " +
          "(open_clip_torch 미설치 또는 로드 실패).";
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

