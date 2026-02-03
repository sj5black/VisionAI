"""
VisionAI Pipeline Web API

FastAPI를 사용한 파이프라인 웹 인터페이스
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import tempfile
from pathlib import Path
import sys

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from visionai_pipeline import VisionAIPipeline

# FastAPI 앱
app = FastAPI(
    title="VisionAI Pipeline API",
    description="동물 행동 예측 파이프라인",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 파이프라인 (한 번만 초기화)
pipeline = None


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 파이프라인 초기화"""
    global pipeline
    print("VisionAI Pipeline 초기화 중...")
    pipeline = VisionAIPipeline(
        device='auto',
        enable_emotion=True,
        enable_temporal=False,  # 단일 이미지 API라 비활성화
        enable_prediction=False
    )
    print("✓ 파이프라인 준비 완료")


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "name": "VisionAI Pipeline API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "analyze": "/api/analyze",
            "health": "/api/health",
            "info": "/api/info"
        }
    }


@app.get("/api/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "pipeline_ready": pipeline is not None
    }


@app.get("/api/info")
async def get_info():
    """파이프라인 정보"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return {
        "device": pipeline.device,
        "emotion_enabled": pipeline.enable_emotion,
        "temporal_enabled": pipeline.enable_temporal,
        "prediction_enabled": pipeline.enable_prediction,
        "emotion_classes": pipeline.emotion_analyzer.EMOTION_CLASSES if pipeline.emotion_analyzer else [],
        "pose_classes": pipeline.emotion_analyzer.POSE_CLASSES if pipeline.emotion_analyzer else []
    }


@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    conf_threshold: float = 0.5,
    visualize: bool = False
):
    """
    이미지 분석
    
    Args:
        file: 업로드된 이미지 파일
        conf_threshold: 탐지 신뢰도 임계값 (0.0-1.0)
        visualize: True면 시각화된 이미지도 반환
    
    Returns:
        JSON 분석 결과
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # 이미지 읽기
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_np = np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    
    # 분석
    try:
        result = pipeline.process_image(image_np, conf_threshold=conf_threshold)
        result_dict = result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    # 시각화
    if visualize:
        try:
            vis_image = pipeline.visualize(image_np, result)
            vis_pil = Image.fromarray(vis_image)
            
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                vis_pil.save(tmp.name, format='JPEG', quality=90)
                result_dict['visualization_path'] = tmp.name
        except Exception as e:
            result_dict['visualization_error'] = str(e)
    
    return JSONResponse(content=result_dict)


@app.post("/api/analyze_batch")
async def analyze_batch(
    files: list[UploadFile] = File(...),
    conf_threshold: float = 0.5
):
    """
    여러 이미지 일괄 분석
    
    Args:
        files: 업로드된 이미지 파일 리스트
        conf_threshold: 탐지 신뢰도 임계값
    
    Returns:
        JSON 분석 결과 리스트
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    results = []
    
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            image_np = np.array(image)
            
            result = pipeline.process_image(image_np, conf_threshold=conf_threshold)
            results.append({
                'filename': file.filename,
                'result': result.to_dict(),
                'status': 'success'
            })
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e),
                'status': 'failed'
            })
    
    return JSONResponse(content={'results': results})


if __name__ == '__main__':
    import uvicorn
    
    print("""
╔════════════════════════════════════════════════════════════╗
║         VisionAI Pipeline - Web API Server                 ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )
